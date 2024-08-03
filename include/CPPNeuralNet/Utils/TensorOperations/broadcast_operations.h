#ifndef CPP_NN_BROADCAST_OPERATIONS
#define CPP_NN_BROADCAST_OPERATIONS

#include "CPPNeuralNet/Utils/utils.h"
#include "CPPNeuralNet/Utils/tensor_like.h"

namespace cpp_nn {
namespace util {

namespace {

template<typename T, typename HeldOperation>
class BroadcastOperation : public TensorLike<T, BroadcastOperation<T, HeldOperation>> {
  typedef BroadcastOperation<T, HeldOperation> Self;
  typedef TensorLike<T, Self> Parent;

// Members ---------------------------------------------
  const HeldOperation& tensor_like_;

  const std::vector<int>& broadcasted_shape_;
  // To be compatible with BroadcastPairHolder, must be reference
  const std::vector<int>& broadcasted_chunk_sizes_;
  const size_t& broadcasted_capacity_;
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  // passed broadcast shape is assumed to be validly formed, and so does no checks in construction
  BroadcastOperation(const TensorLike<T, HeldOperation>& tensor_like, 
                     const std::vector<int>& broadcasted_shape,
                     const std::vector<int>& broadcasted_chunk_sizes,
                     const size_t& broadcasted_capacity) noexcept
      : tensor_like_(tensor_like.getRef()),
        broadcasted_shape_(broadcasted_shape),
        broadcasted_chunk_sizes_(broadcasted_chunk_sizes),
        broadcasted_capacity_(broadcasted_capacity)  {}
// End of Constructor ----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    return broadcasted_shape_;
  } 
  inline int getDimension(int axis) const {
    return broadcasted_shape_[SumIfNegative(axis, getOrder())];
  }
  inline size_t getCapacity() const noexcept {
    return broadcasted_capacity_;
  }
  inline int getOrder() const noexcept {
    return broadcasted_shape_.size();
  }
  const T getElement(const std::vector<int>& indices) const {
    return tensor_like_.getElement(CutToShape(indices, tensor_like_.getShape()));
  }
// End of Tensor-Behaviours ----------------------------

  typedef typename HeldOperation::ConstIterator HeldIterator;

  class ConstIterator : public Parent::template ConstIterator<ConstIterator> {
    const Self* const reference_;
    HeldIterator it_;

    size_t curr_address_;

    size_t original_capacity_;

    // To be constructed in const. body
    std::vector<int> detected_dimensions_;
    std::vector<int> detected_chunk_sizes_;

    size_t curr_inner_address_;
   public:
    ConstIterator(const Self* const ref, 
         HeldIterator it,
         size_t address) 
        : reference_(ref),
          it_(it),
          curr_address_(address),
          original_capacity_(reference_->tensor_like_.getCapacity()) {
      DetectBroadcastAxes(reference_->broadcasted_shape_,
                          reference_->broadcasted_chunk_sizes_,
                          reference_->tensor_like_.getShape(),
                          detected_dimensions_,
                          detected_chunk_sizes_);
      
      curr_inner_address_ = ConvertToUnbroadcastAddress(detected_dimensions_,
                                                        detected_chunk_sizes_,
                                                        original_capacity_,
                                                        curr_address_);
    }

    T operator*() const {
      return *it_;
    }
    
    ConstIterator& operator+=(int increment) {
      curr_address_ += increment;
      if (curr_address_ >= reference_->getCapacity()) {
        curr_address_ = reference_->getCapacity();
      }

      size_t new_inner_address = ConvertToUnbroadcastAddress(detected_dimensions_,
                                                             detected_chunk_sizes_,
                                                             original_capacity_,
                                                             curr_address_);
      
      if (new_inner_address > curr_inner_address_) {
        size_t diff = new_inner_address - curr_inner_address_;
        it_ += diff;
      } else {
        size_t diff = curr_inner_address_ - new_inner_address;
        it_ -= diff;
      }
      curr_inner_address_ = new_inner_address;

      return *this;
    }
    ConstIterator& operator-=(int decrement) {
      if (decrement >= curr_address_) {
        curr_address_ = 0;
      } else {
        curr_address_ -= decrement;
      }

      size_t new_inner_address = ConvertToUnbroadcastAddress(detected_dimensions_,
                                                             detected_chunk_sizes_,
                                                             original_capacity_,
                                                             curr_address_);
      
      if (new_inner_address > curr_inner_address_) {
        size_t diff = new_inner_address - curr_inner_address_;
        it_ += diff;
      } else {
        size_t diff = curr_inner_address_ - new_inner_address;
        it_ -= diff;
      }
      curr_inner_address_ = new_inner_address;

      return *this;
    }

    bool operator==(const ConstIterator& other) const {
      return reference_ == other.reference_ && 
              curr_address_ == other.curr_address_;
    }

  };

  ConstIterator begin() const {
    return {this, tensor_like_.begin(), 0};
  }
  ConstIterator end() const {
    return {this, tensor_like_.begin(), getCapacity()};
  }
}; // End of BroadcastOperation


// NOT A OPERATIONHOLDER, but instead a holder for operations
// internally holds two broadcastOperations
template<typename T, typename HeldOperation1, typename HeldOperation2>
class BroadcastedPairHolder {
  typedef BroadcastOperation<T, HeldOperation1> BroadcastFirst;
  typedef BroadcastOperation<T, HeldOperation2> BroadcastSecond;

  std::vector<int> broadcasted_shape_;

  // WILL PASS AS REFERENCE FIRST, THEN SET IN CONSTRUCT
  std::vector<int> broadcasted_chunk_sizes_;
  size_t broadcasted_capacity_;

  BroadcastFirst first_;
  BroadcastSecond second_;

  // above are to be shared to broadcast operation
  // as those will store reference to this only
 public:
  BroadcastedPairHolder(const TensorLike<T, HeldOperation1>& first_operation,
                        const TensorLike<T, HeldOperation2>& second_operation)
      : broadcasted_shape_(Broadcast(first_operation.getShape().begin(),  first_operation.getShape().end(),
                                     second_operation.getShape().begin(), second_operation.getShape().end())),
        broadcasted_chunk_sizes_(broadcasted_shape_.size(), 1),
        broadcasted_capacity_(1),
        // The broadcasts are passed unset vector and capacity, because they will now stroe them as rference
        first_ (first_operation,  broadcasted_shape_, broadcasted_chunk_sizes_, broadcasted_capacity_),
        second_(second_operation, broadcasted_shape_, broadcasted_chunk_sizes_, broadcasted_capacity_) {
    ComputeCapacityAndChunkSizes(broadcasted_shape_,
                                 broadcasted_chunk_sizes_,
                                 broadcasted_capacity_);
  }

  inline const BroadcastFirst& getFirst() const noexcept {
    return first_;
  }
  inline const BroadcastSecond& getSecond() const noexcept {
    return second_;
  }

  inline const std::vector<int>& getShape() const noexcept {
    return broadcasted_shape_;
  }
  inline size_t getCapacity() const noexcept {
    return broadcasted_capacity_;
  }
}; // End of BroadcastedPairHolder

} // unnamed namespace

} // util
} // cpp_nn

#endif // CPP_NN_BROADCAST_OPERATIONS