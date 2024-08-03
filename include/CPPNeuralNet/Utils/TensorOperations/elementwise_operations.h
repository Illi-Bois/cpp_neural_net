/**
 * TensorOperations that handle elemenet-wise operations.
 */
#ifndef CPP_NN_ELEMENTWISE_OPERATIONS
#define CPP_NN_ELEMENTWISE_OPERATIONS

#include "CPPNeuralNet/Utils/utils.h"
#include "CPPNeuralNet/Utils/tensor_like.h"

#include "CPPNeuralNet/Utils/TensorOperations/broadcast_operations.h"

namespace cpp_nn {
namespace util {
namespace {
/**
 *  sums two tensors with broadcasting applied. 
 */
template<typename T, typename HeldOperation1, typename HeldOperation2>
class SummationOperation : public TensorLike<T, SummationOperation<T, HeldOperation1, HeldOperation2>> {
  typedef SummationOperation<T, HeldOperation1, HeldOperation2> Self;
  typedef TensorLike<T, Self> Parent;

// Members ---------------------------------------------
    // const std::vector<int> broadcast_shape_;
    // const size_t broadcast_capacity_;

    // const BroadcastOperation<T, HeldOperation1> first_;
    // const BroadcastOperation<T, HeldOperation2> second_;
  BroadcastedPairHolder<T, HeldOperation1, HeldOperation2> broadcast_pair_;
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  SummationOperation(const TensorLike<T, HeldOperation1>& first, 
                     const TensorLike<T, HeldOperation2>& second) 
      // Broadcasting also checks for compatiblity.
      : broadcast_pair_(first, second) {}
// End of Constructor ----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    return broadcast_pair_.getShape();
  } 
  inline int getDimension(int axis) const {
    return getShape()[SumIfNegative(axis, getOrder())];
  }
  inline size_t getCapacity() const noexcept {
    return broadcast_pair_.getCapacity();
  }
  inline int getOrder() const noexcept {
    return getShape().size();
  }
  inline const T getElement(const std::vector<int>& indices) const {
    return broadcast_pair_.getFirst().getElement(indices) +
           broadcast_pair_.getSecond().getElement(indices);
  }
// End of Tensor-Behaviours ----------------------------

  class ConstIterator : public Parent::template ConstIterator<ConstIterator> {
    typedef typename BroadcastOperation<T, HeldOperation1>::ConstIterator BroadcastIterator1;
    typedef typename BroadcastOperation<T, HeldOperation2>::ConstIterator BroadcastIterator2;
    BroadcastIterator1 first_it_;
    BroadcastIterator2 second_it_;
   public:
    ConstIterator(BroadcastIterator1 first_it, 
                  BroadcastIterator2 second_it)
        : first_it_(first_it),
          second_it_(second_it) {}

    T operator*() const {
      return *first_it_ + *second_it_;
    }

    ConstIterator& operator+=(int increment) {
      first_it_ += increment;
      second_it_ += increment;
      return *this;
    }
    ConstIterator& operator-=(int decrement) {
      first_it_ -= decrement;
      second_it_ -= decrement;
      return *this;
    }

    bool operator==(const ConstIterator& other) const {
      return first_it_ == other.first_it_ && 
             second_it_ == other.second_it_;
    }
  };

  ConstIterator begin() const {
    return {broadcast_pair_.getFirst().begin(),
            broadcast_pair_.getSecond().begin()};
  }
  ConstIterator end() const {
    return {broadcast_pair_.getFirst().end(), 
            broadcast_pair_.getSecond().end()};
  }
}; // End of SummationOperation


/**
 *  subtracts two tensors with broadcasting applied. 
 */
template<typename T, typename HeldOperation1, typename HeldOperation2>
class SubtractionOperation : public TensorLike<T, SubtractionOperation<T, HeldOperation1, HeldOperation2>> {
  typedef SubtractionOperation<T, HeldOperation1, HeldOperation2> Self;
  typedef TensorLike<T, Self> Parent;

// Members ---------------------------------------------
    // const std::vector<int> broadcast_shape_;
    // const size_t broadcast_capacity_;

    // const BroadcastOperation<T, HeldOperation1> first_;
    // const BroadcastOperation<T, HeldOperation2> second_;
  BroadcastedPairHolder<T, HeldOperation1, HeldOperation2> broadcast_pair_;
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  SubtractionOperation(const TensorLike<T, HeldOperation1>& first, 
                       const TensorLike<T, HeldOperation2>& second) 
      // Broadcasting also checks for compatiblity.
      : broadcast_pair_(first, second) {}
// End of Constructor ----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    return broadcast_pair_.getShape();
  } 
  inline int getDimension(int axis) const {
    return getShape()[SumIfNegative(axis, getOrder())];
  }
  inline size_t getCapacity() const noexcept {
    return broadcast_pair_.getCapacity();
  }
  inline int getOrder() const noexcept {
    return getShape().size();
  }
  inline const T getElement(const std::vector<int>& indices) const {
    return broadcast_pair_.getFirst().getElement(indices) -
           broadcast_pair_.getSecond().getElement(indices);
  }
// End of Tensor-Behaviours ----------------------------

  class ConstIterator : public Parent::template ConstIterator<ConstIterator> {
    typedef typename BroadcastOperation<T, HeldOperation1>::ConstIterator BroadcastIterator1;
    typedef typename BroadcastOperation<T, HeldOperation2>::ConstIterator BroadcastIterator2;
    BroadcastIterator1 first_it_;
    BroadcastIterator2 second_it_;
   public:
    ConstIterator(BroadcastIterator1 first_it, 
                  BroadcastIterator2 second_it)
        : first_it_(first_it),
          second_it_(second_it) {}

    T operator*() const {
      return *first_it_ - *second_it_;
    }

    ConstIterator& operator+=(int increment) {
      first_it_ += increment;
      second_it_ += increment;
      return *this;
    }
    ConstIterator& operator-=(int decrement) {
      first_it_ -= decrement;
      second_it_ -= decrement;
      return *this;
    }

    bool operator==(const ConstIterator& other) const {
      return first_it_ == other.first_it_ && 
             second_it_ == other.second_it_;
    }
  };

  ConstIterator begin() const {
    return {broadcast_pair_.getFirst().begin(),
            broadcast_pair_.getSecond().begin()};
  }
  ConstIterator end() const {
    return {broadcast_pair_.getFirst().end(), 
            broadcast_pair_.getSecond().end()};
  }
}; // End of SubtractionOperation

template<typename T, typename HeldOperation>
class ScalarMultiplicationOperation : public TensorLike<T, ScalarMultiplicationOperation<T, HeldOperation>> {
  typedef ScalarMultiplicationOperation<T, HeldOperation> Self;
  typedef TensorLike<T, Self> Parent;

  const HeldOperation& tensor_like_;
  const T scalar_;

public:
  ScalarMultiplicationOperation(const TensorLike<T, HeldOperation>& A,
                                T scalar)
    : tensor_like_(A.getRef()),
      scalar_(scalar) {};
  inline const std::vector<int>& getShape() const noexcept {
    return tensor_like_.getShape();
  } 
  inline int getDimension(int axis) const {
    return tensor_like_.getDimension(axis);
  }
  inline size_t getCapacity() const noexcept {
    return tensor_like_.getCapacity();
  }
  inline int getOrder() const noexcept {
    return getShape().size();
  }
  inline const T getElement(const std::vector<int>& indicies) const {
    return tensor_like_.getElement(indicies) * scalar_;
  }

  typedef typename HeldOperation::ConstIterator HeldIterator;

  class ConstIterator : public Parent::template ConstIterator<ConstIterator> {
  // Members ---------------------------------------------
    HeldIterator it_;
    T scalar_;
  // End of Members --------------------------------------
   public:
   ConstIterator(HeldIterator it, T scalar) noexcept
      : it_(it), scalar_(scalar) {}
    
    T operator*() const {
      return *it_ * scalar_;
    }

    ConstIterator& operator+=(int increment) {
      it_ += increment;
      return *this;
    }
    ConstIterator& operator-=(int decrement) {
      it_ -= decrement;
      return *this;
    }
    bool operator==(const ConstIterator& other) const {
      return it_ == other.other;
    }
  }; // End of ConstIterator

  ConstIterator begin() const {
    return {tensor_like_.begin(), scalar_}; 
  }
  ConstIterator end() const {
    return {tensor_like_.end(), scalar_}; 
  }
};

} // unnamed namespace
} // util
} // cpp_nn

#endif // CPP_NN_ELEMENTWISE_OPERATIONS