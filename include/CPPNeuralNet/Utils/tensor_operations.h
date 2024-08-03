#ifndef CPP_NN_R_TENSOR_OPERATIONS
#define CPP_NN_R_TENSOR_OPERATIONS

#include "CPPNeuralNet/Utils/utils.h"
#include "CPPNeuralNet/Utils/tensor_like.h"

#include <numeric> // to fill from 0 to n

// INCLUDING INDIVIDUAL OPERATION GROUPS
#include "CPPNeuralNet/Utils/TensorOperations/broadcast_operations.h"
#include "CPPNeuralNet/Utils/TensorOperations/tensorwise_operations.h"
// END OF INCLUDING INDIVIDUAL OPERATION GROUPS

namespace cpp_nn {
namespace util {

// Forward Declaration ======================================
namespace {
template<typename T, typename HeldOperation>
class TransposeOperation;
template<typename T, typename HeldOperation>
class MultiTransposeOperation;
template<typename T, typename HeldOperation1, typename HeldOperation2>
class SummationOperation;
template<typename T, typename HeldOperation1, typename HeldOperation2>
class MultiplicationOperation;
template<typename T, typename HeldOperation>
class ReshapeOperation;
template<typename T, typename HeldOperation>
class PaddingOperation;
template<typename T, typename HeldOperation>
class BroadcastOperation;
template<typename T, typename HeldOperation>
class ScalarMultiplicationOperation;

// Not an Operation
template<typename T, typename HeldOperation1, typename HeldOperation2>
class BroadcastedPairHolder;
} // unnamed namespace

template<typename T>
class Tensor;
// End of Forward Declaration ===============================

namespace {
/* Operation Facilitators ---------------------------------------------------------- */
/*    
  Intermediate classes to be passed to reduce movement of tensor data
  These classes will appear as return type of operations such as Tranpose and summation,
    and only upon assignment to a Tensor will data be moved.
  There are exception to these rules, such as with multiplcation which 
    may benefit from operation itself allocating space for computation and storage,
    but it too will follow the design of the other operations.

 *  ALL OPERATION VALIDITY ARE CHECKED UPON CONSTRUCT TODO:
 */

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

/**
 *  multiplies two tensors with broadcasting applied. 
 *  That is, broadcasting is applied to first order-2 dimensions. 
 *    Last two are treated as matrix.
 */
template<typename T, typename HeldOperation1, typename HeldOperation2>
class MultiplicationOperation : public TensorLike<T, MultiplicationOperation<T, HeldOperation1, HeldOperation2>> {
  typedef MultiplicationOperation<T, HeldOperation1, HeldOperation2> Self;
  typedef TensorLike<T, Self> Parent;

  typedef BroadcastOperation<T, HeldOperation1> BroadcastFirst;
  typedef BroadcastOperation<T, HeldOperation2> BroadcastSecond;

// Members ---------------------------------------------
  // product is resolved upon construct,
  //  so that other multiplcation algorithms can be used.
  Tensor<T>* product_tensor_;
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  // TODO, with this design, a special construcor for Tensor may be prefered for Multiplcation Holder, where a simple move is prefered
  // Product is handled upon constrcution
  MultiplicationOperation(const TensorLike<T, HeldOperation1>& A, 
                          const TensorLike<T, HeldOperation2>& B)
      // tensor is build after all checks
      : product_tensor_(nullptr) {
    // Multiplication possiblity check
    if (A.getOrder() < 2 || B.getOrder() < 2) {
      throw std::invalid_argument("Multiplication- Insufficient Dimension");
    }
    if (A.getDimension(-1) != B.getDimension(-2)) {
      throw std::invalid_argument("Multiplication- Multiplcation Dimension Mismatch");
    }

    // Broadcasting checks for compatiblity 
    std::vector<int> broad_cast_shape(Broadcast(A.getShape().begin(), A.getShape().end() - 2,
                                                B.getShape().begin(), B.getShape().end() - 2)); 

    const int rows = A.getDimension(-2);
    const int cols = B.getDimension(-1);
    const int interm = A.getDimension(-1);

    // !! Does not use conventional Broadcast. Need to have broadcast separaelty

    // First set product_shape
    broad_cast_shape.push_back(rows);
    broad_cast_shape.push_back(cols);

    const size_t order = broad_cast_shape.size();

    // Broadcast A
    std::vector<int> A_broadcast_shape(broad_cast_shape);
    A_broadcast_shape[order - 1] = interm;
    // TODO for this context these might be unnecessary to compute
    // we might as well pass broken chunk sizes
    std::vector<int> A_broadcast_chunk(order, 1);
    size_t A_capacity = 1;
    ComputeCapacityAndChunkSizes(A_broadcast_shape,
                                 A_broadcast_chunk,
                                 A_capacity);
    // but for now, use proper, then reevaluate when doing strassen TODO
    BroadcastFirst A_broadcast(A, 
                               A_broadcast_shape, 
                               A_broadcast_chunk,
                               A_capacity);
    // Broadcast B
    std::vector<int> B_broadcast_shape(broad_cast_shape);
    B_broadcast_shape[order - 2] = interm;
    // TODO for this context these might be unnecessary to compute
    // we might as well pass broken chunk sizes
    std::vector<int> B_broadcast_chunk(order, 1);;
    size_t B_capacity = 1;
    ComputeCapacityAndChunkSizes(B_broadcast_shape,
                                 B_broadcast_chunk,
                                 B_capacity);
    // but for now, use proper, then reevaluate when doing strassen TODO
    BroadcastSecond B_broadcast(B, 
                                B_broadcast_shape, 
                                B_broadcast_chunk,
                                B_capacity);

    // Final storage for C = A*B
    product_tensor_ = new Tensor<T>(broad_cast_shape);

    typename Tensor<T>::Iterator it = product_tensor_->begin();
    typename Tensor<T>::Iterator fin = product_tensor_->end();

    typename BroadcastFirst::ConstIterator  A_it = A_broadcast.begin();
    typename BroadcastSecond::ConstIterator B_it = B_broadcast.begin();

    while (it != fin) {
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          T& element = *it;

          element = T();

          for (int k = 0; k < interm; ++k) {
            element += (*A_it) * (*B_it);

            // Increment A to next column
            ++A_it;
            // increment to next row if only not at the last col
            if (k != interm - 1) {
              B_it += cols;
            }
          } /// k loop --

          // Reset A to start of row
          A_it -= interm;
          // rest to first row
          B_it -= cols * (interm - 1)
                  - 1; // Increment B to next col
          ++it;
        } /// c Loop --

        // Increment A to next row
        A_it += interm;
        // Reset B to first column
        B_it -= cols;
      }/// r Loop --

      // A naturally increments to next matrix by row incr.
      // B Move to next matrix
      B_it += interm * cols;
    } 
  } 
// End of Constructor ----------------------------------

// Destructor ------------------------------------------
  ~MultiplicationOperation() {
    delete product_tensor_;
  }
// End of Destructor -----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    return product_tensor_->getShape();
  } 
  inline int getDimension(int axis) const {
    return product_tensor_->getDimension(axis);
  }
  inline size_t getCapacity() const noexcept {
    return product_tensor_->getCapacity();
  }
  inline int getOrder() const noexcept {
    return product_tensor_->getOrder();
  }
  inline const T getElement(const std::vector<int>& indices) const {
    return product_tensor_->getElement(indices);
  }
// End of Tensor-Behaviours ----------------------------

// ! ConstTerator of Tensor moved to represent ConstIterator of this
  typedef typename Tensor<T>::ConstIterator ConstIterator;

  ConstIterator begin() const {
    return static_cast<const Tensor<T>*>(product_tensor_)->begin();
  }
  ConstIterator end() const {
    return  static_cast<const Tensor<T>*>(product_tensor_)->end();
  }
// friends ---------------------------------------------
  friend Tensor<T>;   // For specialized constructor
// end of friedns --------------------------------------
}; // End of MultiplicationOperation


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


#endif // CPP_NN_R_TENSOR_OPERATIONS
