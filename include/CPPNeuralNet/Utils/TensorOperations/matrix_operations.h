#ifndef CPP_NN_MATRIX_OPERATIONS
#define CPP_NN_MATRIX_OPERATIONS

#include "CPPNeuralNet/Utils/utils.h"
#include "CPPNeuralNet/Utils/tensor_like.h"

#include "CPPNeuralNet/Utils/TensorOperations/broadcast_operations.h"


namespace cpp_nn {
namespace util {

// Forward Declaration ======================================
template<typename T>
class Tensor;
// End of Forward Declaration ===============================


namespace {
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
          B_it += 1 - cols * (interm - 1); // Increment B to next col
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

} // unnamed namespace
} // util
} // cpp_nn

#endif // CPP_NN_MATRIX_OPERATIONS