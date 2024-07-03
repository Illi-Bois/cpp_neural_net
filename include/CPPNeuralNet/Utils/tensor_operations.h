#ifndef CPP_NN_R_TENSOR_OPERATIONS
#define CPP_NN_R_TENSOR_OPERATIONS

#include "CPPNeuralNet/Utils/utils.h"
#include "CPPNeuralNet/Utils/tensor_like.h"


namespace cpp_nn {
namespace util {

// Forward Declaration ======================================
namespace {
template<typename T, typename Derived>
class TensorLike; 
template<typename T, typename HeldOperation>
class TransposeOperation;
template<typename T, typename HeldOperation1, typename HeldOperation2>
class SummationOperation;
template<typename T, typename HeldOperation1, typename HeldOperation2>
class MultiplicationOperation;
template<typename T, typename HeldOperation>
class ReshapeOperation;
} // unnamed namespace

template<typename T>
class rTensor;
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
 */

template<typename T, typename HeldOperation>
class TransposeOperation : public TensorLike<T, TransposeOperation<T, HeldOperation>> {
// Members ---------------------------------------------
  const HeldOperation& tensor_;
  const int axis_1_;
  const int axis_2_;

  std::vector<int> dimension_;    // dimension post-tranposing
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  TransposeOperation(const TensorLike<T, HeldOperation>& tensor_like, 
                     const int axis_1, const int axis_2)
      : tensor_(tensor_like.getRef()), 
        // axis are stored as positive int, though it can be given as negative
        axis_1_(SumIfNegative(axis_1, tensor_.getOrder())), 
        axis_2_(SumIfNegative(axis_2, tensor_.getOrder())), 
        dimension_(tensor_like.getShape()) {
    std::swap(dimension_[axis_1_], dimension_[axis_2_]);
  }
// End of Constructor ----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const {
    return dimension_;
  }
  inline int getDimension(int axis) const {
    return dimension_[SumIfNegative(axis, getOrder())];
  }
  inline size_t getCapacity() const {
    return tensor_.getCapacity();
  }
  inline int getOrder() const noexcept {
    return dimension_.size();
  }  
  // indices for tranpose is passed as value, not reference,
  //   so that tranposed axis may be swapped. 
  const T getElement(std::vector<int> indices) const {
    std::swap(indices[axis_1_], indices[axis_2_]);
    return tensor_.getElement(indices);
  }
  inline const TransposeOperation<T, TransposeOperation<T, HeldOperation>> 
               Tranpose(int axis_1, int axis_2) const {
    return {*this, axis_1, axis_2};
  }
  inline const ReshapeOperation<T, TransposeOperation<T, HeldOperation>> 
               ExternalReshape(const std::vector<int>& new_dimensions) const {
    return {*this, new_dimensions};
  }
// End of Tensor-Behaviours ----------------------------
}; // End of TransposeOperation


// TODO: Currently only handles same-shape addition.
template<typename T, typename HeldOperation1, typename HeldOperation2>
class SummationOperation : public TensorLike<T, SummationOperation<T, HeldOperation1, HeldOperation2>> {
// Members ---------------------------------------------
    const HeldOperation1& first_;
    const HeldOperation2& second_;
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  SummationOperation(const TensorLike<T, HeldOperation1>& first, 
                     const TensorLike<T, HeldOperation2>& second) 
      : first_(first.getRef()), 
        second_(second.getRef()) {
    if (first_.getOrder() != second_.getOrder()) {
      throw std::invalid_argument("Summation Error- Order Mismatch.");
    }
    if (first_.getShape() != second_.getShape()) {
      throw std::invalid_argument("Summation Error- Dimension Mismatch.");
    }
  }
// End of Constructor ----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    // Either shapes must be equal
    return first_.getShape();
  } 
  inline int getDimension(int axis) const {
    // Either shapes must be equal
    return first_.getDimension(axis);
  }
  inline size_t getCapacity() const {
    // either shapes must be equal
    return first_.getCapacity();
  }
  inline int getOrder() const noexcept {
    // Either shapes must be equal
    return first_.getOrder();
  }
  inline const T getElement(const std::vector<int>& indices) const {
    return first_.getElement(indices) + second_.getElement(indices);
  }
  inline const TransposeOperation<T, SummationOperation<T, HeldOperation1, HeldOperation2>> 
               Transpose(int axis1 = -2, int axis2 = -1) const {
    return {*this, axis1, axis2};
  }
  inline const ReshapeOperation<T, SummationOperation<T, HeldOperation1, HeldOperation2>> 
               ExternalReshape(const std::vector<int>& new_dimensions) const {
    return {*this, new_dimensions};
  }
// End of Tensor-Behaviours ----------------------------
}; // End of SummationOperation

// TODO: Currently handles same-shape multiplication
//    will scale up to allow broad-casting (not all combination product)
template<typename T, typename HeldOperation1, typename HeldOperation2>
class MultiplicationOperation : public TensorLike<T, MultiplicationOperation<T, HeldOperation1, HeldOperation2>> {
// Members ---------------------------------------------
  rTensor<T>* element_;
  std::vector<int> product_shape_; 
// End of Members --------------------------------------

// Housekeeping Methods --------------------------------
/** 
 *  increments given index to iterate through each matrix in tensor.
 *  Given index can be any length, only the frist order - 2 will be used. 
 *    This means last two axis are preserved, when full length index is given.
 * 
 *  Returns true when incrementation was successful, false when failed.
 *    Upon failure, the used indices are set to 0. 
 */
  bool MatrixIncrementIndices(std::vector<int>& indices) {
    for (int axis = product_shape_.size() - 2; axis >= 0; --axis) {
      if (++indices[axis] >= product_shape_[axis]) {
        indices[axis] = 0;
      } else {
        return true;
      }
    }
    return false;
  }
  // TODO: Merge this into incrementIndices method
// End of Housekeeping Methods -------------------------

 public:
// Constructor -----------------------------------------
  // TODO, with this design, a special construcor for rTensor may be prefered for Multiplcation Holder, where a simple move is prefered
  // Product is handled upon constrcution
  MultiplicationOperation(const TensorLike<T, HeldOperation1>& A, 
                          const TensorLike<T, HeldOperation2>& B)
      : element_(nullptr),
        product_shape_(A.getShape()) /* dummy for later*/ {
    // assert  [dim1,  r, k] [dim2, k, c] that dim == dim1
    if (A.getOrder() != B.getOrder()) {
      throw std::invalid_argument("Multiplcation- Order Mismatch");
    }
    if (A.getOrder() < 2) {
      throw std::invalid_argument("Multiplication- Insufficient Dimension");
    }
    if (!std::equal(A.getShape().begin(), A.getShape().end() - 2, 
                    B.getShape().begin())) {
      throw std::invalid_argument("Multiplication- Dimension Mismatch");
    }
    if (A.getDimension(-1) != B.getDimension(-2)) {
      throw std::invalid_argument("Multiplication- Multiplcation Dimension Mismatch");
    }

    const int rows = A.getDimension(-2);
    const int cols = B.getDimension(-1);
    const int interm = A.getDimension(-1);

    // product_shape is aleady set as being A
    product_shape_[getOrder() - 1] = cols;

    element_ = new rTensor<T>(product_shape_);
    // indices for which matrix is to be chosen
    std::vector<int> indices(getOrder(), 0);

    do {
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          indices[getOrder() - 2] = r;
          indices[getOrder() - 1] = c;

          T& element = element_->getElement(indices);
          element = T();

          for (int k = 0; k < interm; ++k) {
            indices[getOrder() - 2] = r;
            indices[getOrder() - 1] = k;
            T a_element = A.getElement(indices);

            indices[getOrder() - 2] = k;
            indices[getOrder() - 1] = c;
            T b_element = B.getElement(indices);

            element += a_element * b_element;
          }
        }
      }
    } while (IncrementIndicesByShape(product_shape_.begin(), product_shape_.end() - 2,
                                     indices.begin(),        indices.end() - 2));
  }
// End of Constructor ----------------------------------

// Destructor ------------------------------------------
  ~MultiplicationOperation() {
    delete element_;
  }
// End of Destructor -----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    return product_shape_;
  } 
  inline int getDimension(int axis) const {
    return product_shape_[SumIfNegative(axis, getOrder())];
  }
  inline size_t getCapacity() const {
    // either shapes must be equal
    return element_->getCapacity();
  }
  inline int getOrder() const noexcept {
    return product_shape_.size();
  }
  inline const T getElement(const std::vector<int>& indices) const {
    return element_->getElement(indices);
  }
  inline const TransposeOperation<T, MultiplicationOperation<T, HeldOperation1, HeldOperation2>> 
               Transpose(int axis1 = -2, int axis2 = -1) const {
    return {*this, axis1, axis2};
  }
  inline const ReshapeOperation<T, MultiplicationOperation<T, HeldOperation1, HeldOperation2>> 
               ExternalReshape(const std::vector<int>& new_dimensions) const {
    return {*this, new_dimensions};
  }
// End of Tensor-Behaviours ----------------------------

// friends ---------------------------------------------
  friend rTensor<T>;  // required for constrctor
// end of friedns --------------------------------------
}; // End of MultiplicationOperation


template<typename T, typename HeldOperation>
class ReshapeOperation : public TensorLike<T, ReshapeOperation<T, HeldOperation>> {
// Members ---------------------------------------------
  const HeldOperation& tensor_like_;
  std::vector<int> dimension_;    // dimension post-tranposing

  std::vector<int> chunk_size_; // to be computed
  size_t capacity_;
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  ReshapeOperation(const TensorLike<T, HeldOperation>& tensor_like, 
                   const std::vector<int>& new_shape)
      : tensor_like_(tensor_like.getRef()),
        dimension_(new_shape),
        chunk_size_(new_shape.size(), 1),
        capacity_(1) {
    cpp_nn::util::ComputeCapacityAndChunkSizes(dimension_, chunk_size_, capacity_);

    if (capacity_ != tensor_like_.getCapacity()) {
      throw std::invalid_argument("TensorReshape- Capacity mismatch");
    }

    // it may be faster to simply move everyhting? make a new tensor and access it through that?
  }

  // Remove Nested Reshapes
  ReshapeOperation(const ReshapeOperation<T, HeldOperation>& reshape_operation, 
                   const std::vector<int>& new_shape)
      : tensor_like_(reshape_operation.tensor_like_),
        dimension_(new_shape),
        chunk_size_(new_shape.size(), 1),
        capacity_(1) {
    cpp_nn::util::ComputeCapacityAndChunkSizes(dimension_, chunk_size_, capacity_);

    if (capacity_ != tensor_like_.getCapacity()) {
      throw std::invalid_argument("TensorReshape- Capacity mismatch");
    }

    // it may be faster to simply move everyhting? make a new tensor and access it through that?
  }
// End of Constructor ----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    // Either shapes must be equal
    return dimension_;
  } 
  inline int getDimension(int axis) const {
    // Either shapes must be equal
    return dimension_[SumIfNegative(axis, getOrder())];
  }
  inline size_t getCapacity() const {
    // either shapes must be equal
    return capacity_;
  }
  inline int getOrder() const noexcept {
    // Either shapes must be equal
    return dimension_.size();
  }
  inline const T getElement(const std::vector<int>& indices) const {
    // need to recompute index to new shape, ie shape of old

    // TODO: Refactor this to reduce duplicate code
    if (indices.size() != getOrder()) {
      throw std::invalid_argument("Reshape getter- Incorrect order");
    }

    int address = 0;
    int curr_idx,
        curr_dim;
    // as dimension accepts int, we should keep axis as int as well
    for (int axis = getOrder() - 1; axis >= 0; --axis) {
      curr_idx = indices[axis];
      curr_dim = getDimension(axis);

      if (curr_idx >= -curr_dim && curr_idx < curr_dim) {
        address += SumIfNegative(curr_idx, curr_dim) * chunk_size_[axis];
      } else {
        throw std::invalid_argument("Reshape getter- Incorrect order");
      }
    } 

    // convert address to old index shape
    std::vector<int> old_indices = AddressToIndices(tensor_like_.getShape(), address);

    return tensor_like_.getElement(old_indices);
  }
  inline const TransposeOperation<T, ReshapeOperation<T, HeldOperation>> 
               Transpose(int axis1 = -2, int axis2 = -1) const {
    return {*this, axis1, axis2};
  }
  inline const ReshapeOperation<T, ReshapeOperation<T, HeldOperation>> 
               ExternalReshape(const std::vector<int>& new_dimensions) const {
    return {*this, new_dimensions};
  }
// End of Tensor-Behaviours ----------------------------

/*
 when moving, will move tensor_ to recieving tensor,
  then compute dimension again
*/
};

// BROADCASTING MIGHT BE DONE IN THIS MANNER AS WELL!

} // unnamed namespace 



// Operations =================================================
/**
 *  returns summation holder.
 */
template<typename T, typename HeldOperation1, 
                     typename HeldOperation2> 
SummationOperation<T, HeldOperation1, HeldOperation2> operator+(const TensorLike<T, HeldOperation1>& A, 
                                                                const TensorLike<T, HeldOperation2>& B) {
  return {A, B};
}
/**
 *  returns multiplication holder.
 */
template<typename T, typename HeldOperation1, 
                     typename HeldOperation2>
MultiplicationOperation<T, HeldOperation1, HeldOperation2> operator*(const TensorLike<T, HeldOperation1>& A,
                                                                     const TensorLike<T, HeldOperation2>& B) {
  return{A, B};
}
// End of Operations ==========================================

} // util
} // cpp_nn


#endif // CPP_NN_R_TENSOR_OPERATIONS
