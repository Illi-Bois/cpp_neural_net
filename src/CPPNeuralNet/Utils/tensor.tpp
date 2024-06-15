#include "include/CPPNeuralNet/Utils/tensor.h"

namespace cpp_nn {
namespace util {
// Tensor ==========================================================================

// TensorElement ============================================================
// TensorElement Constructor ------------------------------------------
/** TensorElement Dimension Const. */
template<typename T>
Tensor<T>::TensorElement::TensorElement(const std::initializer_list<int>& dims, 
                                        T initial_value /*= T()*/)
    : dimensions(dims), capacity(0) {
  if (dimensions.size() != 0) {
    capacity = 1;
    for (const int& dim : dims) {
      if (dim >= 0) { // Non-Positive dimension is incorrect
        capacity *= dim;
        continue;
      }
      throw std::invalid_argument("TensorElement Constructor- Non-Positive Dimension Error"); 
    }
  }

  elements.resize(capacity, initial_value);
}
/** TensorElement Copy Constructor */
template<typename T>
Tensor<T>::TensorElement::TensorElement(const TensorElement& other)
    : dimensions(other.dimensions), elements(other.elements), capacity(other.capacity) {}
// End of TensorElement Constructor ----------------------------------

// TensorElement Accessor --------------------------------------------
template<typename T>
T& Tensor<T>::TensorElement::getElement(const std::vector<int>& indices) {
  if (indicies.size() != order()) throw std::invalid_argument("TensorElement ElementGetter- Indices Order Mismatch"); 
  
  // TODO
  // transpose should map indiceis at this point.

  int array_index = 0;

  // block_size is chuck-size that i-th index jumps each time.
  // Bottom-Up apporoach. By multiplying up the block_size, division is avoided
  for (int i = order() - 1, block_size = 1; i >= 0; --i) {
    if (indices[i] >= 0 && indices[i] < dimensions[i]) {
      array_index += block_size * indices[i];
      block_size *= dimensions[i];
    } else {
      throw std::invalid_argument("TensorElement ElementGetter- Index Out of Bounds"); 
    }
  }

  return elements[array_index];
}
// End of TensorElement Accessor -------------------------------------
// End of TensorElement =====================================================

// Constructors --------------------------------------------------------
/** Dimension Contructors */
template<typename T>
Tensor<T>::Tensor(std::initializer_list<int> dims, T initial_value) 
    : elements_(new TensorElement(dims, initial_value)), ownership_(true) {}
/** Dimension Contructors, Vector */
template<typename T>
Tensor<T>::Tensor(std::vector<int> dims, T initial_value) 
    : elements_(new TensorElement(dims, initial_value)), ownership_(true) {}
/** Copy Constructor */
template<typename T>
Tensor<T>::Tensor(const Tensor& other)
    : elements_(new TensorElement(*other.elements_)), ownership_(true) {}
/** Move Constrcutor */
template<typename T>
Tensor<T>::Tensor(Tensor&& other)
    : elements_(other.elements_), ownership_(true) {
  // unlink other
  other.elements_ = nullptr;
  other.ownership_ = false;
}
// End of Constructors -------------------------------------------------

// Accessors -----------------------------------------------------------
/** Element Getter */
template<typename T>
T& Tensor<T>::getElement(const std::vector<int>& indices) {
  return elements_->getElement(indices);
}
// End of Accessors ----------------------------------------------------

// Tensor Operations ---------------------------------------------------
template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const {
  if (getOrder() < 2 || other.getOrder() < 2) 
    throw std::invalid_argument("Tensor Multiplication- Tensor is not Matrix");

  if (elements_->dimensions[getOrder() - 1] != other.elements_->dimensions[other.getOrder() - 2])
    throw std::invalid_argument("Tensor Multiplication- Multiplcation Dimension Mismatch");
  
  // [res_rows, inter_dim] * [inter_dim, res_cols]
  int res_rows = getDimension(getOrder() - 2);
  int res_cols = getDimension(other.getOrder() - 1);
  int inter_dim = getDimension(getOrder() - 1); 
  
  // given A[dim1..., r, k] and B[dim2..., k, c], the resulting product is of dim C[dim1..., dim2..., r, c]
  std::vector<int> res_dim{res_rows, res_cols};
  res_dim.insert(res_dim.begin(), other.elements_->dimensions.begin(), other.elements_->dimensions.end() - 2);
  res_dim.insert(res_dim.begin(), elements_->dimensions.begin(), elements_->dimensions.end() - 2);
  Tensor<T> res(res_dim);

  // Each Matrix chunk is handled via MatrixReference
  MatrixReference A(*this);
  MatrixReference B(other);
  MatrixReference C(res);

  // Multiply each chunk: C = A * B
  do { // while A has next
    do { // while B has next
      // Multiply 
      for (int r = 0; r < res_rows; ++r) {
        for (int c = 0; c < res_col; ++c) {
          C.getElement(r, c) = 0;
          for (int k = 0; k < inter_dim; ++k) {
            C.getElement(r, c) += A.getElement(r, k) * B.getElement(k, c);
          }
        }
      }

      C.incrementIndex(); 
    } while(B.incrementIndex()/* != 0*/);
  } while(A.incrementIndex()/* != 0*/);

  return res;
}
// End of Tensor Operations --------------------------------------------

// Broadcast --------------------------------------------
template <typename T>
std::vector<int> Tensor<T>::broadcast(const Tensor<T>& other) const {
  int max_order = std::max(getOrder(), other.getOrder());
  std::vector<int> result_dim(max_order);

  for (int i = 0; i < max_order; ++i) {
    // Gets dimension value from right to left, pads with 1 if i >= this->order
    int this_dim = (i < getOrder()) ? getDimension(getOrder() - 1 - i) : 1;
    int other_dim = (i < other.getOrder()) ? other.getDimension(other.getOrder() - 1 - i) : 1;

    // Broadcasting requires dimension to be the same, or at least one is 1
    if (this_dim != other_dim && this_dim != 1 && other_dim != 1) {
      throw std::runtime_error("Not compatible for broadcasting.");
    }

    result_dim[max_order - 1 - i] = std::max(this_dim, other_dim);
  }

  return result_dim;
}
// End of Broadcast --------------------------------------------

// End of Tensor ===================================================================

} // util
} // cpp_nn