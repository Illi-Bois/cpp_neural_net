#include "include/CPPNeuralNet/Utils/tensor.h"

namespace cpp_nn {
namespace util {
// Tensor ==========================================================================

// TensorElement ============================================================
// TensorElement Constructor ------------------------------------------
/** TensorElement Dimension Const. */
template<typename T>
Tensor<T>::TensorElement::TensorElement(const std::initializer_list<int>& dims, T initial_value /*= T()*/)
    : dimensions_(dims), kCapacity(0) {
  if (dimensions_.size() != 0) {
    kCapacity = 1;
    for (const int& dim : dims) {
      if (dim >= 0) { // Non-Positive dimension is incorrect
        kCapacity *= dim;
        continue;
      }
      throw std::invalid_argument("TensorElement Constructor- Non-Positive Dimension Error"); 
    }
  }

  elements_.resize(kCapacity, initial_value);
  // Initially all index maps to self
  transpose_map_.reserve(order());
  for (int i = 0; i < order(); ++i) {
    transpose_map_.push_back(i);
  }
}
/** TensorElement Copy Constructor */
template<typename T>
Tensor<T>::TensorElement::TensorElement(const TensorElement& other)
    : dimensions_(other.dimensions_), elements_(other.elements_), 
      kCapacity(other.kCapacity), transpose_map_(other.transpose_map_) {}
// End of TensorElement Constructor ----------------------------------

// TensorElement Accessor --------------------------------------------
template<typename T>
T& Tensor<T>::TensorElement::getElement(const std::vector<int>& indices) {
  return elements_[ConvertToAddress(indices)];
}
template<typename T>
const T& Tensor<T>::TensorElement::getElement(const std::vector<int>& indices) const {
  return elements_[ConvertToAddress(indices)];
}
// End of TensorElement Accessor -------------------------------------


// Housekeeping ------------------------------------------------------
/** Index Address from Vetor Index
 * Converts dimension-based index from vector to array-address
 */
template<typename T>
int Tensor<T>::TensorElement::ConvertToAddress(const std::vector<int>& indices) const {
  if (indicies.size() != order()) throw std::invalid_argument("TensorElement ElementGetter- Indices Order Mismatch"); 

  // Transpose is handled by the fact that Dimension is accessed in Transposed order
  int array_index = 0;
  // block_size is chuck-size that i-th index jumps each time.
  // Bottom-Up apporoach. By multiplying up the block_size, division is avoided
  for (int i = order() - 1, block_size = 1; i >= 0; --i) {
    if (indices[i] >= 0 && indices[i] < getDimension(i)) {
      array_index += block_size * indices[i];
      block_size *= getDimension(i); // By passing through trans._map_, we can transpose in place
    } else {
      throw std::invalid_argument("TensorElement ElementGetter- Index Out of Bounds"); 
    }
  }
}
// End of Housekeeping -----------------------------------------------

// TensorElement Modifier --------------------------------------------
/** Transpose */
template<typename T>
void Tensor<T>::TensorElement::Transpose(int axis_one, int axis_two) {
  std::swap(transpose_map_[axis_one], transpose_map_[axis_two]);
}
// End of TensorElement Modifier -------------------------------------
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
template<typename T>
const T& Tensor<T>::getElement(const std::vector<int>& indices) const {
  return elements_->getElement(indices);
}
// End of Accessors ----------------------------------------------------

// Tensor Operations ---------------------------------------------------
template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const {
  if (getOrder() < 2 || other.getOrder() < 2) 
    throw std::invalid_argument("Tensor Multiplication- Tensor is not Matrix");

  if (elements_->getDimension(getOrder() - 1)  != other.elements_->getDimension(other.getOrder() - 2))
    throw std::invalid_argument("Tensor Multiplication- Multiplcation Dimension Mismatch");
  
  // [res_rows, inter_dim] * [inter_dim, res_cols]
  int res_rows = getDimension(getOrder() - 2);
  int res_cols = getDimension(other.getOrder() - 1);
  int inter_dim = getDimension(getOrder() - 1); 
  
  // given A[dim1..., r, k] and B[dim2..., k, c], the resulting product is of dim C[dim1..., dim2..., r, c]
  std::vector<int> res_dim;
  res_dim.reserve(this->getOrder() + other.getOrder() - 2);
  for (int i = 0; i < this->getOrder() - 2; ++i) {
    res_dim.push_back(this->getDimension(i));
  }
  for (int i = 0; i < other.getOrder() - 2; ++i) {
    res_dim.push_back(other.getDimension(i));
  }
  res_dim.push_back(res_rows);
  res_dim.push_back(res_cols);
  Tensor<T> res(res_dim);

  // Each Matrix chunk is handled via MatrixReference
  MatrixReference A(*this);
  MatrixReference B(other);
  MatrixReference C(res);

  // Multiply each chunk: C = A * B
  do { // while A has next
    do { // while B has next
      // Multiply 
      C.MultiplyInto(A, B);

      C.incrementIndex(); 
    } while(B.incrementIndex()/* != 0*/);
  } while(A.incrementIndex()/* != 0*/);

  return res;
}
// End of Tensor Operations --------------------------------------------

// Broadcast --------------------------------------------
template <typename T>
std::vector<int> Tensor<T>::broadcast(const Tensor<T>& other) const {
  int this_order = getOrder();
  int other_order = other.getOrder();
  int max_order = std::max(this_order, other_order);

  std::vector<int> res_dim(max_order);

  // traverse dimensions backwards
  int this_idx = getOrder() - 1;
  int other_idx = other.getOrder() - 1;
  int res_idx = max_order - 1;

  while (this_idx >= 0 && other_idx >= 0) { // both index are within order bound
    int this_dim = getDimension(this_idx);
    int other_dim = other.getDimension(other_idx);

    if (this_dim != other_dim && this_dim != 1 && other_dim != 1) {
      throw std::runtime_error("Not compatible for broadcasting.");
    }

    res_dim[res_idx] = std::max(this_dim, other_dim);
    
    // dec counter
    --this_idx;
    --other_idx;
    --res_idx;
  }

  // either or both of this_idx or other is depleted
  while (this_idx >= 0) {
    res_dim[res_idx] = getDimension(this_idx);
    --res_idx;
    --this_idx;
  }
  while (other_idx >= 0) {
    res_dim[res_idx] = other.getDimension(other_idx);
    --res_idx;
    --other_idx;
  }

  return res_dim;
}

// End of Broadcast --------------------------------------------

// getIndex --------------------------------------------
template<typename T>
std::vector<int> Tensor<T>::getIndex(int index, const std::vector<int>& shape) const {
  std::vector<int> indices(shape.size());
  for (int i = shape.size() - 1; i >= 0; --i) {
    // remainder (index & shape[i]) is the index for current dimension i
    indices[i] = index % shape[i];
    // Update the flat index to remove the current dimension's contribution
    index /= shape[i];
  }
  return indices;
}
// End of getIndex --------------------------------------------

//Element Wise Operations --------------------------------------------
template <typename T>
Tensor<T> Tensor<T>::elementwise_add(const Tensor<T>& other) const {
  std::vector<int> result_dim = broadcast(other);
  Tensor<T> result(result_dim);

  for (int i = 0; i < result.size(); ++i) {
    // Convert the flat index `i` into multi-dimensional indices for the result tensor
    std::vector<int> result_indices = getIndex(i, result_dim);
    // Initialize vectors to hold the indices for `this` tensor and the `other` tensor
    std::vector<int> this_indices(this->getOrder(), 0);
    std::vector<int> other_indices(other.getOrder(), 0);
    for (int j = 0; j < result_indices.size(); ++j) {
      // If `j` is a part of `this` tensor's dimensions
      if (j >= result_indices.size() - this->getOrder()) {
        // If dimension size == 1 -> 0, otherwise, assign corresponding value from result_indices
        this_indices[j - (result_indices.size() - this->getOrder())] = (this->getDimension(j - (result_indices.size() - this->getOrder())) == 1) ? 0 : result_indices[j];
      }
      // If `j` is a part of the `other` tensor's dimensions
      if (j >= result_indices.size() - other.getOrder()) {
        other_indices[j - (result_indices.size() - other.getOrder())] = (other.getDimension(j - (result_indices.size() - other.getOrder())) == 1) ? 0 : result_indices[j];
      }
    }
    result(result_indices) = this->getElement(this_indices) + other.getElement(other_indices);
  }
  return result;
}



//End of Element Wise Operatios --------------------------------------------


// End of Tensor ===================================================================

} // util
} // cpp_nn