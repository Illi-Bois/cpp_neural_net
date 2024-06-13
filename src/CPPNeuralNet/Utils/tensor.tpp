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
/** Copy Constructor */
template<typename T>
Tensor<T>::Tensor(const Tensor& other)
    : elements_(new TensorElement(*other.elements_)), ownership_(true) {}
/** Move Constrcutor */
template<typename T>
Tensor<T>::Tensor(Tensor&& other)
    : elements_(other.elements_), ownership(true) {
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

// End of Tensor ===================================================================

}
} // cpp_nn