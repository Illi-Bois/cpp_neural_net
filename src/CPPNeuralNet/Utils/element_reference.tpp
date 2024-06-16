#include "include/CPPNeuralNet/Utils/element_reference.h"

namespace cpp_nn {
namespace util {

// ElementReference ================================================================
// Constructor --------------------------------------------------
/** Tensor-Referencing */
template<typename T>
ElementReference<T>::ElementReference(Tensor<T>& tensor) : TensorReference<T>(tensor, 0) {}
/** Tensor-Referencing with Indices */
template<typename T>
ElementReference<T>::ElementReference(Tensor<T>& tensor, const std::vector<int>& indices)
    : TensorReference<T>(tensor, 0, indices) {}
// End of Constructor -------------------------------------------

// Accessors ----------------------------------------------------
template<typename T>
T& ElementReference<T>::getElement() {
  // return directly
  return this->elements_->elements_[this->index_address_];
}
// End of Accessors ---------------------------------------------
// End of ElementReference =========================================================

// BroadcastReference ==============================================================
// Constructor --------------------------------------------------
/** Tensor-Referencing with Broadcasting
 *  Assumes broadcast shape is valid
 */
template<typename T>
BroadcastReference<T>::BroadcastReference(Tensor<T>& tensor, const std::vector<int>& broadcast_shape)
    : ElementReference<T>(tensor), broadcast_shape_(broadcast_shape), 
      indices_(std::vector(broadcast_shape.size(), 0)) {
  if (this->elements_->order() > indices_.size()) {
    throw std::invalid_argument("BroadcastReference Constrcutor- Broadcast Shape smaller than Tensor Shape");
  }
}
/** Tensor-Referencing with Broadcasting
 *  Assumes broadcast shape is valid
 *  Index is broadcast-set, that is in terms of broadcasted shape
 */
template<typename T>
BroadcastReference<T>::BroadcastReference(Tensor<T>& tensor, const std::vector<int>& broadcast_shape, 
                                      const std::vector<int>& indices)
    : ElementReference<T>(tensor), broadcast_shape_(broadcast_shape),
      indices_(indices) {
  // Assumes the shape is boradcastable
  // As such, tensor's dimension check is bypassed

  if (this->elements_->order() > indices_.size()) {
    throw std::invalid_argument("BroadcastReference Constrcutor- Broadcast Shape smaller than Tensor Shape");
  }

  if (indices_.size() != broadcast_shape_.size) {
    throw std::invalid_argument("BroadcastReference Constrcutor- Index does not match Broadcast Order");
  }
  for (int i = 0; i < indices_.size(); ++i) {
    if (indices_[i] < 0 || indices_[i] >= broadcast_shape_[i]) {
      throw std::invalid_argument("BroadcastReference Constrcutor- Index out of Broadcast Bound");
    }
  }
}
// End of Constructor -------------------------------------------


// Iteration ----------------------------------------------------
/** Increments index over.
 *  If broadcasted, will loop over to fit the broadcast shape.
 */
template<typename T>
int BroadcastReference<T>::incrementIndex() {
  int order = indices_.size();
  int tensor_order = this->elements_.order();

  bool carry = false; // to be used to decide of iteration is carrying over

  this->index_address_ = 0;
  int jump_size = 1;

  for (int i = 0; i < order; ++i) {
    carry = false; // reset
    ++indices_[order - i]; // increment index

    // if within tensor_order, should update address as well
    if (i < tensor_order && 
        this->elements_->getDimension(tensor_order - i) != 1) { 
      // if dim is 1, index is either 0, or is broadcasted
      this->index_address_ += indices_[order - i] * this->elements_->getDimension(tensor_order - i);
      jump_size *= this->elements_->getDimension(tensor_order)
    }

    // check for carry over
    if (indices_[order - i] >= broadcast_shape_[order - i]) {
      indices_[order - i] = 0;
      carry = true;
      continue; // carry over
    } else {
      break;
    }
  }

  // success is determined by if final index is carried over or not
  if (!carry) return 1;

  this->index_address_ = 0; // if incrementation is not successful, reset index to 0 anyway
  return 0;
}
// End of Iteration ---------------------------------------------
// End of BroadcastReference =======================================================
} // util
} // cpp_nn