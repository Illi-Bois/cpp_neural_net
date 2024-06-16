#include "include/CPPNeuralNet/Utils/element_reference.h"

namespace cpp_nn {
namespace util {
// ElementReference ================================================================

// Constructor --------------------------------------------------

/** Tensor-Referencing with Broadcasting
 *  Assumes broadcast shape is valid
 *  Index is broadcast-set, that is in terms of broadcasted shape
 */
template<typename T>
ElementReference<T>::ElementReference(Tensor<T>& tensor, const std::vector<int>& broadcast_shape, 
                                      const std::vector<int>& indices)
    : TensorReference<T>(tensor, 0), broadcast_shape_(broadcast_shape),
      indices_(indices) {
  // Assumes the shape is boradcastable
  // As such, tensor's dimension check is bypassed

  if (indices_.size() != broadcast_shape_.size) {
    throw std::invalid_argument("ElementReference Constrcutor- Index does not match Broadcast Order");
  }
  for (int i = 0; i < indices_.size(); ++i) {
    if (indices_[i] < 0 || indices_[i] >= broadcast_shape_[i]) {
      throw std::invalid_argument("ElementReference Constrcutor- Index out of Broadcast Bound");
    }
  }
}
// End of Constructor -------------------------------------------


// End of ElementReference =========================================================
} // util
} // cpp_nn