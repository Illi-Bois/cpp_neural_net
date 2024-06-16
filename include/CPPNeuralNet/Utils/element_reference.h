#ifndef CPP_NN_ELEMENT_REF
#define CPP_NN_ELEMENT_REF

#include "include/CPPNeuralNet/Utils/tensor.h"
#include "include/CPPNeuralNet/Utils/tensor_reference.h"

#include <vector>

namespace cpp_nn {
namespace util {

/** Element-wise iterator of Tensor */
template<typename T = double>
class ElementReference : public TensorReference<T> { // ===================================================
 public:
// Constructor --------------------------------------------------
/** Tensor-Referencing
 * Broadcasting set to none
 */
  ElementReference(Tensor<T>& tensor);
/** Tensor-Referencing
 * Broadcasting set to none
 */
  ElementReference(Tensor<T>& tensor, const std::vector<int>& indices);
// End of Constructor -------------------------------------------

// Accessors ----------------------------------------------------
  /** Getter */
  virtual T& getElement();
// End of Accessors ---------------------------------------------

}; // End of ElementReference =============================================================================


/**
 * Subclass of TensorReference, to help in iterating through chosen broadcast shape.
 * Think of as: 1-Chunk-order TensorReference with custum Iteration by given shape
 */
template<typename T = double>
class BroadcastReference : ElementReference<T> { // ========================================================
 private:
  std::vector<int> broadcast_shape_; // shape of the target. iteration will follow the broadcastes shape
                                    // empty if not broadcasted, then will follow original shape
  std::vector<int> indices_; // For this, we need to go back to using vector of indicies
 public:
 // Constructor --------------------------------------------------
/** Tensor-Referencing with Broadcasting
 *  Assumes broadcast shape is valid
 */
  ElementReference(Tensor<T>& tensor, const std::vector<int>& broadcast_shape);
/** Tensor-Referencing with Broadcasting
 *  Assumes broadcast shape is valid
 *  Index is broadcast-set, that is in terms of broadcasted shape
 */
  ElementReference(Tensor<T>& tensor, const std::vector<int>& broadcast_shape, const std::vector<int>& indices);
// End of Constructor -------------------------------------------

// Accessors ----------------------------------------------------
  /** Getter */
  T& getElement() override;
// End of Accessors ---------------------------------------------

// Iteration ----------------------------------------------------
/** Increments index over.
 *  If broadcasted, will loop over to fit the broadcast shape.
 */
  int incrementIndex() override;
// End of Iteration ---------------------------------------------
}; // End of BroadcastReference ===========================================================================

} // util
} // cpp_nn
#endif // CPP_NN_ELEMENT_REF