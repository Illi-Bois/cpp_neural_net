#ifndef CPP_NN_ELEMENT_REF
#define CPP_NN_ELEMENT_REF

#include "include/CPPNeuralNet/Utils/tensor.h"
#include "include/CPPNeuralNet/Utils/tensor_reference.h"

#include <vector>

/**
 * Subclass of TensorReference, to help in iterating through chosen broadcast shape.
 * Think of as: 1-Chunk-order TensorReference with custum Iteration by given shape
 */
template<typename T = double>
class ElementReference : public TensorReference<T> { // ===================================================
 private:
  std::vector<int> broadcast_shape_; // shape of the target. iteration will follow the broadcastes shape
                                    // empty if not broadcasted, then will follow original shape
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
  T& getElement();
// End of Accessors ---------------------------------------------

// Iteration ----------------------------------------------------
/** Increments index over.
 *  If broadcasted, will loop over to fit the broadcast shape.
 */
  int incrementIndex() override;
// End of Iteration ---------------------------------------------
}; // End of ElementReference =============================================================================


#endif // CPP_NN_ELEMENT_REF