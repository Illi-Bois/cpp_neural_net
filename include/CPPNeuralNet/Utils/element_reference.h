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
/** Tensor-Referencing */
  ElementReference(const Tensor<T>& tensor);
/** Tensor-Referencing with Indices */
  ElementReference(const Tensor<T>& tensor, const std::vector<int>& indices);
/** Tensor-Referencing with Indices as InitList */
  ElementReference(const Tensor<T>& tensor, const std::initializer_list<int>& indices);
// End of Constructor -------------------------------------------

// Accessors ----------------------------------------------------
  /** Getter */
  virtual T& getElement();
  virtual const T& getElement() const;
  /** Getter */
  inline T& operator()() {return getElement();}
  inline const T& operator()() __EDG_CONSTEXPR_ENABLED__ {return getElement();}
// End of Accessors ---------------------------------------------

}; // End of ElementReference =============================================================================


/**
 * Subclass of TensorReference, to help in iterating through chosen broadcast shape.
 * Think of as: 1-Chunk-order TensorReference with custum Iteration by given shape
 */
template<typename T = double>
class BroadcastReference : ElementReference<T> { // ========================================================
 private:
  const std::vector<int> kBroadcastShape; // shape of the target. iteration will follow the broadcastes shape
                                    // empty if not broadcasted, then will follow original shape
  std::vector<int> indices_; // For this, we need to go back to using vector of indicies
 public:
 // Constructor --------------------------------------------------
/** Tensor-Referencing with Broadcasting
 *  Assumes broadcast shape is valid
 */
  BroadcastReference(const Tensor<T>& tensor, const std::vector<int>& broadcast_shape);
/** Tensor-Referencing with Broadcasting with Index
 *  Assumes broadcast shape is valid
 *  Index is broadcast-set, that is in terms of broadcasted shape
 */
  BroadcastReference(const Tensor<T>& tensor, const std::vector<int>& broadcast_shape, const std::vector<int>& indices);
  /** Tensor-Referencing with Broadcasting and Inex as InitList */
  BroadcastReference(const Tensor<T>& tensor, const std::vector<int>& broadcast_shape, const std::initializer_list<int>& indices);
// End of Constructor -------------------------------------------

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