#ifndef CPP_NN_R_TENSOR_LIKE
#define CPP_NN_R_TENSOR_LIKE

#include "CPPNeuralNet/Utils/utils.h"

namespace cpp_nn {
namespace util {

// Forward Declaration ======================================
namespace {
// These are possible return types from member methods
template<typename T, typename HeldOperation>
class TransposeOperation;
template<typename T, typename HeldOperation>
class ReshapeOperation;
template<typename T, typename HeldOperation>
class PaddingOperation;
} // unnamed namespace

// End of Forward Declaration ===============================


namespace { // ===============================================================================
/** 
 *  is a base class of all objects that behave like Tensor. 
 *  As interface facilitating all tensor-like behaviors, defines and allows
 *    the necessary tensor-behaviours. 
 *  As it is not a tensor in itself, it is not responsible for holding and maintaining 
 *    actual tensor-data (though it can, it is advised against it),
 *    but is responsible for being the light-weight data to be passed
 *    between heavier tensor operations.  
 */
template<typename T, typename Derived>
class TensorLike {
 public:
// CRTP ------------------------------------------------
  /** 
   * returns Downcasted object.
   */
  inline const Derived& getRef() const {
    return static_cast<const Derived&>(*this);
  }
// End of CRTP -----------------------------------------

// Tensor-Behaviours -----------------------------------
//    interface methods. Each calls downcasted overriden methods. 
  inline const std::vector<int>& getShape() const noexcept {
    return getRef().getShape();
  }
  inline int getDimension(int axis) const {
    return getRef().getDimension(axis);
  }
  inline size_t getCapacity() const {
    return getRef().getCapacity();
  }
  inline int getOrder() const noexcept {
    return getRef().getOrder();
  }
  // returns by value as operation returns are temp
  inline const T getElement(const std::vector<int>& indices) const {
    return getRef().getElement(indices);
  }

  // TODO, move these externally to tensor_operation 
  //    that is, make these call static inline function
  inline const TransposeOperation<T, Derived> Transpose(int axis1 = -2, int axis2 = -1) const {
    return {*this, axis1, axis2};
  }
  inline const ReshapeOperation<T, Derived> 
               Reshape(const std::vector<int>& new_dimensions) const {
    return {*this, new_dimensions};
  }
  inline const PaddingOperation<T, Derived>
               Padding(const std::vector<int>& padded_dimensions) const {
    return {*this, padded_dimensions};
  }
// End of Tensor-Behaviours ----------------------------

// TODO: Element-wise iterator for fast and lightweight iteration of each element.
//  if each tensor-like has a lightweight iteration, each can call the previous's iterator to
//   quickly access element
};

/* End of Operation Facilitators --------------------------------------------------- */
} // End of unnamed namespace ================================================================

} // util
} // cpp_nn


#endif // CPP_NN_R_TENSOR_LIKE