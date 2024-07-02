
#ifndef CPP_NN_R_TENSOR_LIKE
#define CPP_NN_R_TENSOR_LIKE

#include "CPPNeuralNet/Utils/utils.h"


namespace cpp_nn {
namespace util {

// Forward Declaration ======================================
namespace {
template<typename T, typename HeldOperation>
class TransposeOperation;
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
  inline int getOrder() const noexcept {
    return getRef().getOrder();
  }
  // returns by value as operation returns are temp
  inline const T getElement(const std::vector<int>& indices) const {
    return getRef().getElement(indices);
  }
  inline const TransposeOperation<T, Derived> Transpose(int axis1 = -2, int axis2 = -1) const {
    // return getRef().Transpose(axis1, axis2);
    return {*this, axis1, axis2};
  }
// End of Tensor-Behaviours ----------------------------
};

/* End of Operation Facilitators --------------------------------------------------- */
} // End of unnamed namespace ================================================================

} // util
} // cpp_nn


#endif // CPP_NN_R_TENSOR_LIKE