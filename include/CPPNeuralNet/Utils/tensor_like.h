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
  inline size_t getCapacity() const noexcept {
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
  inline const TransposeOperation<T, Derived> 
               Transpose(int axis1 = -2, int axis2 = -1) const {
    return {*this, axis1, axis2};
  }
  /// Below Operations are better return as nonconstant as they may be modified further in-body. 
  /// When not used for further modification, still are caught as const HeldOperations, 
  ///   and so poses no issue
  inline ReshapeOperation<T, Derived> 
         Reshape(const std::vector<int>& new_dimensions) const {
    return {*this, new_dimensions};
  }
  inline PaddingOperation<T, Derived>
         Padding(const std::vector<int>& padded_dimensions, 
                 T padded_value = T()) const {
    return {*this, padded_dimensions, padded_value};
  }
// End of Tensor-Behaviours ----------------------------
 
 public:
  class Iterator {
    typedef typename Derived::Iterator Derived_Iterator;
   public:
    virtual T& operator*() = 0;

    virtual Derived_Iterator& operator+=(int increment) = 0;
    virtual Derived_Iterator& operator-=(int decrement) = 0;

    Derived_Iterator& operator++() {
      return (*this) += 1;
    }
    Derived_Iterator& operator--() {
      return (*this) -= 1;
    }

    virtual bool operator==(const Derived_Iterator& other) const = 0;
    bool operator!=(const Derived_Iterator& other) const {
      return !(*this == other);
    }
  };

  class ConstIterator {
    typedef typename Derived::ConstIterator Derived_Iterator;
   public:
    virtual const T& operator*() const = 0;

    virtual Derived_Iterator& operator+=(int increment) = 0;
    virtual Derived_Iterator& operator-=(int decrement) = 0;

    Derived_Iterator& operator++() {
      return (*this) += 1;
    }
    Derived_Iterator& operator--() {
      return (*this) -= 1;
    }

    virtual bool operator==(const Derived_Iterator& other) const = 0;
    bool operator!=(const Derived_Iterator& other) const {
      return !(*this == other);
    }
  };
  // TODO: Ideally want to CRTP to remove virtual, but right now
  //      because of nesting inside template that is CRTP, seems impossible...
  // TODO: many of the motions are duplicate, maybe consider making this interface as we had tried
  //      but adding more interface seems to increase runtime

 public:
  // CRTP virtual
  Iterator begin() {
    return getRef().begin();
  }
  ConstIterator begin() const {
    return getRef().begin();
  }

// TODO: Element-wise iterator for fast and lightweight iteration of each element.
//  if each tensor-like has a lightweight iteration, each can call the previous's iterator to
//   quickly access element
};

/* End of Operation Facilitators --------------------------------------------------- */
} // End of unnamed namespace ================================================================

} // util
} // cpp_nn


#endif // CPP_NN_R_TENSOR_LIKE