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

// Iterators -------------------------------------------
/*  Each TensorLike is to implement their version of Iterator. 
    The typename must be present as ConstIterator. */
  class Iterator {
   public:
    typedef typename Derived::Iterator Derived_Iterator;

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
  }; // End of Iterator
  class ConstIterator {
   public:
    typedef typename Derived::ConstIterator Derived_Iterator;
    // as operation return non-reference, accessor for iterator also returns by value
    virtual T operator*() const = 0;

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
  }; // end of ConstIterator
// Iterator Callers --------------------------------
  // TensorLike objects only need implement ConstIter
  ConstIterator begin() const {
    return getRef().begin();
  }
  ConstIterator end() const {
    return getRef().end();
  }
// End of Iterator Callers -------------------------
// End of Iterators ------------------------------------

 protected:
// Default Iterators -----------------------------------
/*  Default iterator iterates the TensorLike using vector-indices.
    Therefore it is unoptimized, but serves as intinial functioning placeholder
    until a better iterator design can be implemented. 
    
    The TensorLike using DefaultConstIterator is to include the line
      typedef TensorLike<T, DERVIED>::DefaultConstIterator ConstIterator 
    with DERIVED changed to its class name, so that typename becomes visible and accessable. */
  class DefaultConstIterator : public ConstIterator {
    typedef typename ConstIterator::Derived_Iterator Derived_Iterator;
  // Members ---------------------------------------------
  protected: // some special cases might want to acccess, it, clean up later
    const Derived* const tensor_like_;
    std::vector<int> current_indices_;

    bool at_end_;  // both begin and end will be indicated by indices at all 0
                    // must be distinguished by at_end_ flag. 
  // End of Members --------------------------------------

   public:
    DefaultConstIterator(const Derived* tensor_like, 
                         const std::vector<int>& idx, 
                         bool is_end)
        : tensor_like_(tensor_like),
          current_indices_(idx),
          at_end_(is_end) {}

    T operator*() const override {
      return tensor_like_->getElement(current_indices_);
    }

    Derived_Iterator& operator+=(int increment) override {
      if (at_end_) {
        // cannot increment from end, quick exit
        return static_cast<Derived_Iterator&>(*this);
      }
      if (IncrementIndicesByShape(tensor_like_->getShape().begin(), 
                                  tensor_like_->getShape().end(),
                                  current_indices_.begin(), 
                                  current_indices_.end(),
                                  increment)) {
        return static_cast<Derived_Iterator&>(*this);
      } else {
        // overflowed
        at_end_ = true;
      }
      return static_cast<Derived_Iterator&>(*this);
    }
    Derived_Iterator& operator-=(int decrement) override {
      // Some initial checks so that end can be handled little more smoothly
      if (decrement < 0) {
        return operator+=(-decrement);
      }
      if (decrement == 0) {
        // do nothing
        return static_cast<Derived_Iterator&>(*this);
      }
      
      // If at end, idx are all set to 0
      if (at_end_) {
        // Manually set to correct maximum size
        for (int axis = 0; axis < tensor_like_->getOrder(); ++axis) {
          current_indices_[axis] = tensor_like_->getDimension(axis) - 1;
        }
        // alternative was to use Derement, but this might be faster
        --decrement;    // this uses up one decrement
        at_end_ = false;

        if (decrement == 0) {
          // quick exit
          return static_cast<Derived_Iterator&>(*this); 
        }
      }

      // decrement is now guaranteed > 0
      if (DecrementIndicesByShape(tensor_like_->getShape().begin(), 
                                  tensor_like_->getShape().end(),
                                  current_indices_.begin(), 
                                  current_indices_.end(),
                                  decrement)) {
        // No underflow, therefore successful
        return static_cast<Derived_Iterator&>(*this);
      } else {
        // underflow set all to 0
        std::fill(current_indices_.begin(), current_indices_.end(), 0);
      }
      return static_cast<Derived_Iterator&>(*this);
    }

    bool operator==(const Derived_Iterator& other) const override {
      return tensor_like_ == other.tensor_like_ && 
              current_indices_ == other.current_indices_;
    }
  }; // End of DefaultConstIterator
// End of Default Iterators ----------------------------

  // TODO: Ideally want to CRTP to remove virtual, but right now
  //      because of nesting inside template that is CRTP, seems impossible...
  // TODO: many of the motions are duplicate, maybe consider making this interface as we had tried
  //      but adding more interface seems to increase runtime
};

/* End of Operation Facilitators --------------------------------------------------- */
} // End of unnamed namespace ================================================================

} // util
} // cpp_nn


#endif // CPP_NN_R_TENSOR_LIKE