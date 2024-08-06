/**
 * TensorOperations that handle elemenet-wise operations.
 */
#ifndef CPP_NN_ELEMENTWISE_OPERATIONS
#define CPP_NN_ELEMENTWISE_OPERATIONS

#include "CPPNeuralNet/Utils/utils.h"
#include "CPPNeuralNet/Utils/tensor_like.h"

#include "CPPNeuralNet/Utils/TensorOperations/broadcast_operations.h"

namespace cpp_nn {
namespace util {
namespace {
/**
 *  sums two tensors with broadcasting applied. 
 */
template<typename T, typename HeldOperation1, typename HeldOperation2>
class SummationOperation : public TensorLike<T, SummationOperation<T, HeldOperation1, HeldOperation2>> {
  typedef SummationOperation<T, HeldOperation1, HeldOperation2> Self;
  typedef TensorLike<T, Self> Parent;

// Members ---------------------------------------------
    // const std::vector<int> broadcast_shape_;
    // const size_t broadcast_capacity_;

    // const BroadcastOperation<T, HeldOperation1> first_;
    // const BroadcastOperation<T, HeldOperation2> second_;
  BroadcastedPairHolder<T, HeldOperation1, HeldOperation2> broadcast_pair_;
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  SummationOperation(const TensorLike<T, HeldOperation1>& first, 
                     const TensorLike<T, HeldOperation2>& second) 
      // Broadcasting also checks for compatiblity.
      : broadcast_pair_(first, second) {}
// End of Constructor ----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    return broadcast_pair_.getShape();
  } 
  inline int getDimension(int axis) const {
    return getShape()[SumIfNegative(axis, getOrder())];
  }
  inline size_t getCapacity() const noexcept {
    return broadcast_pair_.getCapacity();
  }
  inline int getOrder() const noexcept {
    return getShape().size();
  }
  inline const T getElement(const std::vector<int>& indices) const {
    return broadcast_pair_.getFirst().getElement(indices) +
           broadcast_pair_.getSecond().getElement(indices);
  }
// End of Tensor-Behaviours ----------------------------

  class ConstIterator : public Parent::template ConstIterator<ConstIterator> {
    typedef typename BroadcastOperation<T, HeldOperation1>::ConstIterator BroadcastIterator1;
    typedef typename BroadcastOperation<T, HeldOperation2>::ConstIterator BroadcastIterator2;
    BroadcastIterator1 first_it_;
    BroadcastIterator2 second_it_;
   public:
    ConstIterator(BroadcastIterator1 first_it, 
                  BroadcastIterator2 second_it)
        : first_it_(first_it),
          second_it_(second_it) {}

    T operator*() const {
      return *first_it_ + *second_it_;
    }

    ConstIterator& operator+=(int increment) {
      first_it_ += increment;
      second_it_ += increment;
      return *this;
    }
    ConstIterator& operator-=(int decrement) {
      first_it_ -= decrement;
      second_it_ -= decrement;
      return *this;
    }

    bool operator==(const ConstIterator& other) const {
      return first_it_ == other.first_it_ && 
             second_it_ == other.second_it_;
    }
  };

  ConstIterator begin() const {
    return {broadcast_pair_.getFirst().begin(),
            broadcast_pair_.getSecond().begin()};
  }
  ConstIterator end() const {
    return {broadcast_pair_.getFirst().end(), 
            broadcast_pair_.getSecond().end()};
  }
}; // End of SummationOperation


/**
 *  subtracts two tensors with broadcasting applied. 
 */
template<typename T, typename HeldOperation1, typename HeldOperation2>
class SubtractionOperation : public TensorLike<T, SubtractionOperation<T, HeldOperation1, HeldOperation2>> {
  typedef SubtractionOperation<T, HeldOperation1, HeldOperation2> Self;
  typedef TensorLike<T, Self> Parent;

// Members ---------------------------------------------
    // const std::vector<int> broadcast_shape_;
    // const size_t broadcast_capacity_;

    // const BroadcastOperation<T, HeldOperation1> first_;
    // const BroadcastOperation<T, HeldOperation2> second_;
  BroadcastedPairHolder<T, HeldOperation1, HeldOperation2> broadcast_pair_;
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  SubtractionOperation(const TensorLike<T, HeldOperation1>& first, 
                       const TensorLike<T, HeldOperation2>& second) 
      // Broadcasting also checks for compatiblity.
      : broadcast_pair_(first, second) {}
// End of Constructor ----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    return broadcast_pair_.getShape();
  } 
  inline int getDimension(int axis) const {
    return getShape()[SumIfNegative(axis, getOrder())];
  }
  inline size_t getCapacity() const noexcept {
    return broadcast_pair_.getCapacity();
  }
  inline int getOrder() const noexcept {
    return getShape().size();
  }
  inline const T getElement(const std::vector<int>& indices) const {
    return broadcast_pair_.getFirst().getElement(indices) -
           broadcast_pair_.getSecond().getElement(indices);
  }
// End of Tensor-Behaviours ----------------------------

  class ConstIterator : public Parent::template ConstIterator<ConstIterator> {
    typedef typename BroadcastOperation<T, HeldOperation1>::ConstIterator BroadcastIterator1;
    typedef typename BroadcastOperation<T, HeldOperation2>::ConstIterator BroadcastIterator2;
    BroadcastIterator1 first_it_;
    BroadcastIterator2 second_it_;
   public:
    ConstIterator(BroadcastIterator1 first_it, 
                  BroadcastIterator2 second_it)
        : first_it_(first_it),
          second_it_(second_it) {}

    T operator*() const {
      return *first_it_ - *second_it_;
    }

    ConstIterator& operator+=(int increment) {
      first_it_ += increment;
      second_it_ += increment;
      return *this;
    }
    ConstIterator& operator-=(int decrement) {
      first_it_ -= decrement;
      second_it_ -= decrement;
      return *this;
    }

    bool operator==(const ConstIterator& other) const {
      return first_it_ == other.first_it_ && 
             second_it_ == other.second_it_;
    }
  };

  ConstIterator begin() const {
    return {broadcast_pair_.getFirst().begin(),
            broadcast_pair_.getSecond().begin()};
  }
  ConstIterator end() const {
    return {broadcast_pair_.getFirst().end(), 
            broadcast_pair_.getSecond().end()};
  }
}; // End of SubtractionOperation

template<typename T, typename HeldOperation>
class ScalarMultiplicationOperation : public TensorLike<T, ScalarMultiplicationOperation<T, HeldOperation>> {
  typedef ScalarMultiplicationOperation<T, HeldOperation> Self;
  typedef TensorLike<T, Self> Parent;

  const HeldOperation& tensor_like_;
  const T scalar_;

public:
  ScalarMultiplicationOperation(const TensorLike<T, HeldOperation>& tensor_like,
                                T scalar)
    : tensor_like_(tensor_like.getRef()),
      scalar_(scalar) {
    std::cout << "GOT" << std::endl;
  }
  inline const std::vector<int>& getShape() const noexcept {
    return tensor_like_.getShape();
  } 
  inline int getDimension(int axis) const {
    return tensor_like_.getDimension(axis);
  }
  inline size_t getCapacity() const noexcept {
    return tensor_like_.getCapacity();
  }
  inline int getOrder() const noexcept {
    return getShape().size();
  }
  inline const T getElement(const std::vector<int>& indicies) const {
    return tensor_like_.getElement(indicies) * scalar_;
  }

  typedef typename HeldOperation::ConstIterator HeldIterator;

  class ConstIterator : public Parent::template ConstIterator<ConstIterator> {
  // Members ---------------------------------------------
    HeldIterator it_;
    T scalar_;
  // End of Members --------------------------------------
   public:
   ConstIterator(HeldIterator it, T scalar) noexcept
      : it_(it), scalar_(scalar) {}
    
    T operator*() const {
      std::cout << "ACC " << *it_ << std::endl;
      return *it_ * scalar_;
    }

    ConstIterator& operator+=(int increment) {
      it_ += increment;
      std::cout << "ACC " << *it_ << std::endl;
      return *this;
    }
    ConstIterator& operator-=(int decrement) {
      it_ -= decrement;
      return *this;
    }
    bool operator==(const ConstIterator& other) const {
      return it_ == other.other;
    }
  }; // End of ConstIterator

  ConstIterator begin() const {
    return {tensor_like_.begin(), scalar_}; 
  }
  ConstIterator end() const {
    return {tensor_like_.end(), scalar_}; 
  }
};

/**
 *  operation on the element of the tensor one-on-one.
 */
template<typename T, typename HeldOperation>
class ElementOperation : public TensorLike<T, ElementOperation<T, HeldOperation>> {
  typedef ElementOperation<T, HeldOperation> Self;
  typedef TensorLike<T, Self> Parent;

// Members ---------------------------------------------
  const HeldOperation& tensor_like_;
  const std::function<T(T)> operation_;
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  ElementOperation(const TensorLike<T, HeldOperation>& tensor_like,
                       const std::function<T(T)>& operatoin)
      : tensor_like_(tensor_like.getRef()),
        operation_(operatoin) {}
// End of Constructor ----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    return tensor_like_.getShape();
  } 
  inline int getDimension(int axis) const {
    return tensor_like_.getDimension(axis);
  }
  inline size_t getCapacity() const noexcept {
    return tensor_like_.getCapacity();
  }
  inline int getOrder() const noexcept {
    return tensor_like_.getOrder();
  }
  inline const T getElement(const std::vector<int>& indices) const {
    return operation_(tensor_like_.getElement(indices));
  }
// End of Tensor-Behaviours ----------------------------

// Iterator --------------------------------------------
  typedef typename HeldOperation::ConstIterator HeldIterator;

  class ConstIterator : public Parent::template ConstIterator<ConstIterator>,
                        private HeldIterator {
    const std::function<T(T)>& operation_;
   public:
    ConstIterator(HeldIterator inner_iter,
                  const std::function<T(T)>& operation)
        : HeldIterator(inner_iter),
          operation_(operation) {}    

    T operator*() const {
      return operation_(this->HeldIterator::operator*());
    }
    
    using HeldIterator::operator+=;
    using HeldIterator::operator-=;

    using HeldIterator::operator++;
    using HeldIterator::operator--;
    using HeldIterator::operator!=;
  }; // ConstIterator


  inline ConstIterator begin() const {
    return {tensor_like_.begin(),
            operation_};
  }
  inline ConstIterator end() const {
    return {tensor_like_.end(),
            operation_};
  }
// End of Iterator -------------------------------------
}; // ElementOperation

/**
 *  bi-variate function that operate element-wise.
 *  When shapes are mismatched, broadcast is applied.
 */
template<typename T, typename HeldOperation1, typename HeldOperation2>
class ElementwiseOperation : public TensorLike<T, ElementwiseOperation<T, HeldOperation1, HeldOperation2>> {
  typedef ElementwiseOperation<T, HeldOperation1, HeldOperation2> Self;
  typedef TensorLike<T, Self> Parent;

// Members ---------------------------------------------
  BroadcastedPairHolder<T, HeldOperation1, HeldOperation2> broadcast_pair_;
  const std::function<T(T, T)> operation_;
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  ElementwiseOperation(const TensorLike<T, HeldOperation1>& first_tensor,
                       const TensorLike<T, HeldOperation2>& second_tensor,
                       const std::function<T(T, T)>& operatoin)
      : broadcast_pair_(first_tensor, second_tensor),
        operation_(operatoin) {}
// End of Constructor ----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    return broadcast_pair_.getShape();
  } 
  inline int getDimension(int axis) const {
    return getShape()[SumIfNegative(axis, getOrder())];
  }
  inline size_t getCapacity() const noexcept {
    return broadcast_pair_.getCapacity();
  }
  inline int getOrder() const noexcept {
    return getShape().size();
  }
  inline const T getElement(const std::vector<int>& indices) const {
    return operation_(broadcast_pair_.getFirst().getElement(indices),
                      broadcast_pair_.getSecond().getElement(indices));
  }
// End of Tensor-Behaviours ----------------------------

// Iterator --------------------------------------------
  class ConstIterator : public Parent::template ConstIterator<ConstIterator>{
    typedef typename BroadcastOperation<T, HeldOperation1>::ConstIterator BroadcastIterator1;
    typedef typename BroadcastOperation<T, HeldOperation2>::ConstIterator BroadcastIterator2;

    BroadcastIterator1 first_it_;
    BroadcastIterator2 second_it_;
    const std::function<T(T, T)>& operation_;

   public:
    ConstIterator(BroadcastIterator1 first_it, 
                  BroadcastIterator2 second_it,
                  const std::function<T(T, T)>& operation)
        : first_it_(first_it),
          second_it_(second_it),
          operation_(operation) {}    

    T operator*() const {
      return operation_(*first_it_,
                        *second_it_);
    }

    ConstIterator& operator+=(int increment) {
      first_it_ += increment;
      second_it_ += increment;
      return *this;
    }
    ConstIterator& operator-=(int decrement) {
      first_it_ -= decrement;
      second_it_ -= decrement;
      return *this;
    }

    bool operator==(const ConstIterator& other) const {
      return first_it_ == other.first_it_ && 
             second_it_ == other.second_it_;
    }
  }; // ConstIterator


  inline ConstIterator begin() const {
    return {broadcast_pair_.getFirst().begin(),
            broadcast_pair_.getSecond().begin(),
            operation_};
  }
  inline ConstIterator end() const {
    return {broadcast_pair_.getFirst().end(), 
            broadcast_pair_.getSecond().end(),
            operation_};
  }
// End of Iterator -------------------------------------
}; // ElementwiseOperation for Two


} // unnamed namespace
} // util
} // cpp_nn

#endif // CPP_NN_ELEMENTWISE_OPERATIONS