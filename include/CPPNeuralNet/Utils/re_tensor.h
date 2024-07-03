
#ifndef CPP_NN_R_TENSOR
#define CPP_NN_R_TENSOR

#include "CPPNeuralNet/Utils/utils.h"
#include "CPPNeuralNet/Utils/tensor_like.h"
#include "CPPNeuralNet/Utils/tensor_operations.h"

#include <vector>
#include <numeric>                    // for accumulate 

namespace cpp_nn {
namespace util {

// Forward Declaration ======================================
namespace {
template<typename T, typename Derived>
class TensorLike;

template<typename T, typename HeldOperation>
class TransposeOperation;
template<typename T, typename HeldOperation1, typename HeldOperation2>
class SummationOperation;
template<typename T, typename HeldOperation1, typename HeldOperation2>
class MatrixIncrementIndices;
}
// End of Forward Declaration ===============================


template<typename T>
class rTensor : public TensorLike<T, rTensor<T>> { // ========================================
 public:
// Public Constructor-----------------------------------
/**
 *  constucts from list of dimensions.
 *  Only positive dimensions are valid. 
 */
  rTensor(const std::vector<int>& dimensions, 
          T init_val = T()); 
/**
 *  Copy Constructor
 */
  rTensor(const rTensor& other) noexcept;
/**
 *  Move Constructor
 */
  rTensor(rTensor&& other) noexcept;

/** 
 * TensorLike constructor
 * Operatuions are handled upon construction to minimize movement of data.
 */
  template<typename Derived>
  rTensor(const TensorLike<T, Derived>& tensor_like);
// End of Public Constructor----------------------------

// Destrcutor ------------------------------------------
  ~rTensor() noexcept;
// End of Destrcutor -----------------------------------

// Assignment Operators --------------------------------
/** Copy operator */
  rTensor& operator=(rTensor other) noexcept; 
// End of Assignment Operators -------------------------

// Accessors -------------------------------------------
/**
 *  retrieves vector representing diemension of tensor.
 */
  inline const std::vector<int>& getShape() const noexcept;
/** 
 *  retrieves dimension at given axis. 
 *  The axis can be nagative to wrap around python style. 
 * 
 *  Negative index is allowed only until -order.
 *  axis in [-order, order)
 *  Invalid axis will yield undefined behaviour.
 */
  inline int getDimension(int axis) const;
/** 
 *  retrieves order of the tensor. 
 */
  inline int getOrder() const noexcept;
/**
 *  retrieves total capacity of Tensor
 */
  inline size_t getCapacity() const noexcept;
/** 
 *  retrieves reference through vector of index. 
 */
  inline T& getElement(const std::vector<int>& indices);
  inline const T& getElement(const std::vector<int>& indices) const;
// End of Accessors -----------------`-------------------

// Modifiers -------------------------------------------
/**
 *  tranposes given axes. Axes can be negative for python style.
 *  By default, transposes last two axes. 
 *    This conforms to Matrix tranpose.
 */
  inline TransposeOperation<T, rTensor<T>> Transpose(int axis1 = -2, int axis2 = -1) const {
    return {*this, axis1, axis2};
  }
/**
 *  reshapes to new dimension shape.
 *  The capacity of new dimension must be same as current. 
 */
  rTensor<T>& Reshape(const std::vector<int>& new_dimensions);
// End of Modifiers ------------------------------------

 protected:
// Member Fields ---------------------------------------
  std::vector<int> dimensions_;
  std::vector<int> chunk_size_;   // capacity of chunk associated with each index
  size_t capacity_;
  std::vector<T>* elements_;
// End of Member fields --------------------------------

// Swap ------------------------------------------------
/** recommended for unifying copy constrcutor and operator from:
 *    https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
 */
  friend void swap(rTensor& first, rTensor& second) noexcept {
    // The definition seems to necessarily be placed in here, else
    //    linker fails to recognize it.
    std::swap(first.dimensions_, second.dimensions_);
    std::swap(first.chunk_size_, second.chunk_size_);
    std::swap(first.capacity_, second.capacity_);
    std::swap(first.elements_, second.elements_);
  } 
// End of Swap -----------------------------------------

// Housekeeping ----------------------------------------
/**
 *  converts vector-indices to integer address which can be used to access
 *    elements_ vector.
 *  Each index can be [-dimension, dimension) 
 * 
 *  Invalid index will result in std::invalid_argument("TensorIndex- Incorrect Order"),
 *                               std::invalid_argument("TensorIndex- Index Out of Bound"),
 */
  size_t ConvertToAddress(const std::vector<int>& indices) const;

  /** 
   *  with the assumption that 
   *    capacity_ = 1
   *    chunk_sizes = {1.... 1}
   *  compute capacity and chunk_sizes from dimensions.
   *  
   *  This is intended to be used for both constructor and reshape.
   *  
   *  Throws 
   * 
   *  TODO, make this a static in util so it may be used later?
   */
  inline void ComputeCapacityAndChunkSizes();
// End of Housekeeping ---------------------------------

 private:
}; // End of Tensor ==========================================================================

} // util
} // cpp_ nn



namespace cpp_nn {
namespace util {

// rTensor DEFINITION ============================================
// Constructors -----------------------------------------
/** Tensor Contructor with  */
template<typename T>
rTensor<T>::rTensor(const std::vector<int>& dimensions, 
                    T init_val)
    : dimensions_(dimensions),
      chunk_size_(dimensions.size(), 1),
      capacity_(1),
      elements_(nullptr) {
  
  ComputeCapacityAndChunkSizes();
  elements_ = new std::vector<T>(capacity_, init_val);
}

/** copy Constructor */
template<typename T>
rTensor<T>::rTensor(const rTensor& other) noexcept
    : dimensions_(other.dimensions_),
      chunk_size_(other.chunk_size_),
      capacity_(other.capacity_),
      elements_(new std::vector<T>(*other.elements_)) {
  // No exception throwing, as we can assume other is validly costructed
}
/** Move Constrcutro */
template<typename T>
rTensor<T>::rTensor(rTensor&& other) noexcept
    : dimensions_(std::move(other.dimensions_)),
      chunk_size_(std::move(other.chunk_size_)),
      capacity_(std::move(other.capacity_)),
      elements_(std::move(other.elements_)) {
  // free up other's pointer
  other.elements_ = nullptr;
}

/** 
 *  Operation Constrcutor
 * 
 * TODO: make specialized constructor for specific Operation such as Multiplication and Reshape
 */
template<typename T>
template<typename Derived>
rTensor<T>::rTensor(const TensorLike<T, Derived>& tensor_like)
    : rTensor(tensor_like.getShape()) {
  // with tensor initized to 0s, set each element one by one

  std::vector<int> indices(getOrder(), 0);

  do {
    getElement(indices) = tensor_like.getElement(indices);
  } while (IncrementIndicesByShape(getShape().begin(), getShape().end(),
                                   indices.begin(),    indices.end()));
}
// End of Constructors ----------------------------------

// Destructor -------------------------------------------
template<typename T>
rTensor<T>::~rTensor() noexcept {
  delete this->elements_;
}
// End of Destructor ------------------------------------

// Assignment Operators ---------------------------------
/** Copy Operator */
template<typename T>
rTensor<T>& rTensor<T>::operator=(rTensor<T> other) noexcept {
  // uses copy-swap idiom
  // As i undersatnd it, by relying on copy-constrcutor and swap
  //    we let paramater be initated with copy constrcutor, then 
  //    call swap to swap that new instance with current
  swap(*this, other);
  return *this;
}
// End of Assignment Operators --------------------------


// Accessors --------------------------------------------
template<typename T>
inline const std::vector<int>& rTensor<T>::getShape() const noexcept {
  return dimensions_;
}

template<typename T>
inline int rTensor<T>::getDimension(int axis) const {
  // this accomadates for wrap-around behaviours, but 
  //  for sake of optimization, we will bypass usual safety checks
  //  Invalid axis will simply yield UB  
  return dimensions_[SumIfNegative(axis, getOrder())];
}

template<typename T>
inline int rTensor<T>::getOrder() const noexcept {
  return dimensions_.size();
}

template<typename T>
inline size_t rTensor<T>::getCapacity() const noexcept {
  return capacity_;
}

template<typename T>
inline T& rTensor<T>::getElement(const std::vector<int>& indicies) {
  // Some const cast magic gotten from 
  //  https://stackoverflow.com/questions/856542/elegant-solution-to-duplicate-const-and-non-const-getters
  return const_cast<T&>(const_cast<const rTensor*>(this)->getElement(indicies));
  // This in effect pushes all getter aspect to const getter
}
template<typename T>
inline const T& rTensor<T>::getElement(const std::vector<int>& indicies) const {
  return (*elements_)[ConvertToAddress(indicies)];
}
// End of Accessors -------------------------------------


// Modifiers --------------------------------------------
// TODO, make reshape for Tensor-Like
template<typename T>
rTensor<T>& rTensor<T>::Reshape(const std::vector<int>& new_dimensions) {
  
  
  int old_capacity = capacity_;
  
  // Reset for new values
  capacity_ = 1;
  chunk_size_ = std::vector<int>(new_dimensions.size(), 1);
  dimensions_ = new_dimensions;

  ComputeCapacityAndChunkSizes();

  if (getCapacity() != old_capacity) {
    throw std::invalid_argument("TensorReshape- Capacity mismatch");
  }
  return *this;
}

// End of Modifiers -------------------------------------

// Housekeeping -----------------------------------------
template<typename T>
size_t rTensor<T>::ConvertToAddress(const std::vector<int>& indices) const {
  // TODO: consider making block_size which is block-size to be jumped
  //          by associated with axis's index
  //       Making it as external member will
  //          - eliminate need to compute same multiplication each time
  //        However
  //          - extra memeber to move around on copy and move
  //          - need to recompute each time tranpose/reshape is called
  //          
  if (indices.size() != getOrder()) {
    throw std::invalid_argument("TensorIndex- Incorrect order");
  }

  size_t address = 0;
  int curr_idx,
      curr_dim;
  // as dimension accepts int, we should keep axis as int as well
  for (int axis = getOrder() - 1; axis >= 0; --axis) {
    curr_idx = indices[axis];
    curr_dim = getDimension(axis);

    if (curr_idx >= -curr_dim && curr_idx < curr_dim) {
      address += SumIfNegative(curr_idx, curr_dim) * chunk_size_[axis];
    } else {
      throw std::invalid_argument("TensorIndex- Index out of bound");
    }
  }

  return address;
}

template<typename T>
inline void rTensor<T>::ComputeCapacityAndChunkSizes() {
  /*
    assumes capacity_ = 1
            chunk_size_ = std::vector<int>(getOrder(), 1);
  */
  for (int axis = dimensions_.size() - 1; axis >= 1; --axis) {
    if (dimensions_[axis] > 0) {
      chunk_size_[axis - 1] = chunk_size_[axis] * dimensions_[axis];
    } else {
      throw std::invalid_argument("Tensor Dimension- Non-positive dimension given");
    }
  }
  if (dimensions_[0] > 0) {
    capacity_ = chunk_size_[0] * dimensions_[0];
  } else {
    throw std::invalid_argument("Tensor Dimension- Non-positive dimension given");
  }
}

// End of Housekeeping ----------------------------------
// End of rTensor DEFINITION =====================================


// Extreneous ---------------------------------------------
// End of Extreneous --------------------------------------


} // util
} // cpp_nn

#endif // CPP_NN_R_TENSOR
