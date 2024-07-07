
#ifndef CPP_NN_R_TENSOR
#define CPP_NN_R_TENSOR

#include "CPPNeuralNet/Utils/utils.h"
#include "CPPNeuralNet/Utils/tensor_like.h"
#include "CPPNeuralNet/Utils/tensor_operations.h"

#include <vector>
#include <numeric>                    // for accumulate 
#include <iostream>

namespace cpp_nn {
namespace util {

// Forward Declaration ======================================
namespace {
template<typename T, typename Derived>
class TensorLike; 

template<typename T, typename HeldOperation>
class TransposeOperation;
template<typename T, typename HeldOperation>
class MultiTransposeOperation;
template<typename T, typename HeldOperation1, typename HeldOperation2>
class SummationOperation;
template<typename T, typename HeldOperation1, typename HeldOperation2>
class MultiplicationOperation;
template<typename T, typename HeldOperation>
class ReshapeOperation;
template<typename T, typename HeldOperation>
class PaddingOperation;
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

// Psuedo-Specializations -------------------------
// Some Operations will be optimized from
//    specialization. However, this requires explicit statements.
  template<typename HeldOperation1, typename HeldOperation2>
  rTensor(const TensorLike<T, MultiplicationOperation<T, HeldOperation1, HeldOperation2>>& product_tensor);

  template<typename HeldOperation>
  rTensor(const TensorLike<T, ReshapeOperation<T, HeldOperation>>& reshaped_tensor);
// End of Psuedo-Specializations ------------------
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
  inline ReshapeOperation<T, rTensor<T>> Reshape(const std::vector<int>& new_dimensions) const {
    return {*this, new_dimensions};
  }

  inline PaddingOperation<T, rTensor<T>> Padding(const std::vector<int>& padded_dimensions) const {
    return {*this, padded_dimensions};
  }
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
// End of Housekeeping ---------------------------------

 private:
}; // End of Tensor ==========================================================================

// Extreneous ------------------------------------------
namespace {
template<typename T>
void PrintTensor(const rTensor<T>& tensor, std::vector<int>& idx, int axis);
}

template<typename T>
void PrintTensor(const rTensor<T>& tensor);

/**
 *  cuts matrices into chunks of target_row x target_col.
 *  given [dim... R, C] where each matrix is orignally R x C, cuts the tensor so that 
 *    result is [dim... R/target_row, C/target_col, target_row, target_col]
 *  The cuts are made so that elements that appeared in same block remain in resulting cut matrix.
 *  ie
 *      [4, 4]        
 *      1  2  3  4    
 *      5  6  7  8    
 *      9  10 11 12
 *      13 14 15 16
 *      
 *      cut by 2,2
 *      ->
 *      [2, 2, 2, 2]
 *      1 2    3 4
 *      5 6    7 8
 *      
 *      9 10   11 12
 *      13 14  15 16
 */
template<typename T>
rTensor<T> CutMatrix(const rTensor<T>& tens, int target_row, int target_col);

/**
 *  considering that matrix had been cut, merge it back to large matrices.
 *  ie [dim... R, C, r, c] -> [dim... R*r, C*c]
 */
template<typename T>
rTensor<T> MergeCutMatrix(const rTensor<T>& tens);
// End of Extreneous -----------------------------------

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
      capacity_(  1),
      elements_(  nullptr) {
  // computes chunk_sizes and capacity,   may throw exception
  cpp_nn::util::ComputeCapacityAndChunkSizes(dimensions_, chunk_size_, capacity_);
  elements_ = new std::vector<T>(capacity_, init_val);
}
/** copy Constructor */
template<typename T>
rTensor<T>::rTensor(const rTensor& other) noexcept
    : dimensions_(other.dimensions_),
      chunk_size_(other.chunk_size_),
      capacity_(  other.capacity_),
      elements_(  new std::vector<T>(*other.elements_)) {
  // No exception throwing, as we can assume other is validly costructed
}
/** Move Constrcutro */
template<typename T>
rTensor<T>::rTensor(rTensor&& other) noexcept
    : dimensions_(std::move(other.dimensions_)),
      chunk_size_(std::move(other.chunk_size_)),
      capacity_(  std::move(other.capacity_)),
      elements_(  std::move(other.elements_)) {
  // free up other's pointer
  other.elements_ = nullptr;
}
/**
 *  tensor-like / operation constructor
 */
template<typename T>
template<typename Derived>
rTensor<T>::rTensor(const TensorLike<T, Derived>& tensor_like)
    : rTensor(tensor_like.getShape()) {
  // For each index, assign associated data from operation.
  // TODO: implement a element-wise iterator, or global address system
  //                                          latter may now be possible with AddressToIndex
  std::vector<int> indices(getOrder(), 0);
  do {
    getElement(indices) = tensor_like.getElement(indices);
  } while (IncrementIndicesByShape(getShape().begin(), getShape().end(),
                                   indices.begin(),    indices.end()));
}
// Psuedo-Specializations -------------------------
/** Multiplication Specific Constructor */
template<typename T>
template<typename HeldOperation1, typename HeldOperation2>
rTensor<T>::rTensor(const TensorLike<T, 
                                     MultiplicationOperation<T, 
                                                             HeldOperation1, 
                                                             HeldOperation2>>& product_tensor)
      // Only utilize move constrcutor from pre-computed product                                                        
    : rTensor(std::move( *(product_tensor.getRef().product_tensor_) )) {}

/** Reshape Specific Constructor */
template<typename T>
template<typename HeldOperation>
rTensor<T>::rTensor(const TensorLike<T, 
                                     ReshapeOperation<T, 
                                                      HeldOperation>>& reshaped_tensor)
    // Initialize elements data from previous operation
    : rTensor(std::move(reshaped_tensor.getRef().tensor_like_)) {
  // ReshapeOperation Checks reshape validity upon its construction.
  //    therefore we can construct without checks
  dimensions_ = reshaped_tensor.getRef().dimension_;
  chunk_size_ = reshaped_tensor.getRef().chunk_size_;
  capacity_   = reshaped_tensor.getRef().capacity_;
}
// End of Psuedo-Specializations ------------------
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
//  dually acts as tensor-like behaviour
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
  return (*elements_)[IndicesToAddress(getShape(),
                                       chunk_size_,
                                       indicies)];
}
// End of Accessors -------------------------------------
// End of rTensor DEFINITION =====================================

// Extreneous ------------------------------------------

namespace {
template<typename T>
void PrintTensor(const rTensor<T>& tensor, std::vector<int>& idx, int axis) {
  if (axis == tensor.getOrder() - 1) {
    int dim = tensor.getDimension(-1);
    for (int i = 0; i < dim; ++i) {
      std::cout << tensor.getElement(idx) << ",\t";
      IncrementIndicesByShape(tensor.getShape().begin(), tensor.getShape().end(), idx.begin(), idx.end());
    }
    std::cout << std::endl;
  } else {
    int dim = tensor.getDimension(axis);
    for (int i = 0; i < dim; ++i) {
      PrintTensor(tensor, idx, axis + 1);
    }
    std::cout << std::endl;
  }
}
}

template<typename T>
void PrintTensor(const rTensor<T>& tensor) {
  std::vector<int> idx(tensor.getOrder(), 0);
  PrintTensor(tensor, idx, 0);
}

template<typename T>
rTensor<T> CutMatrix(const rTensor<T>& tens, int target_row, int target_col) {
  if (tens.getOrder() < 2) {
    throw std::invalid_argument("Cut- Insufficient order");
  }

  int curr_row = tens.getDimension(-2);
  int curr_col = tens.getDimension(-1);

  if (curr_row % target_row || curr_col % target_col) {
    throw std::invalid_argument("Cut- Indivisible cut size");
  }

  int chunk_row_count = curr_row / target_row;
  int chunk_col_count = curr_col / target_col;

  std::vector<int> newShape(tens.getShape());
  newShape[newShape.size() - 2] = chunk_row_count;
  newShape[newShape.size() - 1] = target_row;
  newShape.push_back(chunk_col_count);
  newShape.push_back(target_col);
  return tens.Reshape(newShape).Transpose(-2, -3);
}

template<typename T>
rTensor<T> MergeCutMatrix(const rTensor<T>& tens) {
  if (tens.getOrder() < 4) {
    throw std::invalid_argument("Cut- Insufficient order for merge");
  }

  int col = tens.getDimension(-1);
  int row = tens.getDimension(-2);
  int chunk_col_count = tens.getDimension(-3);
  int chunk_row_count = tens.getDimension(-4);

  std::vector<int> newShape(tens.getShape().begin(), tens.getShape().end() - 4);
  newShape.push_back(row * chunk_row_count);
  newShape.push_back(col * chunk_col_count);
  return tens.Transpose(-2, -3).Reshape(newShape);
}

// End of Extreneous -----------------------------------

} // util
} // cpp_nn

#endif // CPP_NN_R_TENSOR
