
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

template<typename T>
class Tensor : public TensorLike<T, Tensor<T>> { // ========================================
 public:
// Public Constructor-----------------------------------
/**
 *  constucts from list of dimensions.
 *  Optionally makes in initial value, which defaults to default T type.
 * 
 *  Dimensions must be positive, and must be at least 1-order.
 */
  Tensor(const std::vector<int>& dimensions, 
         T init_val = T()); 
/**
 *  Copy Constructor
 *  assumes other was validly constructed and thus throws no exception.
 */
  Tensor(const Tensor& other) noexcept;
/**
 *  Move Constructor
 *  assumes other was validly constructed and thus throws no exception.
 */
  Tensor(Tensor&& other) noexcept;

/** 
 *  constructs Tensor from TensorLike objects. 
 * 
 *  Operations-holders are also of this type, and therefore
 *    operations are resolved only upon TensorLike constructor
 *    to minimize movement of data.
 * 
 *  assumes operations were validly constructed and thus throws no exception.
 */
  template<typename Derived>
  Tensor(const TensorLike<T, Derived>& tensor_like) noexcept;

// Psuedo-Specializations -------------------------
/*  Some Operations will be optimized from specialization. 
      However, this requires explicit statements. */
/**
 *  Multiplication Resolution Constructor
 * 
 *  assumes operations were validly constructed and thus throws no exception.
 */
  template<typename HeldOperation1, typename HeldOperation2>
  Tensor(const TensorLike<T, MultiplicationOperation<T, HeldOperation1, HeldOperation2>>& product_tensor) noexcept;
/**
 *  Reshape Resolution Constructor
 * 
 *  assumes operations were validly constructed and thus throws no exception.
 */
  template<typename HeldOperation>
  Tensor(const TensorLike<T, ReshapeOperation<T, HeldOperation>>& reshaped_tensor) noexcept;
// End of Psuedo-Specializations ------------------
// End of Public Constructor----------------------------

// Destrcutor ------------------------------------------
  ~Tensor() noexcept;
// End of Destrcutor -----------------------------------

// Assignment Operators --------------------------------
/** 
 *  Copy assignment
*   Together with Move Constructor, also defines move assignment.
 */
  Tensor& operator=(Tensor other) noexcept; 
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
 *  retrieves reference through vector of indices. 
 *  Each index can be negative for python-style wrap around. 
 *  
 *  As with vector's [] accessor, does not perform checks on validity,
 *    and results in unexpected behaviour.
 */
  inline       T& getElement(const std::vector<int>& indices) noexcept;
  inline const T& getElement(const std::vector<int>& indices) const noexcept;
// End of Accessors -----------------`-------------------

// Modifiers -------------------------------------------
/*  Motifiers can be chained for more optimal operation.
    The intermediate Opertation classes are not intended to be caught
      outside of this class, and should only be used to 
      construct or assign to Tensor */
/**
 *  tranposes given axes. Axes can be negative for python style.
 *  By default, transposes last two axes for matrix transpose.
 */
  inline const TransposeOperation<T, Tensor<T>> Transpose(int axis1 = -2, int axis2 = -1) const {
    return {*this, axis1, axis2};
  }
/**
 *  reshapes to new dimension shape.
 *  The capacity of new dimension must be same as current. 
 */
  inline ReshapeOperation<T, Tensor<T>> 
         Reshape(const std::vector<int>& new_dimensions) const {
    return {*this, new_dimensions};
  }
/**
 *  changes dimension and pads the margins with padded_value.
 *  The dimension can be padded or cropped.
 *    When dimension is smaller, the elements beyond the dimension are lost.
 */
  inline PaddingOperation<T, Tensor<T>> 
         Padding(const std::vector<int>& padded_dimensions, 
                 T padded_value = T()) const {
    return {*this, padded_dimensions, padded_value};
  }
// End of Modifiers ------------------------------------

 protected:
// Member Fields ---------------------------------------
  std::vector<int> dimensions_;
  std::vector<int> chunk_size_;   // capacity of chunk associated with each index
  size_t capacity_;
  std::vector<T>* elements_;      // containter of elements of tensor.
// End of Member fields --------------------------------

// Swap ------------------------------------------------
/** recommended for unifying copy constrcutor and assignment from:
 *    https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
 */
  friend void swap(Tensor& first, Tensor& second) noexcept {
    // The definition seems to necessarily be placed in here, else
    //    linker fails to recognize it.
    std::swap(first.dimensions_, second.dimensions_);
    std::swap(first.chunk_size_, second.chunk_size_);
    std::swap(first.capacity_, second.capacity_);
    std::swap(first.elements_, second.elements_);
  } 
// End of Swap -----------------------------------------
}; // End of Tensor ==========================================================================

// Operations =================================================
/**
 *  returns summation holder.
 */
template<typename T, typename HeldOperation1, 
                     typename HeldOperation2> 
SummationOperation<T, HeldOperation1, HeldOperation2> operator+(const TensorLike<T, HeldOperation1>& A, 
                                                                const TensorLike<T, HeldOperation2>& B) {
  return {A, B};
}
/**
 *  returns multiplication holder.
 */
template<typename T, typename HeldOperation1, 
                     typename HeldOperation2>
MultiplicationOperation<T, HeldOperation1, HeldOperation2> operator*(const TensorLike<T, HeldOperation1>& A,
                                                                     const TensorLike<T, HeldOperation2>& B) {
  return{A, B};
}

// Special Tensor Operations ---------------------------
/* Some Tensor operations which may be useful to have shorthand calls for. */
/**
 *  cuts matrices into blocks of specified arguments. 
 *  The blocks are formed as they appear in the initial tensor.
 *  Given that tensor was of dimension of [dims... R, C] with target block size of r x c,
 *    result will be [dims... R/r, C/c, r, c].
 *  
 *  ie)
 *      [4, 4]        
 *      1  2  3  4    
 *      5  6  7  8    
 *      9  10 11 12
 *      13 14 15 16
 *      
 *      cut as 2x2
 *      ->
 *      [2, 2, 2, 2]
 *      1  2    3  4
 *      5  6    7  8
 *      
 *      9  10   11 12
 *      13 14   15 16
 *  
 *  Requires that R and C are divisible by r and c.
 */
template<typename T>
Tensor<T> CutMatrix(const Tensor<T>& tensor, int block_row, int block_col);
/**
 *  merges back blocks of matrices into a single matrix.
 *  Can be considered as undoing the cut operation. 
 *  ie)
 *    [dim... R, C, r, c] -> [dim... R*r, C*c]
 */
template<typename T>
Tensor<T> MergeCutMatrix(const Tensor<T>& tensor);
// End of Special Tensor Operations --------------------

//  TODO: consider defining external tranpose and reshape and padding as well. 
//  TODO: consider making these specializations of more general Tensor-Operation external methods
// End of Operations ==========================================


// Extreneous ------------------------------------------
namespace {
/** for recursive tensor printing. */
template<typename T>
void PrintTensor(const Tensor<T>& tensor, std::vector<int>& idx, int axis) noexcept;
} // unnamed namespace
/**
 *  prints Tensor into cout.
 *  For each axes, prints additional line_skips.
 */
template<typename T>
void PrintTensor(const Tensor<T>& tensor) noexcept;
// End of Extreneous -----------------------------------

} // util
} // cpp_ nn




namespace cpp_nn {
namespace util {

// Tensor DEFINITION ============================================
// Constructors -----------------------------------------
/** constucts from list of dimensions. */
template<typename T>
Tensor<T>::Tensor(const std::vector<int>& dimensions, 
                  T init_val)
    : dimensions_(dimensions),
      chunk_size_(dimensions.size(), 1),
      capacity_(  1),
      elements_(  nullptr) {
  // Computes chunk_sizes and capacity from dimension. 
  // Also performs checks and can throw exception
  cpp_nn::util::ComputeCapacityAndChunkSizes(dimensions_, 
                                             chunk_size_, 
                                             capacity_);
  elements_ = new std::vector<T>(capacity_, init_val);
}
/** Copy Constructor */
template<typename T>
Tensor<T>::Tensor(const Tensor& other) noexcept
    : dimensions_(other.dimensions_),
      chunk_size_(other.chunk_size_),
      capacity_(  other.capacity_),
      elements_(  new std::vector<T>(*other.elements_)) {}
/** Move Constructor */
template<typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
    : dimensions_(std::move(other.dimensions_)),
      chunk_size_(std::move(other.chunk_size_)),
      capacity_(  std::move(other.capacity_)),
      elements_(  std::move(other.elements_)) {
  // free up other's pointer
  other.elements_ = nullptr;
}
/**
 *  constructs Tensor from TensorLike objects. 
 */
template<typename T>
template<typename Derived>
Tensor<T>::Tensor(const TensorLike<T, Derived>& tensor_like) noexcept
    // pre-allocate and construct by shape of tensor_like
    : Tensor(tensor_like.getShape()) {
  // TODO: implement a element-wise iterator, or global address system
  //                                          latter may now be possible with AddressToIndex

  // Assign each element with associated element from tensor_like
  std::vector<int> indices(getOrder(), 0);
  do {
    getElement(indices) = tensor_like.getElement(indices);
  } while (IncrementIndicesByShape(getShape().begin(), getShape().end(),
                                   indices.begin(),    indices.end()));
}

// Psuedo-Specializations -------------------------
/** Multiplication Resolution Constructor */
template<typename T>
template<typename HeldOperation1, typename HeldOperation2>
Tensor<T>::Tensor(const TensorLike<T, 
                                     MultiplicationOperation<T, 
                                                             HeldOperation1, 
                                                             HeldOperation2>>& product_tensor) noexcept                                            
    // Takes advantage of the way multiplcation operation is implemented. 
    : Tensor(std::move( *(product_tensor.getRef().product_tensor_) )) {}

/** Reshape Resolution Constructor */
template<typename T>
template<typename HeldOperation>
Tensor<T>::Tensor(const TensorLike<T, 
                                     ReshapeOperation<T, 
                                                      HeldOperation>>& reshaped_tensor) noexcept\
    // As reshape only manipulates dimensions, and so can call copy constructor
    //    then alter only the shape
    : Tensor(std::move(reshaped_tensor.getRef().tensor_like_)) {
  // Assumes dimensions and chunk_sizes are validly checked upon ReshapeOperation construction
  dimensions_ = reshaped_tensor.getRef().dimension_;
  chunk_size_ = reshaped_tensor.getRef().chunk_size_;
  capacity_   = reshaped_tensor.getRef().capacity_;
}
// End of Psuedo-Specializations ------------------
// End of Constructors ----------------------------------

// Destructor -------------------------------------------
template<typename T>
Tensor<T>::~Tensor() noexcept {
  delete this->elements_;
}
// End of Destructor ------------------------------------

// Assignment Operators ---------------------------------
/** Copy Operator */
template<typename T>
Tensor<T>& Tensor<T>::operator=(Tensor<T> other) noexcept {
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
inline const std::vector<int>& Tensor<T>::getShape() const noexcept {
  return dimensions_;
}
template<typename T>
inline int Tensor<T>::getDimension(int axis) const {
  // this accomadates for wrap-around behaviours, but 
  //  for sake of optimization, we will bypass usual safety checks
  //  Invalid axis will simply yield UB  
  return dimensions_[SumIfNegative(axis, getOrder())];
}
template<typename T>
inline int Tensor<T>::getOrder() const noexcept {
  return dimensions_.size();
}
template<typename T>
inline size_t Tensor<T>::getCapacity() const noexcept {
  return capacity_;
}
template<typename T>
inline T& Tensor<T>::getElement(const std::vector<int>& indicies) noexcept {
  // Some const cast magic gotten from 
  //  https://stackoverflow.com/questions/856542/elegant-solution-to-duplicate-const-and-non-const-getters
  return const_cast<T&>(const_cast<const Tensor*>(this)->getElement(indicies));
  // This in effect pushes all getter aspect to const getter
}
template<typename T>
inline const T& Tensor<T>::getElement(const std::vector<int>& indicies) const noexcept {
  return (*elements_)[IndicesToAddress(getShape(),
                                       chunk_size_,
                                       indicies)];
}
// End of Accessors -------------------------------------
// End of Tensor DEFINITION =====================================

// Operations =================================================
template<typename T>
Tensor<T> CutMatrix(const Tensor<T>& tensor, int block_row, int block_col) {
  if (tensor.getOrder() < 2) {
    throw std::invalid_argument("Cut- Insufficient order");
  }

  int curr_row = tensor.getDimension(-2);
  int curr_col = tensor.getDimension(-1);

  if (curr_row % block_row || curr_col % block_col) {
    throw std::invalid_argument("Cut- Indivisible cut size");
  }

  int chunk_row_count = curr_row / block_row;
  int chunk_col_count = curr_col / block_col;

  // cut can be achieved by subdividing in each dimension into blocks and block-sizes
  // then transposing to reorder to desired shape
  std::vector<int> cut_shape(tensor.getShape().begin(), tensor.getShape().end() - 2);
  cut_shape.push_back(chunk_row_count);
  cut_shape.push_back(block_row);
  cut_shape.push_back(chunk_col_count);
  cut_shape.push_back(block_col);
  return tensor.Reshape(cut_shape).Transpose(-2, -3);
}

template<typename T>
Tensor<T> MergeCutMatrix(const Tensor<T>& tensor) {
  if (tensor.getOrder() < 4) {
    throw std::invalid_argument("Cut- Insufficient order for merge");
  }
  int col = tensor.getDimension(-1);
  int row = tensor.getDimension(-2);
  int chunk_col_count = tensor.getDimension(-3);
  int chunk_row_count = tensor.getDimension(-4);

  std::vector<int> merged_shape(tensor.getShape().begin(), tensor.getShape().end() - 4);
  merged_shape.push_back(row * chunk_row_count);
  merged_shape.push_back(col * chunk_col_count);
  return tensor.Transpose(-2, -3).Reshape(merged_shape);
}
// End of Operations ==========================================

// Extreneous ------------------------------------------
namespace {
/* for recurrsion only */
template<typename T>
void PrintTensor(const Tensor<T>& tensor, std::vector<int>& idx, int axis) noexcept {
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
    std::cout << "." << std::endl;
  }
}
} // unnamed namespace

template<typename T>
void PrintTensor(const Tensor<T>& tensor) noexcept {
  std::vector<int> idx(tensor.getOrder(), 0);
  PrintTensor(tensor, idx, 0);
}
// End of Extreneous -----------------------------------

} // util
} // cpp_nn

#endif // CPP_NN_R_TENSOR
