
#ifndef CPP_NN_R_TENSOR
#define CPP_NN_R_TENSOR

#include "CPPNeuralNet/Utils/utils.h"

#include <vector>
#include <numeric>                    // for accumulate 

namespace cpp_nn {
namespace util {

/**
 * 
 * FUTURE CONSIDERATIONS:
 * TODO:
 * 
 * should we use size_t instead of int?
 */

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

template<typename T>
class rTensor;
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


/* Operation Facilitators ---------------------------------------------------------- */
/*    
  Intermediate classes to be passed to reduce movement of tensor data
  These classes will appear as return type of operations such as Tranpose and summation,
    and only upon assignment to a Tensor will data be moved.
  There are exception to these rules, such as with multiplcation which 
    may benefit from operation itself allocating space for computation and storage,
    but it too will follow the design of the other operations.
 */


template<typename T, typename HeldOperation>
class TransposeOperation : public TensorLike<T, TransposeOperation<T, HeldOperation>> {
// Members ---------------------------------------------
  const HeldOperation& tensor_;
  const int axis_1_;
  const int axis_2_;

  std::vector<int> dimension_;    // dimension post-tranposing
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  TransposeOperation(const TensorLike<T, HeldOperation>& tensor_like, 
                     const int axis_1, const int axis_2)
      : tensor_(tensor_like.getRef()), 
        // axis are stored as positive int, though it can be given as negative
        axis_1_(axis_1 + (axis_1 < 0 ? tensor_like.getOrder() : 0)), 
        axis_2_(axis_2 + (axis_2 < 0 ? tensor_like.getOrder() : 0)),
        dimension_(tensor_like.getShape()) {
    std::swap(dimension_[axis_1_], dimension_[axis_2_]);
  }
// End of Constructor ----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const {
    return dimension_;
  }
  inline int getDimension(int axis) const {
    return dimension_[axis + (axis < 0 ? getOrder() : 0)];
  }
  inline int getOrder() const noexcept {
    return dimension_.size();
  }  
  // indices for tranpose is passed as value, not reference,
  //   so that tranposed axis may be swapped. 
  const T getElement(std::vector<int> indices) const {
    std::swap(indices[axis_1_], indices[axis_2_]);
    return tensor_.getElement(indices);
  }
  inline const TransposeOperation<T, TransposeOperation<T, HeldOperation>> 
               Tranpose(int axis_1, int axis_2) const {
    return {*this, axis_1, axis_2};
  }
// End of Tensor-Behaviours ----------------------------
}; // End of TransposeOperation


// TODO: Currently only handles same-shape addition.
template<typename T, typename HeldOperation1, typename HeldOperation2>
class SummationOperation : public TensorLike<T, SummationOperation<T, HeldOperation1, HeldOperation2>> {
// Members ---------------------------------------------
    const HeldOperation1& first_;
    const HeldOperation2& second_;
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  SummationOperation(const TensorLike<T, HeldOperation1>& first, 
                     const TensorLike<T, HeldOperation2>& second) 
      : first_(first.getRef()), 
        second_(second.getRef()) {
    if (first_.getOrder() != second_.getOrder()) {
      throw std::invalid_argument("Summation Error- Order Mismatch.");
    }
    if (first_.getShape() != second_.getShape()) {
      throw std::invalid_argument("Summation Error- Dimension Mismatch.");
    }
  }
// End of Constructor ----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    // Either shapes must be equal
    return first_.getShape();
  } 
  inline int getDimension(int axis) const {
    // Either shapes must be equal
    return first_.getDimension(axis);
  }
  inline int getOrder() const noexcept {
    // Either shapes must be equal
    return first_.getOrder();
  }
  inline const T getElement(const std::vector<int>& indices) const {
    return first_.getElement(indices) + second_.getElement(indices);
  }
  inline const TransposeOperation<T, SummationOperation<T, HeldOperation1, HeldOperation2>> 
               Transpose(int axis1 = -2, int axis2 = -1) const {
    return {*this, axis1, axis2};
  }
// End of Tensor-Behaviours ----------------------------
}; // End of SummationOperation

// TODO: Currently handles same-shape multiplication
//    will scale up to allow broad-casting (not all combination product)
template<typename T, typename HeldOperation1, typename HeldOperation2>
class MultiplicationOperation : public TensorLike<T, MultiplicationOperation<T, HeldOperation1, HeldOperation2>> {
// Members ---------------------------------------------
  rTensor<T>* element_;
  std::vector<int> product_shape_; 
// End of Members --------------------------------------

// Housekeeping Methods --------------------------------
/** 
 *  increments given index to iterate through each matrix in tensor.
 *  Given index can be any length, only the frist order - 2 will be used. 
 *    This means last two axis are preserved, when full length index is given.
 * 
 *  Returns true when incrementation was successful, false when failed.
 *    Upon failure, the used indices are set to 0. 
 */
  bool MatrixIncrementIndices(std::vector<int>& indices) {
    for (int axis = product_shape_.size() - 2; axis >= 0; --axis) {
      if (++indices[axis] >= product_shape_[axis]) {
        indices[axis] = 0;
      } else {
        return true;
      }
    }
    return false;
  }
  // TODO: Merge this into incrementIndices method
// End of Housekeeping Methods -------------------------

 public:
// Constructor -----------------------------------------
  // TODO, with this design, a special construcor for rTensor may be prefered for Multiplcation Holder, where a simple move is prefered
  // Product is handled upon constrcution
  MultiplicationOperation(const TensorLike<T, HeldOperation1>& A, 
                          const TensorLike<T, HeldOperation2>& B)
      : element_(nullptr),
        product_shape_(A.getShape()) /* dummy for later*/ {
    // assert  [dim1,  r, k] [dim2, k, c] that dim == dim1
    if (A.getOrder() != B.getOrder()) {
      throw std::invalid_argument("Multiplcation- Order Mismatch");
    }
    if (A.getOrder() < 2) {
      throw std::invalid_argument("Multiplication- Insufficient Dimension");
    }
    if (!std::equal(A.getShape().begin(), A.getShape().end() - 2, 
                    B.getShape().begin())) {
      throw std::invalid_argument("Multiplication- Dimension Mismatch");
    }
    if (A.getDimension(-1) != B.getDimension(-2)) {
      throw std::invalid_argument("Multiplication- Multiplcation Dimension Mismatch");
    }

    const int rows = A.getDimension(-2);
    const int cols = B.getDimension(-1);
    const int interm = A.getDimension(-1);

    // product_shape is aleady set as being A
    product_shape_[getOrder() - 1] = cols;

    element_ = new rTensor<T>(product_shape_);
    // indices for which matrix is to be chosen
    std::vector<int> indices(getOrder(), 0);

    do {
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          indices[getOrder() - 2] = r;
          indices[getOrder() - 1] = c;

          T& element = element_->getElement(indices);
          element = T();

          for (int k = 0; k < interm; ++k) {
            indices[getOrder() - 2] = r;
            indices[getOrder() - 1] = k;
            T a_element = A.getElement(indices);

            indices[getOrder() - 2] = k;
            indices[getOrder() - 1] = c;
            T b_element = B.getElement(indices);

            element += a_element * b_element;
          }
        }
      }
    } while (IncrementIndicesByShape(product_shape_.begin(), product_shape_.end() - 2,
                                     indices.begin(),        indices.end() - 2));
  }
// End of Constructor ----------------------------------

// Destructor ------------------------------------------
  ~MultiplicationOperation() {
    delete element_;
  }
// End of Destructor -----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    return product_shape_;
  } 
  inline int getDimension(int axis) const {
    return product_shape_[axis + (axis < 0 ? getOrder() : 0)];
  }
  inline int getOrder() const noexcept {
    return product_shape_.size();
  }
  inline const T getElement(const std::vector<int>& indices) const {
    return element_->getElement(indices);
  }
  inline const TransposeOperation<T, MultiplicationOperation<T, HeldOperation1, HeldOperation2>> 
               Transpose(int axis1 = -2, int axis2 = -1) const {
    return {*this, axis1, axis2};
  }
// End of Tensor-Behaviours ----------------------------
}; // End of MultiplicationOperation

// BROADCASTING MIGHT BE DONE IN THIS MANNER AS WELL!

/* End of Operation Facilitators --------------------------------------------------- */
} // End of unnamed namespace ================================================================


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
  TransposeOperation<T, rTensor<T>> Transpose(int axis1 = -2, int axis2 = -1) const;
/**
 *  reshapes to new dimension shape.
 *  The capacity of new dimension must be same as current. 
 */
  void Reshape(const std::vector<int> new_dimensions);
// End of Modifiers ------------------------------------

 protected:
// Member Fields ---------------------------------------
  std::vector<int> dimensions_;
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
// End of Housekeeping ---------------------------------

 private:
}; // End of Tensor ==========================================================================

// Tensor Operations --------------------------------------
/**
 *  Summs elementwise if dimensions all match.
 *  Will impement broadcasting later.
 */
template<typename T, typename Holder1, typename Holder2> 
SummationOperation<T, Holder1, Holder2> operator+(const TensorLike<T, Holder1>& A, const TensorLike<T, Holder2>& B);
/** 
 * Multiplication done only if multi-array of matrices match
 * Will implement broadcasting later
 */
template<typename T, typename Holder1, typename Holder2>
MultiplicationOperation<T, Holder1, Holder2> operator*(const TensorLike<T, Holder1>& A, const TensorLike<T, Holder2>& B);
// End of Tensor Operation --------------------------------


// Extreneous ---------------------------------------------
// End of Extreneous --------------------------------------
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
    try : dimensions_(dimensions),
          // as capacity_ is constant, must be initiazed this way
          capacity_(std::accumulate(dimensions.begin(),
                                    dimensions.end(),
                                    1, std::multiplies<int>())),
          // vector initialization with invalid capapcity will throw vector_excp
          //  which will be caught by function-try-block and throw our own
          elements_(new std::vector<T>(capacity_, init_val)) {
  // only need dimension check
  for (const int& dimension : dimensions_) {
    if (dimension <= 0) {
      // throw same error as vector's then let outer catch handle it all
      throw std::invalid_argument("TensorElement Constructor- Non-positive dimension given");
    }
  }
} catch(std::length_error err) {
  throw std::invalid_argument("TensorElement Constructor- Non-positive dimension given");
}
/** copy Constructor */
template<typename T>
rTensor<T>::rTensor(const rTensor& other) noexcept
    : dimensions_(other.dimensions_),
      capacity_(other.capacity_),
      elements_(new std::vector<T>(*other.elements_)) {
  // No exception throwing, as we can assume other is validly costructed
}
/** Move Constrcutro */
template<typename T>
rTensor<T>::rTensor(rTensor&& other) noexcept
    : dimensions_(std::move(other.dimensions_)),
      capacity_(std::move(other.capacity_)),
      elements_(std::move(other.elements_)) {
  // free up other's pointer
  other.elements_ = nullptr;
}

/** 
 *  Tranpose Constrcutor
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
  return dimensions_[axis + (axis < 0 ? getOrder() : 0)];
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
  const size_t& address = ConvertToAddress(indicies);
  return (*elements_)[address];
}
// End of Accessors -------------------------------------


// Modifiers --------------------------------------------
template<typename T>
TransposeOperation<T, rTensor<T>> rTensor<T>::Transpose(int axis1, int axis2) const {
  return TransposeOperation<T, rTensor<T>>{*this, axis1, axis2};
}

/**
 *  Summs elementwise if dimensions all match.
 *  Will impement broadcasting later.
 */
template<typename T, typename Holder1, typename Holder2> 
SummationOperation<T, Holder1, Holder2> operator+(const TensorLike<T, Holder1>& A, const TensorLike<T, Holder2>& B) {
  return {A, B};
}

template<typename T, typename Holder1, typename Holder2>
MultiplicationOperation<T, Holder1, Holder2> operator*(const TensorLike<T, Holder1>& A, const TensorLike<T, Holder2>& B) {
  return{A, B};
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
  size_t block_size = 1;

  // as dimension accepts int, we should keep axis as int as well
  for (int axis = getOrder() - 1; axis >= 0; --axis) {
    const int& curr_idx = indices[axis];
    const int& curr_dim = getDimension(axis);

    if (curr_idx >= -curr_dim && curr_idx < curr_dim) {
      address += (curr_idx + (curr_idx < 0 ? curr_dim : 0)) * block_size;
      block_size *= curr_dim;
    } else {
      throw std::invalid_argument("TensorIndex- Index out of bound");
    }
  }

  return address;
}
// End of Housekeeping ----------------------------------
// End of rTensor DEFINITION =====================================


// Extreneous ---------------------------------------------
// End of Extreneous --------------------------------------


} // util
} // cpp_nn

#endif // CPP_NN_R_TENSOR
