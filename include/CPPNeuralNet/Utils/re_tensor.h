
#ifndef CPP_NN_R_TENSOR
#define CPP_NN_R_TENSOR

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
template<typename T, typename HolderType>
class TransposeHolder;
}

template<typename T>
class rTensor;
// End of Forward Declaration ===============================

/**
 * Unnamed namespace for all local structs.
 */
namespace {
/**
 *  is Interace for objects with Tensor-like behaviours.
 *  With CRTP, provides static polymorphic interactions.
 * 
 *  Will provide all const interface for Tensor-Like behaviour.
 */
template<typename T, typename Derived>
class TensorLike {
 public:
/** 
 * downcasting of current Object.
 */
  inline const Derived& getRef() const {
    return static_cast<const Derived&>(*this);
  }

  inline const std::vector<int>& getShape() const noexcept {
    return getRef().getShape();
  }

  inline int getDimension(int axis) const {
    return getRef().getDimension(axis);
  }

  inline int getOrder() const noexcept {
    return getRef().getOrder();
  }

  inline const T getElement(const std::vector<int>& indices) const {
    return getRef().getElement(indices);
  }

  const TransposeHolder<T, Derived> Transpose(int axis1 = -2, int axis2 = -1) const {
    return getRef().Transpose(axis1, axis2);
  }

};


template<typename T, typename HolderType>
class TransposeHolder : public TensorLike<T, TransposeHolder<T, HolderType>> {
  const HolderType& tensor_;
  const int axis_1_;
  const int axis_2_;

  std::vector<int> dimension_;

 public:
  TransposeHolder(const TensorLike<T, HolderType>& tensor, const int axis_1, const int axis_2)
      : tensor_(tensor.getRef()), 
        axis_1_(axis_1 + (axis_1 < 0 ? tensor.getOrder() : 0)), 
        axis_2_(axis_2 + (axis_2 < 0 ? tensor.getOrder() : 0)),
        dimension_(tensor.getShape()) {
    std::swap(dimension_[axis_1_], dimension_[axis_2_]);
  }



  inline const std::vector<int>& getShape() const {
    return dimension_;
  }

  inline int getDimension(int axis) const {
    return dimension_[axis + (axis < 0 ? getOrder() : 0)];
  }

  inline int getOrder() const noexcept {
    return dimension_.size();
  }  

  // this is non-reference now as manipulating index alone is easier to pass down
  const T getElement(std::vector<int> indices) const {
    std::swap(indices[axis_1_], indices[axis_2_]);
    const T& res = tensor_.getElement(indices);
    // swap back
    return res;
  }

  const TransposeHolder<T, TransposeHolder<T, HolderType>> Tranpose(int axis_1, int axis_2) const {
    return {*this, axis_1, axis_2};
  }
};

// FOR NOW, PURE ELEMENTWISE SUMMATION
template<typename T, typename HolderType1, typename HolderType2>
class SummationHolder : public TensorLike<T, SummationHolder<T, HolderType1, HolderType2>> {
    const HolderType1& first_;
    const HolderType2& second_;
 public:
  SummationHolder(const TensorLike<T, HolderType1>& first, const TensorLike<T, HolderType2>& second) 
      : first_(first.getRef()), second_(second.getRef()) {
    // check for dimension match
    // TODO....?
    if (first_.getOrder() != second_.getOrder()) {
      throw std::invalid_argument("Summation Error- Order Mismatch.");
    }
    for (int i = 0; i < first_.getOrder(); ++i) {
      if (first_.getDimension(i) != second_.getDimension(i)) {
        throw std::invalid_argument("Summation Error- Dimension Mismatch.");
      }
    }
  }


  inline const std::vector<int>& getShape() const noexcept {
    // as first and second are the same
    return first_.getShape();
  } 

  inline int getDimension(int axis) const {
    return first_.getDimension(axis);
  }

  inline int getOrder() const noexcept {
    return first_.getOrder();
  }

  inline const T getElement(const std::vector<int>& indices) const {
    return first_.getElement(indices) + second_.getElement(indices);
  }

  inline TransposeHolder<T, SummationHolder<T, HolderType1, HolderType2>> Transpose(int axis1 = -2, int axis2 = -1) const {
    return {*this, axis1, axis2};
  }

};

// instead of all-combination tensor mult, 
//  we will choose matrix-broadcast shape propagation
// As for initial proof of concept, we will only allow if all
//    multi-array shape matches
template<typename T, typename HolderType1, typename HolderType2>
class MultiplicationHolder : public TensorLike<T, MultiplicationHolder<T, HolderType1, HolderType2>> {
  rTensor<T> element_;
  std::vector<int> product_shape_; 

  // again, right now assumes same multi array shape
  //  indices may either be order or order - 2.
  bool MatrixIncrementIndices(std::vector<int>& indices) {
    int axis = product_shape_.size() - 2;
    // overall repeat of increment for Tensor, and ideally would want to clean up
    while(axis >= 0) {
      if (++indices[axis] >= product_shape_[axis]) {
        indices[axis] = 0;
        --axis;
      } else {
        return true;
      }
    }
    return false;
  }
 public:

  // TODO, with this design, a special construcor for rTensor may be prefered for Multiplcation Holder, where a simple move is prefered
  // Product is handled upon constrcution
  MultiplicationHolder(const TensorLike<T, HolderType1>& A, const TensorLike<T, HolderType2>& B)
      : element_({1}) /* dummy for later*/ {
    // assert  [dim1,  r, k] [dim2, k, c] that dim == dim1
    if (A.getOrder() != B.getOrder()) {
      throw std::invalid_argument("Multiplcation- Order Mismatch");
    }
    if (A.getOrder() < 2) {
      throw std::invalid_argument("Multiplication- Insufficient Dimension");
    }

    int capacity = 1;

    // as for now order therefore must be equal 
    product_shape_ = std::vector<int>(A.getOrder());
    for (int i = 0; i < product_shape_.size() - 2; ++i) {
      if (A.getDimension(i) == B.getDimension(i)) {
        product_shape_[i] = A.getDimension(i);
        capacity *= product_shape_[i];
      } else {
        throw std::invalid_argument("Multiplication- Dimension Mismatch");
      }
    }

    if (A.getDimension(-1) != B.getDimension(-2)) {
      throw std::invalid_argument("Multiplication- Multiplcation Dimension Mismatch");
    }

    const int interm = A.getDimension(-1);
    const int rows = A.getDimension(-2);
    const int cols = B.getDimension(-1);

    product_shape_[product_shape_.size() - 2] = rows;
    capacity *= rows;
    product_shape_[product_shape_.size() - 1] = cols;
    capacity *= cols;


    element_ = rTensor<T>(product_shape_);
    // TODO implement product
    std::vector<int> a_idx(getOrder(), 0);
    std::vector<int> b_idx(getOrder(), 0);
    std::vector<int> c_idx(getOrder(), 0);

    do {
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          c_idx[getOrder() - 2] = r;
          c_idx[getOrder() - 1] = c;

          element_.getElement(c_idx) = T();

          a_idx[getOrder() - 2] = r;

          b_idx[getOrder() - 1] = c;
          for (int k = 0; k < interm; ++k) {
            a_idx[getOrder() - 1] = k;
            b_idx[getOrder() - 2] = k;

            element_.getElement(c_idx) += A.getElement(a_idx) * B.getElement(b_idx);
          }
        }
      }

      MatrixIncrementIndices(a_idx);
      MatrixIncrementIndices(b_idx);
    } while (MatrixIncrementIndices(c_idx));
  }


  inline const std::vector<int>& getShape() const noexcept {
    // as first and second are the same
    return product_shape_;
  } 

  inline int getDimension(int axis) const {
    return product_shape_[axis + (axis < 0 ? getOrder() : 0)];
  }

  inline int getOrder() const noexcept {
    return product_shape_.size();
  }

  inline const T getElement(const std::vector<int>& indices) const {
    return element_.getElement(indices);
  }

  inline TransposeHolder<T, MultiplicationHolder<T, HolderType1, HolderType2>> Transpose(int axis1 = -2, int axis2 = -1) const {
    return {*this, axis1, axis2};
  }
};

// BROADCASTING MIGHT BE DONE IN THIS MANNER AS WELL!

} // unnamed


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
  TransposeHolder<T, rTensor<T>> Transpose(int axis1 = -2, int axis2 = -1) const;
/**
 *  reshapes to new dimension shape.
 *  The capacity of new dimension must be same as current. 
 */
  void Reshape(const std::vector<int> new_dimensions);
// End of Modifiers ------------------------------------

// Tensor Operations -----------------------------------
// /** 
//  *  computes element-wise sum of two tensors.
//  *  When dimension is mismatched, then performs broadcasted sum. 
//  */
//   template<typename Holder1, typename Holder2>
//   friend SummationHolder<T, Holder1, Holder2>  operator+(const TensorLike<T, Holder1>& A, const TensorLike<T, Holder2>& B);
// TODO: MAYBE THIS SHOULD NOT BE FRIEND AT ALL
// /** 
//  *  computes matrix multiplication of tensors,
//  *    traeting each as multi-array of matrices. 
//  *  When dimension mistahced, then produces dimension according to:
//  *    [dim1..., r, k] x [dim2...., k, c] -> [dim1, dim2, r, c] 
//  */
//   friend rTensor operator*(const rTensor& A, const rTensor& B);
// End of Tensor Operations ----------------------------

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
}; // End of Tensor =======================================

// Extreneous ---------------------------------------------
/**
 * increments passed indicies to loop the shape given.
 * When incrementation is successful, returns true.
 * If incrementatoin fails, ie) loops back to 0, returns false.
 *  So the return value can be used in while loop.
 * 
 * When return is false, indicies are reset to 0, and so ready to increment again.
 */
bool incrementIndices(std::vector<int>& indices, const std::vector<int>& shape);
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
  } while (incrementIndices(indices, getShape()));
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
TransposeHolder<T, rTensor<T>> rTensor<T>::Transpose(int axis1, int axis2) const {
  return TransposeHolder<T, rTensor<T>>{*this, axis1, axis2};
}

/**
 *  Summs elementwise if dimensions all match.
 *  Will impement broadcasting later.
 */
template<typename T, typename Holder1, typename Holder2> 
SummationHolder<T, Holder1, Holder2> operator+(const TensorLike<T, Holder1>& A, const TensorLike<T, Holder2>& B) {
  return {A, B};
}

template<typename T, typename Holder1, typename Holder2>
MultiplicationHolder<T, Holder1, Holder2> operator*(const TensorLike<T, Holder1>& A, const TensorLike<T, Holder2>& B) {
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
bool incrementIndices(std::vector<int>& indices, const std::vector<int>& shape) {
  int axis = indices.size() - 1; 
  while (axis >= 0) {
    if (++indices[axis] >= shape[axis]) {
      // loop back to 0 and continue on next axis down
      indices[axis] = 0;
      --axis;
    } else {
      return true;
    }
  }
  // loop continued for all axis thus failed
  return false;
}
// End of Extreneous --------------------------------------


} // util
} // cpp_nn

#endif // CPP_NN_R_TENSOR
