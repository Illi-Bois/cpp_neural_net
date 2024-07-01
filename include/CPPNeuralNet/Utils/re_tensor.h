
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
template<typename T>
class rTensor;
// End of Forward Declaration ===============================

/**
 * Unnamed namespace for all local structs.
 */
namespace {
/**
 *  for CRTP
 */
template<typename Derived>
struct Base {
  const Derived& getRef() const {
    return static_cast<const Derived&>(*this);
  }
};


template<typename T, typename HolderType>
class TransposeHolder : public Base<TransposeHolder<T, HolderType>> {
  const HolderType& tensor_;
  const int axis_1_;
  const int axis_2_;

 public:
  TransposeHolder(const HolderType& tensor, const int axis_1, const int axis_2)
      : tensor_(tensor), 
        axis_1_(axis_1 + (axis_1 < 0 ? tensor.getOrder() : 0)), 
        axis_2_(axis_2 + (axis_2 < 0 ? tensor.getOrder() : 0)) {
  }

  const T& getElement(std::vector<int>& indices) const {
    std::swap(indices[axis_1_], indices[axis_2_]);
    const T& res = tensor_.getElement(indices);
    // swap back
    std::swap(indices[axis_1_], indices[axis_2_]);
    return res;
  }

  std::vector<int> getShape() const {
    std::vector<int> original_shape = tensor_.getShape();
    std::swap(original_shape[axis_1_], original_shape[axis_2_]);
    return original_shape;
  }

  inline int getOrder() const {
    return tensor_.getOrder();
  }

  TransposeHolder<T, TransposeHolder<T, HolderType>> Tranpose(int axis_1, int axis_2) {
    return {*this, axis_1, axis_2};
  }
};

} // unnamed


template<typename T>
class rTensor : public Base<rTensor<T>> { // ========================================
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
 *  Tranpose Constructor
 */
  template<typename Holder>
  rTensor(const TransposeHolder<T, Holder>& tranpose_holder);
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
/** 
 *  computes element-wise sum of two tensors.
 *  When dimension is mismatched, then performs broadcasted sum. 
 */
  friend rTensor operator+(const rTensor& A, const rTensor& B);
/** 
 *  computes matrix multiplication of tensors,
 *    traeting each as multi-array of matrices. 
 *  When dimension mistahced, then produces dimension according to:
 *    [dim1..., r, k] x [dim2...., k, c] -> [dim1, dim2, r, c] 
 */
  friend rTensor operator*(const rTensor& A, const rTensor& B);
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
template<typename Holder>
rTensor<T>::rTensor(const TransposeHolder<T, Holder>& holder)
    : rTensor(holder.getShape()) {
  // with tensor initized to 0s, set each element one by one

  std::vector<int> indices(getOrder(), 0);

  do {
    getElement(indices) = holder.getElement(indices);
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
