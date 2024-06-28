
#ifndef CPP_NN_R_TENSOR
#define CPP_NN_R_TENSOR

#include <vector>
#include <numeric>                    // for accumulate 

namespace cpp_nn {
namespace util {

template<typename T>
class rTensor { // ========================================
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
  rTensor(const rTensor& other);
/**
 *  Move Constructor
 */
  rTensor(rTensor&& other);
// End of Public Constructor----------------------------

// Destrcutor ------------------------------------------
  ~rTensor();
// End of Destrcutor -----------------------------------

// Assignment Operators --------------------------------
/** Copy operator */
  rTensor& operator=(rTensor other); 
/** Move operator */
  rTensor& operator=(rTensor&& other); 
// End of Assignment Operators -------------------------

// Accessors -------------------------------------------
/**
 *  retrieves vector representing diemension of tensor.
 */
  const std::vector<int>& getShape() const;
/** 
 *  retrieves dimension at given axis. 
 *  The axis can be nagative to wrap around python style. 
 */
  int getDimension(int axis) const;
/** 
 *  retrieves order of the tensor. 
 */
  int getOrder() const;
/**
 *  retrieves total capacity of Tensor
 */
  int getCapacity() const;
/** 
 *  retrieves reference through vector of index. 
 */
  T& getElement(const std::vector<int>& indices);
  const T& getElement(const std::vector<int>& indices) const;
// End of Accessors ------------------------------------

// Modifiers -------------------------------------------
/**
 *  tranposes given axes. Axes can be negative for python style.
 *  By default, transposes last two axes. 
 *    This conforms to Matrix tranpose.
 */
  void Transpose(int axis1 = -2, int axis2 = -1);
/**
 *  reshapes to new dimension shape.
 *  The capacity of new dimension must be same as current. 
 */
  void Reshape(const std::vector<int> new_dimensions);
// End of Modifiers ------------------------------------

 protected:
// Member Fields ---------------------------------------
  std::vector<int> dimensions_;
  const int capacity_;
  std::vector<T>* elements_;
// End of Member fields --------------------------------

// Swap ------------------------------------------------
/** recommended for unifying copy constrcutor and operator from:
 *    https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
 */
  friend void swap(rTensor& first, rTensor& second);
// End of Swap -----------------------------------------

 private:
}; // End of Tensor =======================================

// Tensor Operators -------------------------------------
/** 
 *  computes element-wise sum of two tensors.
 *  When dimension is mismatched, then performs broadcasted sum. 
 */
template<typename T>
rTensor<T> operator+(const rTensor<T>& A, const rTensor<T>& B);
/** 
 *  computes matrix multiplication of tensors,
 *    traeting each as multi-array of matrices. 
 *  When dimension mistahced, then produces dimension according to:
 *    [dim1..., r, k] x [dim2...., k, c] -> [dim1, dim2, r, c] 
 */
template<typename T>
rTensor<T> operator*(const rTensor<T>& A, const rTensor<T>& B);
// End of Tensor Operators ------------------------------

} // util
} // cpp_ nn



// rTensor DEFINITION ============================================
namespace cpp_nn {
namespace util {

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
      throw std::invalid_argument("TensorElement Constructor- Non-Positive Dimension Error");
    }
  }
} catch(std::length_error err) {
  throw std::invalid_argument("TensorElement Constructor- Non-Positive Dimension Error");
}
/** copy Constructor */
template<typename T>
rTensor<T>::rTensor(const rTensor& other) 
    : dimensions_(other.dimensions_),
      capacity_(other.capacity_),
      elements_(new std::vector<T>(*other.elements_)) {
  // No exception throwing, as we can assume other is validly costructed
}
// End of Constructors ----------------------------------

// Destructor -------------------------------------------
template<typename T>
rTensor<T>::~rTensor() {
  delete this->elements_;
}
// End of Destructor ------------------------------------

// Assignment Operators ---------------------------------
/** Copy Operator */
template<typename T>
rTensor<T>& rTensor<T>::operator=(rTensor<T> other) {
  // uses copy-swap idiom
  // As i undersatnd it, by relying on copy-constrcutor and swap
  //    we let paramater be initated with copy constrcutor, then 
  //    call swap to swap that new instance with current
  swap(*this, other);
  return *this;
}
// End of Assignment Operators --------------------------



// Swap ------------------------------------------------
template<typename T>
void swap(rTensor<T>& first, rTensor<T>& second) {
  std::swap(first.dimensions_, second.dimensions_);
  std::swap(first.capacity_, second.capacity_);
  std::swap(first.elements_, second.elements_);
}
// End of Swap -----------------------------------------

} // util
} // cpp_nn
// End of rTensor DEFINITION =====================================

#endif // CPP_NN_R_TENSOR
