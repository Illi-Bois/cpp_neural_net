
#ifndef CPP_NN_R_TENSOR
#define CPP_NN_R_TENSOR

#include <vector>
#include <initializer_list>

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
  rTensor(const std::initializer_list<int>& dimensions, 
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

// Accessors -------------------------------------------
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



} // util
} // cpp_nn
// End of rTensor DEFINITION =====================================

#endif // CPP_NN_R_TENSOR
