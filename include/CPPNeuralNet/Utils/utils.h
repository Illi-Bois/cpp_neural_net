#ifndef CPP_NN_UTIL
#define CPP_NN_UTIL


#include <vector>
#include <initializer_list>

namespace cpp_nn {

namespace util {

/**
 * Lowest level math object for NN. Vectors will be treated as special case of matrices.
*/

template<typename T= double>
class Matrix {
 private:
  int num_rows_, num_cols_; 
  std::vector<std::vector<T>> elements_; // elements[row][col]
 public:
  /**
   * Constructor with Initial Value
  */
  Matrix(int num_rows, int num_cols, T initial_value = T());
  /**
   * Constructor with Initializer list
  */
  Matrix(std::initializer_list<std::initializer_list<T>> list);
  /**
   * Returns another instance of matrix with this * other; 
   * Edits made : put const infront; apparantly Matrix& const other = Matrix& as references are constant inherently
   * Const Overloading : 
   * const Before Return Type: Ensures the returned reference is a reference to a const vector, preventing modification of the vector's contents through this reference. 
   * const After Function Signature: Ensures the member function can be called on const instances and does not modify the state of the object.
  */
  Matrix operator*(const Matrix& other) const;
  const std::vector<T>& operator[](int index) const;
  std::vector<T>& operator[](int index);
  static T dot(const std::vector<T> v1, const std::vector<T> v2);
  
};

/**
 * Inputs and outputs will be prepresented by vectors.
 * 
 * 
*/
template<typename T=double>
class Vector : public Matrix<T> {
 private:
 public:
  Vector(int dim, T initial_value = T())
      : Matrix<T>(dim, 1, initial_value) {};
};



// TODO implement tensors. 

} // util

} // cpp_nn


#include "src/CPPNeuralNet/Utils/Utils.tpp";

#endif  // CPP_NN_UTIL
