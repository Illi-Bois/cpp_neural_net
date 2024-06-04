#ifndef CPP_NN_UTIL
#define CPP_NN_UTIL


#include <vector>
#include <initializer_list>

namespace cpp_nn {

namespace util {

/**
 * Lowest level math object for NN. Vectors will be treated as special case of matrices.
*/

// TODO organize methods and locations
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
  

  // Copy constrcutor. 
  // Deep copy of other
  Matrix(Matrix& const other)
      : num_rows_(other.num_rows), num_cols_(other.num_cols),
        elements_(other.elements) {}
  
  // Copy Operator.
  //   Note: it is essential that copy operator and constrcutor be different. 
  Matrix& operator=(const Matrix& other);

  // Updates elements of self to be this * B.
  // Notice that MatMul inheritly updates current's elements
  Matrix& MatMul(Matrix& const B);
 
  /**
   * Returns another instance of matrix with this * other; 
   * Edits made : put const infront; apparantly Matrix& const other = Matrix& as references are constant inherently
   * Const Overloading : 
   * const Before Return Type: Ensures the returned reference is a reference to a const vector, preventing modification of the vector's contents through this reference. 
   * const After Function Signature: Ensures the member function can be called on const instances and does not modify the state of the object.
  */
  Matrix operator*(const Matrix& other) const;

  // ! Notice that MatMul and * are intrinsictally connected. Meaning We can define one nith the orther.
  //   ie. MatMul(...) := (*this) = (*this * other)
  //       operator*(...) := (Matrix(*this)).MatMul(other)


  inline T& getElement(const int row, const int col) {
    return elements_[row][col];
  }
  T& operator()(const int row, const int col) {
    return getElement(row, col);
  }

  int getNumRows() const{
    return num_rows_;
  }
  int getNumCols() const{
    return num_cols_;
  }

  // ! I would recommend not parsing matrix like this. Rather use the above Parenthesis indexing which allows multiple arguements, such as matrix(row, col);
  // const std::vector<T>& operator[](int index) const;
  // std::vector<T>& operator[](int index);
  
  // ! Dot should be member of Vector calss. Further, it seems like you made typo where it should be Vector not vector. 
  // static T dot(const std::vector<T> v1, const std::vector<T> v2);
  
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
  static T dot(const Vector<T>& v1, const Vector<T>& v2);
  

  // Remove two indexed getters for Vector.
  inline T& getElement(const int row, const int col) = delete;
  T& operator()(const int row, const int col) = delete;

  inline T& getElement(const int row) {
    return elements_[row][0];
  }
  T& operator()(const int row) {
    return this->getElement(row);
  }

};



// TODO implement tensors. 

} // util

} // cpp_nn


#include "src/CPPNeuralNet/Utils/Utils.tpp";

#endif  // CPP_NN_UTIL
