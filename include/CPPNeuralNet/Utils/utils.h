#ifndef CPP_NN_UTIL
#define CPP_NN_UTIL


#include <vector>
#include <initializer_list>

namespace cpp_nn {
namespace util {

// TODO Move Matrix and Vector objects into their own headers

/**
 * Lowest level math object for NN. Vectors will be treated as special case of matrices.
*/
template<typename T= double>
class Matrix {
 private:
  int num_rows_, num_cols_; 
  std::vector<std::vector<T>> elements_; // elements[row][col]

 public:
 // Constructor ------------------------------------------------------------- 
  /**
   * Construct Matrix with all elements set to initial_value. 
   *   Set to default T() if none give.
  */
  Matrix(int num_rows, int num_cols, T initial_value = T());
  /**
   * Constructor with Initializer list.
   *    Must ensure that dimension always matches. For example, each row must have equal columns.
  */
  Matrix(std::initializer_list<std::initializer_list<T>> list);
  /**
   * Copy Constructor. 
   *   Contruct deep copy from given matrix.
  */
  Matrix(Matrix& const other)
      : num_rows_(other.num_rows), num_cols_(other.num_cols),
        elements_(other.elements) {}
 // End of Constructor -------------------------------------------------------

  
// Move and Copy Operators --------------------------------------------------
  // Copy Operator.
  //   Note: it is essential that copy operator and constrcutor be different. 
  Matrix& operator=(const Matrix& other);
// End of Move and Copy Operators -------------------------------------------


// Getters and Setters --------------------------------------------------------
  inline T& getElement(const int row, const int col) {
    return elements_[row][col];
  }

  T& operator()(const int row, const int col) {
    return getElement(row, col);
  }

  inline int getNumRows() const {
    return num_rows_;
  }
  inline int getNumCols() const {
    return num_cols_;
  }
// End of Getters and Setters --------------------------------------------------


// Self Operators --------------------------------------------------------------
//   Operations are done onto the matrix itself. Meaning matrix from which these operations are called will be updated.

  // Matrix Product
  Matrix<T>& MatMul(const Matrix<T>& B);
  // Element-wise Sum
  Matrix<T>& MatAdd(const Matrix<T>& B);
 
// End of Self Operators --------------------------------------------------------


// Separate Operators -----------------------------------------------------------
//   Operations are done to a copied Matrix entity. This means elements of involved arguements will remain un-updated.

  // Matrix Product
  Matrix operator*(const Matrix& other) const;
  // Matrix Product
  Matrix operator+(const Matrix& other) const;

// End of Separate Operators ----------------------------------------------------

/* As designed so far, so called Self-operators and Separate-operators are duals, meaning one can and possibly should be defined in terms of each other. 
   ie. operator+ := return Matrix(*this).MatAdd(other);
       MatMul := return (*this) = std::move((*this) * other);
*/
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



template<typename T = double>
class Tensor{
  private:
    std::vector<int> dimensions;
    std::vector<T> elements;
  public:
    //Tensor with initial value constructor
    Tensor(const std::vector<int>& dims, T initial_value = T());
    //copy constructor
    //allows to create new instance by copying existing instance
    Tensor(const Tensor& other);
    //assign new value to existing object
    Tensor& operator=(const Tensor& other);
}

} // util
} // cpp_nn


#include "src/CPPNeuralNet/Utils/Utils.tpp";

#endif  // CPP_NN_UTIL
