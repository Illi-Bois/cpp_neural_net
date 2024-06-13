#include "include/CPPNeuralNet/Utils/utils.h"

namespace cpp_nn {
namespace util {

// Matrix =======================================================================================
// Constrcutors ------------------------------------------------------------------------
// Initial Value Constructor 
template<typename T>
Matrix<T>::Matrix(int num_rows, int num_cols, T initial_value = T()) 
    : num_rows_(num_rows), num_cols_(num_cols), 
      elements_(std::vector<std::vector<T>>(num_rows, std::vector<T>(num_cols, initial_value))) {}

// Initializer List Constrcutor
template<typename T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> list)
    : num_rows_(list.size()), 
      num_cols_(num_rows_ ? list[0].size() : 0), 
      elements_(list) {
  checkDimansion();
}

// Copy Constrcutor : Implemented in Header
//   Matrix(Matrix& const other) 

template<typename T>
Matrix<T>::Matrix(std::vector<std::vector<T>> elements)
    : num_rows_(elements.size()), 
      num_cols_(num_rows_ ? elements[0].size() : 0), 
      elements_(elements) {
  checkDimansion();
}
// End of Contsrctors -------------------------------------------------------------------


// Move and Copy Operators --------------------------------------------------
// Copy Operator.
template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other) {
  this->num_rows_ = other.num_rows_;
  this->num_cols_ = other.num_cols_;
  this->elements_ = other.elements_; // reply on std::vector's own deep copy
  return *this;
}
// End of Move and Copy Operators -------------------------------------------


// Self Operators --------------------------------------------------------------
template<typename T>
Matrix<T>& Matrix<T>::MatMul(const Matrix<T>& B) {
  // MatMul defined through *
  *this = std::move((*this) * other); // Note that as move operator is not yet explicitly defined, copy operator is called. 
  return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::MatAdd(const Matrix<T>& B) {
  if (this->num_rows_ != B.num_rows_ || this->num_cols_ != B.num_cols_) throw std::invalid_argument("Matrix Addition - Dimension Mismatch");

  for (int r = 0; r < this->num_rows_; ++r) {
    for (int c = 0; c < this->num_cols_; ++c) {
      this->getElement(r, c) += B(r, c); // using (r, c) indexing operator
    }
  }
  return *this;
}
// End of Self Operators -------------------------------------------------------

// Separate Operators -----------------------------------------------------------
template<typename T>
template<class MatrixLike>
MatrixLike Matrix<T>::operator*(const MatrixLike& other) const{
  static_assert(std::is_base_of<Matrix, MatrixLike>::value, "Inheritence Error: not a Matrix-like");
  
  if (num_cols_ != other.num_rows_) throw std::invalid_argument("Matrix Multiplication - Dimension Mismatch");
  
  MatrixLike result(this->num_rows_, other.num_cols_); // TODO will need to force all Matrix-derivitves to have this constructor
  // Task of contrcutor is only to avail the specified dimensions

  // Will force Matrix behaviour by referencing MatrixLike as Matrix
  Matrix& result_as_matrix = result;

  for(int r = 0 ; r < result_as_matrix.num_rows_; ++r) {
    for(int c = 0 ; j < result_as_matrix.num_cols_; ++c) {
      T result_element = T();
      for(int k = 0; k < num_cols_; ++k) {
        result_element += this->getElement(r, k) * other.getElement(k, c);
      }
      result_as_matrix(i,j) = result_element;
    }
  }

  // Return MatrixLike
  return result;
}

template<typename T>
template<class MatrixLike>
MatrixLike Matrix<T>::operator+(const MatrixLike& other) const {
  static_assert(std::is_base_of<Matrix, MatrixLike>::value, "Inheritence Error: not a Matrix-like");

  // + operator is defined by copy then MatAdd
  // Since A+B = B+A, to ensure type consistency, will swap the orders.

  // TODO ALL MATRIXLIKE MUST HAVE COPY CONST
  MatrixLike result(other);
  Matrix& result_as_matrix = result;
  result_as_matrix.MatAdd(*this); // other += this

  return result;
}
// End of Separate Operators ----------------------------------------------------

//Transpose ----------------------------------------------------
template<typename T>
Matrix<T> Matrix<T>::transpose() const {
  Matrix<T> transposed_matrix(num_cols_, num_rows_);
  for (int i = 0; i < num_rows_; ++i) {
    for (int j = 0; j < num_cols_; ++j) {
      transposed_matrix.elements_[j][i] = elements_[i][j];
    }
  }
  return transposed_matrix;
}
//End of Transpose ----------------------------------------------------

// Housekeeping ----------------------------------------------------------------
template<typename T>
void Matrix<T>::checkDimension() const { 
  if (!num_rows_) return; // if no rows, nothing to check

  // Ensure that each row has same number of columns, else throw exception.
  for (auto const& row : elements_) {
    if (row.size() != num_cols_) throw std::invalid_argument("checkDimension - All rows must have same number of columns"); 
  }

  // TODO Alternative Idea: simply force all rows to match maximum columns count?
}
// End of Housekeeping ---------------------------------------------------------
// End of Matrix =================================================================================




// Vector ========================================================================================
// Constructors ----------------------------------------------------------------
/**
 * Construct Vector with all elements set to initial_value. 
 *   Set to default T() if none give.
*/
// Vector(int dim, T initial_value = T())
//  Done in Header

/**
 * Construct Vector with all elements set to initial_value. 
 * Takes in extreneous column_size to Matrix-like compatibility.
 *    Will throw error if col_num not 1
 *   Set to default T() if none give.
*/
// Vector(int dim, int col_num = 1, T initial_value = T());
//   Done in Header

/**
 * Constructor with Initializer list.
 *    Must ensure that dimension always matches. For example, each row must have equal columns.
*/
template<typename T>
Vector<T>::Vector(std::initializer_list<T> list) 
    : Vector(list.size()) { // Call default copnstrucot first
  
  int r = 0;
  for (const T& element : list) {
    getElement(r++) = element;
  }
  // TODO find a more element way to use init_list
}

/**
 * From Vector
 */
template<typename T>
Vector<T>::Vector(std::vector<T> elements)
    : Vector(elements.size()) {
  int r = 0;
  for (const T& element : elements) {
    getElement(r++) = element;
  }
  // TODO same as init_list
}


/**
 * Copy Constructor. 
 *   Contruct deep copy from given matrix.
*/
// Vector<T>::Vector(Matrix& const other)
//    Done in Header
// End of Constructors ---------------------------------------------------------


// Vector Operators --------------------------------------------------------
template<typename T>
T Vector<T>::dot(const Vector<T>& v1, const Vector<T>& v2) const {
  if (v1.getNumRows() != v2.getNumRows()) { // col_check unnecessary, as Vector asserts 1-column.
    throw std::invalid_argument("Vector Dot - Dimension mismacth");
  }

  T temp = T();
  const int rows = v1.num_rows_;
  for (int i = 0; i < rows; ++i) {
    temp += v1(i) * v2(i);
  }
  return temp;
}


// End of Vector Operators -------------------------------------------------

// End of Vector =================================================================================




// Extrenous Operators =========================================================================

// End of Extrenous Operators ===================================================================


}
}