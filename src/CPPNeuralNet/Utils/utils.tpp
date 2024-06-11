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
  if (!num_rows_) return; // if no rows, nothing to check

  // Ensure that each row has same number of columns, else throw exception.
  for (auto const& row : list) {
    if (row.size() != num_cols_) throw std::invalid_argument("Matrix Construction: Initization List - All rows must have same number of columns"); 
  }

  // TODO Alternative Idea: simply force all rows to match maximum columns count?
}

// Copy Constrcutor : Implemented in Header
//   Matrix(Matrix& const other) 

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
Matrix<T> Matrix<T>::operator*(const Matrix& other) const{
  if (num_cols_ != other.num_rows_) throw std::invalid_argument("Matrix Multiplication - Dimension Mismatch");
  
  Matrix<T> result(this->num_rows_, other.getNumCols());
  for(int r = 0 ; r < result.num_rows_; ++r) {
    for(int c = 0 ; j < result.num_cols_; ++c) {
      T result_element = T();
      for(int k = 0; k < num_cols_; ++k) {
        result_element += this->getElement(r, k) * other.getElement(k, c);
      }
      result(i,j) = result_element;
    }
  }
  return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
    // + operator is defined by copy then MatAdd
    return Matrix(*this).MatAdd(other);
}
// End of Separate Operators ----------------------------------------------------

// End of Matrix =================================================================================




/**
 * Dot product
*/

template<typename T>
T Vector<T>::dot(const Vector<T>& v1, const Vector<T>& v2) {
    if (v1.getNumRows() != v2.getNumRows()) { // col_check unnecessary, as Vector asserts 1-column.
        throw std::invalid_argument("Vector dimensions not compatible for Dot Product");
    }
    T temp = T();
    const int rows = v1.getNumRows;
    for (int i = 0; i < rows; ++i) {
        temp += v1(i) * v2(i);
    }
    return temp;
}




// Extrenous Operators =========================================================================
// Matrix * Vector  
//   Identical in function as Matrix *, but asserts return is Vector
template<typename T=double>
Vector<T> operator*(Matrix<T> M, Vector<T> v) {
  if (M.getNumCols() != v.getNumRows()) { 
      throw std::invalid_argument("Matrix*Vector Multiplication - Dimension Mismatch");
  }

  return asVector(M * v); 
}

// Given nx1 matrix return Vector object that interpretes Matrix as Vector
template<typename T=double>
Vector<T> asVector(Matrix<T> M) {
  if (M.num_cols_() != 1) { 
      throw std::invalid_argument("asVector - Column Dimension Greater than 1");
  }

  return Vector<T>(M.elements_); // TODO need to check if valid
}
// End of Extrenous Operators ===================================================================


}
}