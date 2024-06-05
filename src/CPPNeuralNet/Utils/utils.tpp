#include "include/CPPNeuralNet/Utils/utils.h"

namespace cpp_nn {
namespace util {

// Matrix =======================================================================================

// Constrcutors ------------------------------------------------------------------------
// Initial Value Constructor 
template<typename T>
Matrix<T>::Matrix(int num_rows, int num_cols, T initial_value = T()) 
    : num_rows_(num_rows), num_cols_(num_cols), 
      elements_(num_rows, std::vector<T>(num_cols, initial_value)) {}

// Initializer List Constrcutor
template<typename T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> list)
    : num_rows(list.size()), elements_(list) {
  // Ensure that each row has same number of columns, else throw exception.
  if (!num_rows) return; // if no rows, nothing to check

  num_cols_ = elements_[0].size();
  for (auto const& row : list) {
    if (row.size() != num_cols_) throw std::invalid_argument("All rows should have same amount of columns");
  }

  // TODO Alternative Idea: simply force all rows to match maximum columns count?
}

// Copy Constrcutor : Implemented in Header
//   Matrix(Matrix& const other) 

// End of Contsrctors -------------------------------------------------------------------


template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix& other) const{
    if(num_cols_ != other.num_rows_){
        throw std::invalid_argument("Matrix dimensions not compatible for Matrix Multiplication");
    }
    Matrix<T> result(num_rows_,other.getNumCols(),T());
    for(int i = 0 ; i < num_rows_;++i){
        for(int j = 0 ; j < other.getNumCols();++j){
            T temp = T();
            for(int k = 0; k < num_cols_;++k){
                temp+= elements_[i][k]* other(k,j);
            }
            result(i,j)=temp;
        }
    }
    return result;
}

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



}
}