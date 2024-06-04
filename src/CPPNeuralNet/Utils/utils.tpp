namespace cpp_nn{
namespace util{
// /**
//   * Constructor with Initial Value
//  */
template<typename T>
Matrix<T>::Matrix(int num_rows, int num_cols, T initial_value = T()) 
    : num_rows_(num_rows), num_cols_(num_cols), elements_(num_rows, std::vector<T>(num_cols, initial_value)){}


//  /**
//   * Constructor with Initializer list
//  */

template<typename T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> list)
    : num_rows_(list.size()),num_cols_(list.empty() ? 0 : list.begin()->size()){
    elements_.resize(num_rows_);
    int row = 0;
    for(const std::initializer_list<T>& sublist : list){
        if(sublist.size() != num_cols_){
            throw std::invalid_argument("All rows should have same amount of columns");
        }
        elements_[row].assign(sublist.begin(),sublist.end());
        ++row; 
    }
}



// template<typename T>
// const std::vector<T>& Matrix<T>::operator[](int index) const{
//     return elements_[index];
// }


// template<typename T>
// std::vector<T>& Matrix<T>::operator[](int index){
//     return elements_[index];
// }





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



/**
 * Dot product
*/

template<typename T>
T Vector<T>::dot(const Vector<T>& v1, const Vector<T>& v2){
    if(v1.getNumRows() != v2.getNumRows() || v1.getNumCols() != 1 || v2.getNumCols() != 1){
        throw std::invalid_argument("Vector dimensions not compatible for Dot Product");
    }
    T temp = T();
    for(int i = 0; i < v1.getNumRows();++i){
        temp+= v1(i,0) * v2(i,0);
    }
    return temp;
}



}
}