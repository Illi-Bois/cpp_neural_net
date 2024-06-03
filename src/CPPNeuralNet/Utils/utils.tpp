



#include "include/CPPNeuralNet/Utils/utils.h"


namespace cpp_nn{
namespace util{
    
template<typename T>
Matrix<T>::Matrix(int num_rows, int num_cols, T initial_value = T()) 
    : num_rows_(num_rows), num_cols_(num_col), elements_(num_rows, std::vector<T>(num_cols, initial value)){}

template<typename T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> list): num_rows_(list.size()),num_cols_(list.empty() ? 0 : list.begin->size()){
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
}
}

// /**
//   * Constructor with Initial Value
//  */

//  /**
//   * Constructor with Initializer list
//  */

    
