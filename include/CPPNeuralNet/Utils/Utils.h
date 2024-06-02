#ifndef CPP_NN_UTIL
#define CPP_NN_UTIL


#include <vector>

namespace cpp_nn {

namespace util {

/**
 * Lowest level math object for NN. Vectors will be treated as special case of matrices.
*/
class Matrix {

  // TODO define how matrix is stored
  //    vec<vec<>>
  //    Should it be a template? double
  // TODO make matrix multiplication
 private:
  int num_rows_, num_cols_; 
  std::vector<std::vector<double>> elements_; // elements[row][col]


 public:
  /**
   * Returns another instance of matrix with this * other; 
  */
  Matrix operator*(Matrix& const other);
};

/**
 * Inputs and outputs will be prepresented by vectors.
 * 
 * 
*/
class Vector : public Matrix {

};


// TODO implement tensors. 

} // util

} // cpp_nn

#endif  // CPP_NN_UTIL