#ifndef CPP_NN_UTIL
#define CPP_NN_UTIL


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