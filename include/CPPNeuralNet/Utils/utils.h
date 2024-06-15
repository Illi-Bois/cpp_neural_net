#ifndef CPP_NN_UTIL
#define CPP_NN_UTIL


#include <vector>
#include <initializer_list>
#include <utility>

namespace cpp_nn {
namespace util {

// TODO Move Matrix and Vector objects into their own headers

/**
 * Lowest level math object for NN. Vectors will be treated as special case of matrices.
*/
template<typename T = double>
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
  Matrix(const Matrix& other)
      : num_rows_(other.num_rows), num_cols_(other.num_cols),
        elements_(other.elements) {}
  
  /**
   * From Double Vector
   *   Performs internal dimension check
   */
  Matrix(std::vector<std::vector<T>> elements);
  
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

  /** Matrix Product
   *    Multiplies MatrixLike object onto current Matrix. MatrixLike must inherit from Matrix.
   *    Return type is specified as MatrixLike, which of course can be cast to Matrix.
   *  Returns separate instance of MatrixLike
   */
  template<class MatrixLike>
  MatrixLike operator*(const MatrixLike& other) const;

  /** Matrix Sum
   *    Sums MatrixLike object onto current Matrix. MatrixLike must inherit from Matrix.
   *    Return type is specified as MatrixLike, which of course can be cast to Matrix.
   *  Returns separate instance of MatrixLike
   */
  template<class MatrixLike>
  MatrixLike operator+(const MatrixLike& other) const;
// End of Separate Operators ----------------------------------------------------


// Transpose Function ----------------------------------------------------
  /**
   * Returns a transposed copy of the matrix
   */
  Matrix<T> transpose() const;

/* As designed so far, so called Self-operators and Separate-operators are duals, meaning one can and possibly should be defined in terms of each other. 
   ie. operator+ := return Matrix(*this).MatAdd(other);
       MatMul := return (*this) = std::move((*this) * other);
*/

// Housekeeping ----------------------------------------------------------------
// Checks if each row's column dimensions match.
// throws expcetion
  void checkDimension() const;
// End of Housekeeping ---------------------------------------------------------
};


/**
 * Inputs and outputs will be prepresented by vectors.
 * 
 * 
*/
template<typename T = double>
class Vector : public Matrix<T> {
 public:
// Constructor ------------------------------------------------------------- 
  /**
   * Construct Vector with all elements set to initial_value. 
   *   Set to default T() if none give.
  */
  Vector(int dim, T initial_value = T())
      : Matrix<T>(dim, 1, initial_value) {};
  /**
   * Construct Vector with all elements set to initial_value. 
   * Takes in extreneous column_size to Matrix-like compatibility.
   *    Will throw error if col_num not 1
   *   Set to default T() if none give.
  */
  Vector(int dim, int col_num = 1, T initial_value = T())
      : Matrix<T>(dim, 1, initial_value) {
    if (col_num != 1) throw std::invalid_argument("Vector Legacy Constructor - Col Dimension Mismatch");
  };
  /**
   * Constructor with Initializer list.
   *    Must ensure that dimension always matches. For example, each row must have equal columns.
  */
  Vector(std::initializer_list<T> list);
  /**
   * From Vector
   */
  Vector(std::vector<T> elements);

  /**
   * Copy Constructor. 
   *   Contruct deep copy from given matrix.
  */
  Vector(const Matrix<T>& other) : Matrix<T>(other) {}
// End of Constructor ------------------------------------------------------

// Vector Operators --------------------------------------------------------
  T dot(const Vector<T>& v1, const Vector<T>& v2) const;
// End of Vector Operators --------------------------------------------------------


// Vector Accessors --------------------------------------------------------
  // Remove two indexed getters for Vector.
  inline T& getElement(const int row, const int col) = delete;
  T& operator()(const int row, const int col) = delete;

  inline T& getElement(const int row) {
    return this->elements_[row][0];
  }
  T& operator()(const int row) {
    return this->getElement(row);
  }
// End of Vector Accessors -------------------------------------------------
};

// Extrenous Operators =========================================================================

// End of Extrenous Operators ===================================================================

// // TODO Tensor should be treated as generalization of Matirces. That means, once fully implemented, Matrix inherits from Tensor
// template<typename T = double>
// class Tensor{
//  private:
//   std::vector<int> dimensions_;
//   std::vector<T> elements_;
//  public:
//   //Tensor with initial value constructor
//   Tensor(const std::vector<int>& dims, T initial_value = T());
//   //copy constructor
//   //allows to create new instance by copying existing instance
//   Tensor(const Tensor& other);
//   //assign new value to existing object
//   Tensor& operator=(const Tensor& other);
// };


// Additional Static Operations ==========================================================================
/**
 * Reorder Vector based on Index Mapper
 * Code from https://stackoverflow.com/questions/838384/reorder-vector-using-a-vector-of-indices
 */
template< typename order_iterator, typename value_iterator >
void reorder( order_iterator order_begin, order_iterator order_end, value_iterator v )  {   
  typedef typename std::iterator_traits< value_iterator >::value_type value_t;
  typedef typename std::iterator_traits< order_iterator >::value_type index_t;
  typedef typename std::iterator_traits< order_iterator >::difference_type diff_t;
  
  diff_t remaining = order_end - 1 - order_begin;
  for ( index_t s = index_t(), d; remaining > 0; ++ s ) {
    for ( d = order_begin[s]; d > s; d = order_begin[d] ) ;
    if ( d == s ) {
      --remaining;
      value_t temp = v[s];
      while ( d = order_begin[d], d != s ) {
        std::swap( temp, v[d] );
        --remaining;
      }
      v[s] = temp;
    }
  }
}
// End of Additional Static Operations ===================================================================

} // util
} // cpp_nn


#endif  // CPP_NN_UTIL



/**
 * Some Planning:
 * 
 * 
 * Essential Functionalities:
 * - MatMul
 * - MatSum
 * - Transpose
 * 
 * ** Often, tranpose only appears when multiplcation or other operation is to follow it immediately
 * *** as such, it may be more beneficial/effective if tranpose is done as 'Flag' rather than
 *      Element movement
 * 
 * -> suggests, of course, need for MatMul with Transpose?
 * -> or simply change in accessor is enough
 * 
 * - pros for TransposeMatMul
 * allows quick and light weight transpose transitions
 * for temperory transposing, effectively no memories are moved
 * 
 * - cons
 * no quick in place operation
 * ie, we need to define separate operation like *_tranposed to use in line
 * else we will have to keep track of when and where transpose were called
 * 
 * 
 * This whole issue and more may be averted by forming a more structured Tensor definition.
 */