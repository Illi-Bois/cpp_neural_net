/**
 * Tensor, as more felxible method of working with Matrix-like objects.
 * 
 * Tensor :: class containing TensorElement
 *            All other TensorLike are to inherit from this class
 * 
 * TensorElement :: private struct or a 'lighter weight' object that contains contents
 *                    of tensor only.
 *                  All operations are done ONTO TensorElement through its wrapper Tensor class.
 *                  Intention of this is for other 'accessor' or 'reference objects' to simply point to 
 *                    a given TensorElement without needing to reallocate potentially massive data.
 *                  ** CAREFUL: This of cource may lead to ownership issue. ie) incorrect destruction time
 *                    This may be resolved at object initialization by a simple flag of ownership
 * 
 * Referencer :: We may want to use this to reference smallest subtensor of TensorElement
 *                ie) only first columns, or [:][2:] like with numpy
 *              This should be very carefully implemented especailly due to ownership.
 *              We may like to implement 'export' function which returns separate Tensor
 *                that copies over the selected subtensor
 * 
 * Matrix :: May or maynot need to be derived from Tensor
 *            Most operations and maths will be defined in terms of Matrix, but 
 *              would ideally allow tensors to be valid input to these matrices.
 *            Asserts that order is 2
 * 
 * Vector :: Similar to above, may or may not derive from Matrix
 *            Asserts order is 1           
 *            However, when computed with matrix, it is treated as nx1 matrix
 * 
 * 
 * Notes on Operation:
 * Matrix Multiplication is ideally implemted on tensor level
 * - Tensor will perform internal checks on order and decide multicability.
 * - - [2, 3, 4] * [4, 5] is to be understood as 'array of 2 matrices mutlplied each to latter matrix"
 * - -   so result will be [2, 3, 5]
 * - -
 * - - [2, 4] * [4, 4, 5] likewise
 * - -
 * - - [4, 2, 3] * [2, 3, 4] may be understood as computing every combination of matrices
 * - - will result in [4, 2, 2, 4], where the outer [4,2] specifies which combination of matrices are
 * - -   chosen to result in [2,4] matrix
 * - - 
 * - Vectors are naturally of order 1,
 * - - [2, 4] * [4] are understood as 4d vector input to [2x4] matrix
 * - -   in effect [4] is treated then as [4, 1],
 * - -   Such may be called 'internal reshaping'
 * - 
 * - Covectors are then treated as [1, 4]
 * - 
 * - Special flags may be added to multiplication to indicate that certain inputs are to be treated as
 * -   collection of vectors or matrices
 * - - For example, [4, 2] * [4, 2] may be flagged with 'asVectors' to indicate the latter is to be treated as 
 * - -   4-array of 2d vectors
 * - - As such result is to be [4, 4] a 4 array of 4d vector
 * - - This can be explicitly forced behavior by reshaping latter as [4, 2, 1]
 * 
 * As above suggests, we mainly focus on matrix mult only 
 *   with tensor serving as container for such matrices. 
 * Perhaps inplication of this is that Matrices should remain lowest object.?
 * 
 * TODO more multiplcation behaviours should be consdiered. 
 *   else throw dimension error?
 * So far only up to 3-orders are allowed in such.
 * 
 * 
 * On Transpose:
 * Transpose is ideally done in flags as to not move around data constantly.
 * This is easily said for Matrices, but for Tensors of higher order may be more difficult to maintain.
 * - In fact transposing in higher orders need clear defining rn.
 * - - Transpose(i, j) where i and j are index of order or (axes)
 * - -   for example, if we transpose T[4, 5, 2] at axes (0, 2) we want T[i][j][k] to access T[k][j][i] instead
 * - -   transposing will reshape it to [2, 5, 4]
 * Maybe we can implement this with transpose Mapper of [0, 1, 2] changing to [2, 0, 1] where we just do get[Mapper[0]] and so on
 */

#ifndef CPP_NN_TENSOR
#define CPP_NN_TENSOR

#include <vector>
#include <initializer_list>

namespace cpp_nn {
namespace util {

// Forward Declarations -------------------------------
template <typename>
class TensorReference;
// End of Forward Declarations ------------------------

template<typename T = double>
class Tensor { // =========================================================================================
 private:
  /**
   * The contents of elements are stored in TensorElement struct. 
   * This allows easy lightweight multi-accessors. 
   */
  class TensorElement { // =================================================================
   private:
    std::vector<int> dimensions_;
    std::vector<T> elements_;
    int kCapacity; // Total Number of elements in Tensor, = Product of Dimensions
    std::vector<int> transpose_map_; // Map maintaining tranpose mapping. 
                                      // tm_[i] will give which stored-axes corresponds to ith order's dimension
                                      /**
                                       * ie) 
                                       * [4, 5, 2] :=: {0, 1, 2}
                                       * -> tp(0, 2)
                                       * [2, 5, 4] :=: {2, 1, 0}
                                       * -> tp(1, 0)
                                       * [5, 2, 4] :=: {1, 2, 0}
                                       */
                                      // TLDR:
                                      //  ith dimension is now given by dimension[tanspose_map_[i]]
    inline int order() const {return dimensions_.size()};

    // TODO
    // Export Transpose
    // Function to actually move the data to match transpose.
    // For when multiple transpose is to be done, or when transpose is temperory
    //   first is done only as indices, then only when exported is moved in elements

    // TODO Reshape
   public:

  // TensorElement Constructor ----------------------------------
  /** Dimension Constructor */
    TensorElement(const std::initializer_list<int>& dims, T initial_value = T());
  /** Copy Constructor */
    TensorElement(const TensorElement& other);
  // End of TensorElement Constructor ---------------------------

  // Accessors ----------------------------------------------------
  /** Element Getter
   *  Throws 'Order Mismatch' when number of indicies is incorrect
   *  Throws 'Dimension Mismatch' when index attempted is out of bounds.
   * In Practice, intended to be used with init_list {i,j,...}
   */
    T& getElement(const std::vector<int>& indices);
  /** Parenthesis Getter
   *  Same as Element Getter but with More accessible notation.
   * In Practice, intended to be used with init_list {i,j,...}
   */
    inline T& operator()(const std::vector<int>& indices) {
      return getElement(indices);
    }
  /** Order gettet */
    inline int getOrder() const {return this->order();}
  /** Dimension Getter */
    inline int getDimension(int axis) const {return dimensions_[transpose_map_[i]];}
                                            // As dimension is accessed in transposed order,
                                            // it effectively transposes the entire tensor
  // End of Accessors ---------------------------------------------

  // TensorElement Modifiers --------------------------------------
  /** Tranpose Axes
   *  Tranposes given axes in the Tensor. 
   */
    void Transpose(int axis_one, int axis_two); 
    // TODO: if axes' dimension is 1, maybe no need to tranpose but just move the dimension only in dimensions?
  // End of TensorElement Modifiers -------------------------------

  // friend ===================================
    friend T& TensorReference<T>::getElement(std::vector<int> index);
    friend T& MatrixReference<T>::getElement(int row, int col);
  // end of friend ============================
  }; // End of TensorElement =================================================================

  TensorElement* elements_;
  bool ownership_; // indicates if elements_ are owned by current Tensor
                   // If owned, must delete upon destrcutor
 public:
// Constructors -------------------------------------------------
/** Dimension Contructors */
  Tensor(std::initializer_list<int> dims, T initial_value = T());
/** Dimension Contructors, Vector*/
  Tensor(std::vector<int> dims, T initial_value = T());
/** Copy Constructor */
  Tensor(const Tensor& other);
/** Move Constrcutor */
  Tensor(Tensor&& other);
// End of Constructors ------------------------------------------

// Accessors ----------------------------------------------------
/** Element Getter
 *  Throws 'Order Mismatch' when number of indicies is incorrect
 *  Throws 'Dimension Out of Bounds' when index attempted is out of bounds.
 * In Practice, intended to be used with init_list {i,j,...}
 */
  T& getElement(const std::vector<int>& indices);
/** Parenthesis Getter
 *  Same as Element Getter but with More accessible notation.
 * In Practice, intended to be used with init_list {i,j,...}
 */
  inline T& operator()(const std::vector<int>& indices) {
    return getElement(indices);
  }
  inline int getOrder() const {
    return elements_->getOrder();
  }
  inline int getDimension(int axis) const {
    return elements_->getDimension(axis);
  }
// End of Accessors ---------------------------------------------

// Operations ---------------------------------------------------
/** Tensor Multiplcation
 * To be understood as matrix multiplications when possible. 
 *
 * Returned Tensor is another instance of the resulting product.
 *  
 * Rules of Multiplcation:
 * If [dims1..., n, m] * [dim2..., m, d] -> [dims1..., dims2..., n, d]
 *  Understood as multiarrays of [n x m] and [m x d] matrices
 *  Produces mutliarray of all combinations of such multiplcations
 * 
 * Other Vector-like behaviours are to be induced by reshaping. 
 *  Vector of n-dimension are Matrices of dimension [n x 1]
 *  For [dim2..., m] to be understood as multiarray of m-dim vectors,
 *    reshape to [dims2..., m, 1]
 *  [dim1..., n, m] * [dim2..., m, 1] -> [dim1..., dim2..., n, 1]
 *    -reshape-> [dim1..., dim2..., n]
 * 
 * Likewise covectors are [1 x n] matrices. 
 * Dot Product are implemented by 
 *  [dim1..., 1, n] * [dim2..., n, 1] -> [dim1..., dim2..., 1, 1] 
 *    -reshape-> [dim1.., dim2.., 1] 
 * Outter Product are implemented by
 *  [dim1..., n, 1] * [dim2..., 1, m] -> [dim1..., dim2..., n, m] 
 */
  Tensor operator*(const Tensor& other) const;

/** Tensor Summation
 * 
 * Returned Tensor is another instance of the resulting Sum.
 * 
 * TODO: see if other implementation is more valid
 * Rules of Summation:
 * Order of summation does matter. 
 * The latter summand defines shape of block-tensor to be summed
 *  When [dims..., blockDim...] + [blockDim...] -> [dims..., blockDim...]
 * Therefore, if current does not match other's shape in its inner-shape, error is thrown.
 * 
 */
  Tensor operator+(const Tensor& other) const;
// End of Operations --------------------------------------------

/**
 * TODO IDEAS TO IMPLEMENT
 * Broadcasting : used for operations like adding bias to activations and applying layer weights to input tensors
 * Concat and Splitting : Not needed now just yet, used in CNN so maybe soon
 * Elementwise Operations (+,-,/) 
 * Tensor Reduction Operations (sum, mean, max, min) : Loss function and Pooling Layers
 * Transpose
 */

/** Broadcasting 
 * Pads the smaller tensor so both shapes are same, allowing element wise operation
 * Rule : Pad the smaller tensor with 1s on the left until they have the same length
 * This Function doesn't do Broadcasting; Instead it returns a vector of the size of Broadcasted Tensor
 */
  std::vector<int> broadcast(const Tensor<T>& other) const;

// Element Wise Operations -----------------------------------------------





//End of Element Wise Operations --------------------------------------------
  

// friends =======================
  friend class TensorReference<T>;
// end of friends :( =============
}; // End of Tensor =======================================================================================

} // util
} // cpp_nn
#endif // CPP_NN_TENSOR