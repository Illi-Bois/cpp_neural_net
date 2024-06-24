#ifndef CPP_NN_TENSOR_REF
#define CPP_NN_TENSOR_REF

#include "include/CPPNeuralNet/Utils/tensor.h"

#include <vector>
#include <initializer_list>
#include <stdexcept>

namespace cpp_nn {
namespace util {
/**
 * TensorReference.
 * A lightweight accessor and iterator over Tensor. 
 * Given a Tensor and order of TensorChunk, ie 2 for Matrix,
 *    Will provide accessors and iterators over the multiarray of Tensors.
 * 
 * Chunk can be of order 0, in which case ech individual element is iterated
 */
template <typename T = double>
class TensorReference { // ================================================================================
 private:
// Members ------------------------------------------------------
  Tensor<T>::TensorElement* elements_; // ownership is never given
  const int kChunkOrder;   // size of TensorChunk to be iterating
  const int kChunkCapacity; // Capacity of individual Chunks

  int index_address_; // Direct Integer Address on TensorElement's element_ vector
                      // This will allow us to bypass recalculating array-index from Tensor-Index
// End of Members -----------------------------------------------

// Housekeeping -------------------------------------------------
/** Chunk Index to Address
 * This is index within the current chunk block,
 *  Must be summed with index_address_ for absolute address */
  int ConvertToAddress(const std::vector<int>& indices) const;
// End of Housekeeping ------------------------------------------
 public:
// Constructor --------------------------------------------------
/** Tensor-Referencing
 *  Index is set to 0th, or very first matrix. */
  TensorReference(const Tensor<T>& tensor, const int chunkOrder);
/** Tensor-Referencing with Index
 *  Index is set as specified. 
 *    Throws error for 
 *      'Order Mismatch'
 *      'Index out of Bounds' */
  TensorReference(const Tensor<T>& tensor, const int chunkOrder, const std::vector<int>& indices);
/** Tensor-Referencing with Index as InitList */
  TensorReference(const Tensor<T>& tensor, const int chunkOrder, const std::initializer_list<int>& indices);
  /** Tensor-Referencing with Index 
   *  ChunkOrder is implied by index*/
  TensorReference(const Tensor<T>& tensor, const std::initializer_list<int>& indices);
    // TODO is chunkOrder necessary when it can be implied by index's order?
// End of Constructor -------------------------------------------

// Accessors ----------------------------------------------------
/** Getter
 *  Retrieves element at index of current TensorChunk
 */
  T& getElement(const std::vector<int>& indices);
  const T& getElement(const std::vector<int>& indices) const ;
/** Getter Parenthesis Notation */
  inline T& operator()(const std::vector<int>& indices) {return getElement(indices);}
  inline const T& operator()(const std::vector<int>& indices) const {return getElement(indices);}
// End of Accessors ---------------------------------------------

// Iteration ----------------------------------------------------
/** increments one Tensor over. 
 *  Returns 1 for successful incrementation, 0 for failed incrementation.
 * 
 * After failure, is set to 0th index again. Therefore checking terminatin with return flag is crucial.
 */
  virtual int incrementIndex();
// End of Iteration ---------------------------------------------
}; // End of TensorReference ==============================================================================


/**
 * Intermediary object to aid in referencing matrices in a Tensor. 
 * MatrixReference will allow easy access into given Tensor's lowest level matrices
 *    and iterator through 
 */
template<typename T = double>
class MatrixReference : public TensorReference<T> { // ================================================================================
 private: 
  const int kRows;
  const int kCols;
 public:
// Constructor --------------------------------------------------
/** Tensor-Referencing
 *  Index is set to 0th, or very first matrix. */
  MatrixReference(const Tensor<T>& tensor);
/** Tensor-Referencing with Index
 *  Index is set as specified. 
 *    Throws error for 
 *      'Order Mismatch'
 *      'Index out of Bounds' */
  MatrixReference(const Tensor<T>& tensor, const std::vector<int>& indices);
/** Tensor-Referencing with Index as InitList */
  MatrixReference(const Tensor<T>& tensor, const std::initializer_list<int>& indices);
// End of Constructor -------------------------------------------

// Accessors ----------------------------------------------------
/** Getter
 *  for [row][col] on the index_th Matrix of referencing Tensor */
  T& getElement(int row, int col);
  const T& getElement(int row, int col) const;
/** Getter Parenthesis Notation */
  inline T& operator()(int row, int col) {return getElement(row, col);}
  inline const T& operator()(int row, int col) const {return getElement(row, col);}
// End of Accessors ---------------------------------------------

// Matrix Operations --------------------------------------------
/** Multiply Into
 * Given MatrixReferences A,B, set the current chunk as A*B, where A,B point to their respective chunks
 * Throws dimension check errors as necessary. 
 */
  void MultiplyInto(const MatrixReference<T>& A, const MatrixReference<T>& B);
// End of Matrix Operations -------------------------------------
}; // End of MatrixReference ==============================================================================

} // util
} // cpp_nn

#include "../src/CPPNeuralNet/Utils/tensor_reference.tpp"

#endif // CPP_NN_TENSOR_REF
