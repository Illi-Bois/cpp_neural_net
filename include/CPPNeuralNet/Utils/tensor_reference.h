#ifndef CPP_NN_TENSOR_REF
#define CPP_NN_TENSOR_REF

#include "include/CPPNeuralNet/Utils/tensor.h"

#include <vector>
#include <initializer_list>

namespace cpp_nn {
namespace util {

/**
 * TensorReference.
 * A lightweight accessor and iterator over Tensor. 
 * Given a Tensor and order of TensorChunk, ie 2 for Matrix,
 *    Will provide accessors and iterators over the multiarray of Tensors.
 */
template <typename T>
class TensorReference { // ================================================================================
 private:
// Members ------------------------------------------------------
  Tensor<T>::TensorElement* elements_; // ownership is never given
  const int kChunkOrder;   // size of TensorChunk to be iterating
  const int kChunkCapacity; // Capacity of individual Chunks

  std::vector<int> index_; // order = order(Tensor) - kChunkOrder
  int index_address_; // Direct Integer Address on TensorElement's element_ vector
                      // This will allow us to bypass recalculating array-index from Tensor-Index
// End of Members -----------------------------------------------

 public:
// Constructor --------------------------------------------------
/** Tensor-Referencing
 *  Index is set to 0th, or very first matrix. */
  TensorReference(Tensor<T>& tensor, const int chunkOrder);
/** Tensor-Referencing with Index
 *  Index is set as specified. 
 *    Throws error for 
 *      'Order Mismatch'
 *      'Index out of Bounds' */
  TensorReference(Tensor<T>& tensor, const int chunkOrder, std::initializer_list<int> indices);
  /** Tensor-Referencing with Index 
   *  ChunkOrder is implied by index*/
  TensorReference(Tensor<T>& tensor, std::initializer_list<int> indices);
    // TODO is chunkOrder necessary when it can be implied by index's order?
// End of Constructor -------------------------------------------


// Accessors ----------------------------------------------------
/** Getter
 *  Retrieves element at index of current TensorChunk
 */
  T& getElement(std::vector<int> index);
/** Getter Parenthesis Notation */
  inline T& operator()(std::vector<int> index) {return getElement(index);}
// End of Accessors ---------------------------------------------

// Iteration ----------------------------------------------------
/** increments one matrix over. 
 *  Returns 1 for successful incrementation, 0 for failed incrementation.
 * 
 * After failure, is set to 0th index again. Therefore checking terminatin with return flag is crucial.
 */
  int incrementIndex();
// End of Iteration ---------------------------------------------
}; // End of TensorReference ==============================================================================

/**
 * Intermediary object to aid in referencing matrices in a Tensor. 
 * MatrixReference will allow easy access into given Tensor's lowest level matrices
 *    and iterator through 
 */
// TODO: MatrixIterator may be a more suitable name. Consider it.
template<typename T = double>
class MatrixReference { // ================================================================================
 private: 
  // Index of Matrix currently referenced on TensorElement
  Tensor<T>::TensorElement* elements_; // ownership is never given
  std::vector<int> index_; // order = order(TensorElement) - 2
 public:
// Constructor --------------------------------------------------
/** Tensor-Referencing
 *  Index is set to 0th, or very first matrix. */
  MatrixReference(Tensor<T>& tensor);
/** Tensor-Referencing with Index
 *  Index is set as specified. 
 *    Throws error for 
 *      'Order Mismatch'
 *      'Index out of Bounds' */
  MatrixReference(Tensor<T>& tensor, std::initializer_list<int> indices);
// End of Constructor -------------------------------------------

// Accessors ----------------------------------------------------
/** Getter
 *  for [row][col] on the index_th Matrix of referencing Tensor */
  T& getElement(int row, int col);
/** Getter Parenthesis Notation */
  inline T& operator()(int row, int col) {return getElement(row, col);}
// End of Accessors ---------------------------------------------

// Iteration ----------------------------------------------------
/** increments one matrix over. 
 *  Returns 1 for successful incrementation, 0 for failed incrementation.
 * 
 * After failure, is set to 0th index again. Therefore checking terminatin with return flag is crucial.
 */
  int incrementIndex();
// End of Iteration ---------------------------------------------
}; // End of MatrixReference ==============================================================================

} // util
} // cpp_nn

#endif // CPP_NN_TENSOR_REF