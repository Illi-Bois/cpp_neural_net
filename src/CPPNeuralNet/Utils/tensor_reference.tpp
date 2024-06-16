#include "include/CPPNeuralNet/Utils/tensor_reference.h"

namespace cpp_nn {
namespace util {
// TensorReference =================================================================
// Constructor ---------------------------------------------------------
/** Tensor-Referencing */
template <typename T>
TensorReference<T>::TensorReference(const Tensor<T>& tensor, const int chunkOrder)
    : elements_(tensor.elements_), 
      kChunkOrder(chunkOrder),
      index_address_(0) {
  if (kChunkOrder < 0) 
    throw std::invalid_argument("TensorReference Constructor- Negative ChunkOrder");
  if (tensor.getOrder() < kChunkOrder) 
    throw std::invalid_argument("TensorReference Constructor- Insufficient Tensor Order for Matrix");
}
/** Tensor-Referencing with Index */
template <typename T>
TensorReference<T>::TensorReference(const Tensor<T>& tensor, const int chunkOrder, const std::vector<int>& indices) 
    : elements_(tensor.elements_), 
      kChunkOrder(chunkOrder), 
      index_address_(0) {
  if (kChunkOrder < 0) 
    throw std::invalid_argument("TensorReference Constructor- Negative ChunkOrder");
  if (tensor.getOrder() < kChunkOrder) 
    throw std::invalid_argument("TensorReference Constructor- Insufficient Tensor Order for TensorChunk");
  if (index_.size() != tensor.getOrder() - kChunkOrder) 
    throw std::invalid_argument("TensorReference Index Constructor- Index Order Mismatch"); 
  
  // Set ChunkCapacity
  kChunkCapacity = 1;
  for (int i = indices.size(); i < tensor.getOrder(); ++i) {
    kChunkCapacity *= tensor.getDimension(i);
  }

  // Compute index_address_ while checking
  int block_size = kChunkCapacity;
  for (int i = index_.size() - 1; i >= 0; --i) {
    if (index_[i] < 0 || index_[i] >= elements_->getDimension(i)) {
      throw std::invalid_argument("TensorReference Index Constructor- Index Out of Bounds"); 
    }

    index_address_ += block_size * index_[i];
    block_size *= tensor.getDimension(i);
  }
}
/** Tensor-Referencing with Index as InitList */
template <typename T>
TensorReference<T>::TensorReference(const Tensor<T>& tensor, const int chunkOrder, const std::initializer_list<int>& indices) 
    : TensorReference<T>(tensor, chunkOrder, std::vector<int>(indices)) {}
/** Tensor-Referencing with Index and ChunkOrder implication */
template <typename T>
TensorReference<T>::TensorReference(const Tensor<T>& tensor, const std::initializer_list<int>& indices) 
    : TensorReference(tensor, tensor.getOrder() - indices.size(), indices) {} // Reuse Contructor
// End of Constructor --------------------------------------------------

// Housekeeping --------------------------------------------------------
/** Chunk Index to Addres */
template<typename T>
int TensorReference<T>::ConvertToAddress(const std::vector<int>& indices) const {
  int chunk_address = 0;
  int block_size = 1;
  for (int i = 0; i < kChunkOrder; ++i) {
    int idx = elements_->getOrder() - 1 - i;
    if (index[i] >= 0 && index[i] < elements_->getDimension(i)) {
      chunk_address += block_size * index[i];
      block_size *= elements_->getDimension(i);
    } else {
      throw std::invalid_argument("TensorReference ElementGetter- Index Out of Bounds"); 
    }
  }
  return chunk_address;
}
// End of Housekeeping -------------------------------------------------

// Accessors -----------------------------------------------------------
/** Getter */
template <typename T>
T& TensorReference<T>::getElement(const std::vector<int>& indicies) {
  // Retrieves element directly from array, advoid recalculating index's address
  return elements_->elements_[index_address_ + ConvertToAddress(indicies)]; 
}
template <typename T>
const T& TensorReference<T>::getElement(const std::vector<int>& indicies) const {
  // Retrieves element directly from array, advoid recalculating index's address
  return elements_->elements_[index_address_ + ConvertToAddress(indicies)]; 
}
// End of Accessors ----------------------------------------------------

// Iteration -----------------------------------------------------------
/** increments one matrix over. 
 *  Returns 1 for successful incrementation, 0 for failed incrementation.
 * 
 * After failure, is set to 0th index again. Therefore checking terminatin with return flag is crucial.
 */
template <typename T>
int TensorReference<T>::incrementIndex() {
  index_address_ += kChunkCapacity; // This provides fast way to increment

  if (index_address_ >= elements_->kCapacity) {
    index_address_ = 0; // reset to 0
    return 0; // index beyond capacity
  }
  return 1;
}
// End of Iteration ----------------------------------------------------
// End of TensorReference ==========================================================

// MatrixReference =================================================================
// Constructor ---------------------------------------------------------
/** Tensor-Referencing */
template<typename T>
MatrixReference<T>::MatrixReference(const Tensor<T>& tensor)
    : TensorReference<T>(tensor, 2),
      kRows(tensor.getDimension(tensor.getOrder() - 2)),
      kCols(tensor.getDimension(tensor.getOrder() - 1)) {}
/** Tensor-Referencing with Index */
template<typename T>
MatrixReference<T>::MatrixReference(const Tensor<T>& tensor, const std::vector<int>& indices) 
    : TensorReference<T>(tensor, 2, indices),
      kRows(tensor.getDimension(tensor.getOrder() - 2)),
      kCols(tensor.getDimension(tensor.getOrder() - 1)) {}
/** Tensor-Referencing with Index as InitList */
template<typename T>
MatrixReference<T>::MatrixReference(const Tensor<T>& tensor, const std::initializer_list<int>& indices) 
    : TensorReference<T>(tensor, 2, indices),
      kRows(tensor.getDimension(tensor.getOrder() - 2)),
      kCols(tensor.getDimension(tensor.getOrder() - 1)) {}
// End of Constructor --------------------------------------------------

// Accessors -----------------------------------------------------------
/** Getter */
template<typename T>
T& MatrixReference<T>::getElement(int row, int col) {
  // Best to bypass forming index-vectors at all
  return elements_->elements_[this->index_address_ + kCols * row + col];
}
template<typename T>
const T& MatrixReference<T>::getElement(int row, int col) const {
  // Best to bypass forming index-vectors at all
  return elements_->elements_[this->index_address_ + kCols * row + col];
}
// End of Accessors ----------------------------------------------------

// Matrix Operations ---------------------------------------------------
/** Multiply Into */
template<typename T>
void MatrixReference<T>::MultiplyInto(const MatrixReference<T>& A, const MatrixReference<T>& B) {
  if (this->kRows != A.kRows ||
      this->kCols != B.kCols ||
      A.kCols != B.kCols) {
    throw std::invalid_argument("MatrixReference Multiplication- Dimension Mismatch"); 
  }

  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kCols; ++c) {
      getElement(r, c) = 0;
      for (int k = 0; k < A.kCols; ++k) {
        getElement(r, c) += A.getElement(r, k) * B.getElement(k, c);
      }
    }
  }
}
// End of Matrix Operations --------------------------------------------
// End of MatrixReference ==========================================================

} // util
} // cpp_nn