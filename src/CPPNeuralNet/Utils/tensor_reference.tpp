#include "include/CPPNeuralNet/Utils/tensor_reference.h"

namespace cpp_nn {
namespace util {

// TensorReference =================================================================
// Constructor ---------------------------------------------------------
/** Tensor-Referencing */
template <typename T>
TensorReference<T>::TensorReference(Tensor<T>& tensor, const int chunkOrder)
    : elements_(tensor.elements_), kChunkOrder(chunkOrder) {
  if (kChunkOrder <= 0) 
    throw std::invalid_argument("TensorReference Constructor- Non-Positive ChunkOrder");
  if (tensor.getOrder() < kChunkOrder) 
    throw std::invalid_argument("TensorReference Constructor- Insufficient Tensor Order for Matrix");
  // Initial Index at {0,...,0}
  index_.reshape(tensor.getOrder() - kChunkOrder, 0);
}
/** Tensor-Referencing with Index */
template <typename T>
TensorReference<T>::TensorReference(Tensor<T>& tensor, const int chunkOrder, std::initializer_list<int> indices) 
    : elements_(tensor.elements_), kChunkOrder(chunkOrder), 
      index_(indices) {
  if (kChunkOrder <= 0) 
    throw std::invalid_argument("TensorReference Constructor- Non-Positive ChunkOrder");
  if (tensor.getOrder() < kChunkOrder) 
    throw std::invalid_argument("TensorReference Constructor- Insufficient Tensor Order for TensorChunk");
  if (index_.size() != elements_->order() - kChunkOrder) 
    throw std::invalid_argument("TensorReference Index Constructor- Index Order Mismatch"); 
  
  for (int i = 0; i < index_.size(); ++i) {
    if (index_[i] < 0 || index_[i] >= elements_->dimensions[i])
      throw std::invalid_argument("TensorReference Index Constructor- Index Out of Bounds"); 
  }
}
/** Tensor-Referencing with Index and ChunkOrder implication */
template <typename T>
TensorReference<T>::TensorReference(Tensor<T>& tensor, std::initializer_list<int> indices) 
    : elements_(tensor.elements_), kChunkOrder(tensor.getOrder() - inindices.size()), 
      index_(indices) {
  if (kChunkOrder <= 0) 
    throw std::invalid_argument("TensorReference Constructor- Non-Positive Implied ChunkOrder");
    
  for (int i = 0; i < index_.size(); ++i) {
    if (index_[i] < 0 || index_[i] >= elements_->dimensions[i])
      throw std::invalid_argument("TensorReference Index Constructor- Index Out of Bounds"); 
  }
}
// End of Constructor --------------------------------------------------
// End of TensorReference ==========================================================





// MatrixReference =================================================================
// Constructor ---------------------------------------------------------
/** Tensor-Referencing */
template<typename T>
MatrixReference<T>::MatrixReference(Tensor<T>& tensor)
    : elements_(tensor.elements_), index_(std::vector<int>(tensor.getOrder() >= 2 ? tensor.getOrder() - 2 : 0, 0)) {
  if (tensor.getOrder() < 2) 
    throw std::invalid_argument("MatrixReference Constructor- Insufficient Tensor Order for Matrix");
}
/** Tensor-Referencing with Index */
template<typename T>
MatrixReference<T>::MatrixReference(Tensor<T>& tensor, std::initializer_list<int> indices) 
    : elements_(tensor.elements_), index_(indices) {
  if (tensor.getOrder() < 2) 
    throw std::invalid_argument("MatrixReference Constructor- Insufficient Tensor Order for Matrix");

  if (index_.size() != elements_->order() - 2) 
    throw std::invalid_argument("MatrixReference Index Constructor- Index Order Mismatch"); 
  
  for (int i = 0; i < index_.size(); ++i) {
    if (index_[i] < 0 || index_[i] >= elements_->dimensions[i])
      throw std::invalid_argument("MatrixReference Index Constructor- Index Out of Bounds"); 
  }
}
// End of Constructor --------------------------------------------------

// Accessors -----------------------------------------------------------
template<typename T>
T& MatrixReference<T>::getElement(int row, int col) {
  index_.push_back(row);
  index_.push_back(col);
  T& res = elements_->getElement(index_);
  index_.pop_back();
  index_.pop_back();

  return res;
}
// End of Accessors ----------------------------------------------------

// Iteration -----------------------------------------------------------
template<typename T>
int MatrixReference<T>::incrementIndex() {
  for (int order = index_.size() - 1; order >= 0; --order) {
    if (++index_[order] >= elements_->dimensions[order]) {
      index_[order] = 0;
      continue;
    } else {
      // incrementation successful
      return 1;
    }
  }
  // no more index to increment
  return 0;
}
// End of Iteration ----------------------------------------------------
// End of MatrixReference ==========================================================

} // util
} // cpp_nn