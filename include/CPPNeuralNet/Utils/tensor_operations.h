#ifndef CPP_NN_R_TENSOR_OPERATIONS
#define CPP_NN_R_TENSOR_OPERATIONS

#include "CPPNeuralNet/Utils/utils.h"
#include "CPPNeuralNet/Utils/tensor_like.h"

#include <numeric> // to fill from 0 to n


namespace cpp_nn {
namespace util {

// Forward Declaration ======================================
namespace {
template<typename T, typename HeldOperation>
class TransposeOperation;
template<typename T, typename HeldOperation>
class MultiTransposeOperation;
template<typename T, typename HeldOperation1, typename HeldOperation2>
class SummationOperation;
template<typename T, typename HeldOperation1, typename HeldOperation2>
class MultiplicationOperation;
template<typename T, typename HeldOperation>
class ReshapeOperation;
template<typename T, typename HeldOperation>
class PaddingOperation;
template<typename T, typename HeldOperation>
class BroadcastOperation;
} // unnamed namespace

template<typename T>
class Tensor;
// End of Forward Declaration ===============================

namespace {
/* Operation Facilitators ---------------------------------------------------------- */
/*    
  Intermediate classes to be passed to reduce movement of tensor data
  These classes will appear as return type of operations such as Tranpose and summation,
    and only upon assignment to a Tensor will data be moved.
  There are exception to these rules, such as with multiplcation which 
    may benefit from operation itself allocating space for computation and storage,
    but it too will follow the design of the other operations.

 *  ALL OPERATION VALIDITY ARE CHECKED UPON CONSTRUCT TODO:
 */

/**
 *  holds transpose operation.
 *  Transposing again on TransposeOperation will return 
 *    MultiTranspose which collapses each operation into single 
 *    MultiTranspose.
 *  Therefore in practice, transposing together is prefered.
 */
template<typename T, typename HeldOperation>
class TransposeOperation : public TensorLike<T, TransposeOperation<T, HeldOperation>> {
  typedef TransposeOperation<T, HeldOperation> Self;
  typedef TensorLike<T, Self> Parent;

// Members ---------------------------------------------
  const HeldOperation& tensor_;
  const int axis_1_;
  const int axis_2_;

  std::vector<int> dimension_;    // dimension post-tranposing
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  TransposeOperation(const TensorLike<T, HeldOperation>& tensor_like, 
                     const int axis_1, const int axis_2)
      : tensor_(tensor_like.getRef()), 
        // axis are stored as positive int, though it can be given as negative
        axis_1_(SumIfNegative(axis_1, tensor_.getOrder())), 
        axis_2_(SumIfNegative(axis_2, tensor_.getOrder())), 
        dimension_(tensor_like.getShape()) {
    std::swap(dimension_[axis_1_], dimension_[axis_2_]);
  }
// End of Constructor ----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    return dimension_;
  }
  inline int getDimension(int axis) const noexcept {
    return dimension_[SumIfNegative(axis, getOrder())];
  }
  inline size_t getCapacity() const noexcept {
    return tensor_.getCapacity();
  }
  inline int getOrder() const noexcept {
    return dimension_.size();
  }  
  // indices for tranpose is passed as value, not reference,
  //   so that tranposed axis may be swapped. 
  const T getElement(std::vector<int> indices) const {
    std::swap(indices[axis_1_], indices[axis_2_]);
    return tensor_.getElement(indices);
  }
  // Transpose on TranspeOperation becomes MultiTranspose. 
  //    Further additional transposes are handled from that class.
  inline MultiTransposeOperation<T, HeldOperation>
         Transpose(int axis_1 = -1, int axis_2 = -2) const {
    return {*this, axis_1, axis_2};
  }
// End of Tensor-Behaviours ----------------------------

// Iterator --------------------------------------------
  typedef typename HeldOperation::ConstIterator HeldIterator;

  class ConstIterator : public Parent::DefaultConstIterator {
   private:
    typedef typename Parent::DefaultConstIterator Parent;
  // Members ---------------------------------------------
    HeldIterator it_;
    size_t curr_address_;

    std::vector<int> old_chunk_sizes_;
    size_t capacity_;

    const std::vector<int>& held_shape_;
  // End of Members --------------------------------------

  // Housekeeping ----------------------------------------
  /** Updates current_address to given address */
    void IncrementAddressTo(size_t address_to) {
      size_t diff;
      if (curr_address_ > address_to) {
        diff = curr_address_ - address_to;
        it_ -= diff;;
      } else {
        diff = address_to - curr_address_;
        it_ += diff;;
      }
      curr_address_ = address_to;
    }
  // End of Housekeeping ---------------------------------

   public:
    ConstIterator(const Self* transpose_ptr, 
                  std::vector<int> idx, 
                  bool is_end,
                  HeldIterator it,
                  size_t address)
        : Parent::DefaultConstIterator(transpose_ptr, 
                                       idx, 
                                       is_end),
          it_(it),
          curr_address_(address),
          old_chunk_sizes_(transpose_ptr->getOrder(), 1),
          capacity_(1),
          held_shape_(transpose_ptr->tensor_.getShape()) {
      // compute chunk sizes from old, then swap
      ComputeCapacityAndChunkSizes(transpose_ptr->tensor_.getShape(),
                                   old_chunk_sizes_,
                                   capacity_);
      std::swap(old_chunk_sizes_[transpose_ptr->axis_1_], 
                old_chunk_sizes_[transpose_ptr->axis_2_]);
    }

    T operator*() const override {
      return *it_;
    }

    ConstIterator& operator+=(int increment) override {
      // Early exits
      if (increment < 0) {
        return operator+=(-increment);
      } else if (increment == 0) {
        return *this;
      }
      // Rely on default incrementation
      Parent::operator+=(increment);

      // when at end, address cannot be deduced from idx alone
      size_t address;
      if (Parent::at_end_) {
        address = Parent::tensor_like_->getCapacity();
      } else {
        address = IndicesToAddress(held_shape_, 
                                   old_chunk_sizes_,
                                   Parent::current_indices_);
      }
      IncrementAddressTo(address);
      return *this;
    }
    ConstIterator& operator-=(int decrement) override {
      // Early exits
      if (decrement < 0) {
        return operator+=(-decrement);
      } else if (decrement == 0) {
        return *this;
      }
      // Rely on default decrementation
      Parent::operator-=(decrement);

      // as -= always sets to expected idx, can call address immediately 
      IncrementAddressTo(IndicesToAddress(held_shape_, 
                                          old_chunk_sizes_,
                                          Parent::current_indices_));
      return *this;
    }
  }; // End of ConstIterator

  ConstIterator begin() const {
    return {this, std::vector<int>(getOrder(), 0), false, tensor_.begin(), 0};
  }
  ConstIterator end() const {
    return {this, std::vector<int>(getOrder(), 0), true, tensor_.end(), getCapacity()};
  }
// End of Iterator -------------------------------------

// friends ---------------------------------------------
  friend class MultiTransposeOperation<T, HeldOperation>;
// end of friends --------------------------------------
}; // End of TransposeOperation

/** 
 *  handles multiple tranposes in single operation holder. 
 *  Can be treated as identical to TransposeOperation in practice.
 *    Only TransposeOperation can generate it.
 * 
 *  Must be non-const to take advantage of in-place modifier.
 *  Else returns capsulation of TransposeOperation again.
 */
template<typename T, typename HeldOperation>
class MultiTransposeOperation : public TensorLike<T, MultiTransposeOperation<T, HeldOperation>> {
  typedef MultiTransposeOperation<T, HeldOperation> Self;
  typedef TensorLike<T, Self> Parent;

// Members ---------------------------------------------
  const HeldOperation& tensor_like_;

  std::vector<int> dimension_;    // dimension post-tranposing
  // tranposed_index -> untranpose_map_ -> heldOperation_index
  std::vector<int> untranpose_map_;             // use to convert tranposed_indices to indices for tensor_like_
  // heldOperation_index -> untranpose_map_ -> tranposed_index
  std::vector<int> tranpose_map_;               // use to convert tensor_like_'s indicies to tranposed indices
// End of Members --------------------------------------
 
 public:
// Constructor -----------------------------------------
  // MultiTranspose is contructed singularly from TransposeOperation
  MultiTransposeOperation(const TransposeOperation<T, HeldOperation>& tranpose_operation, 
                          int axis_1, 
                          int axis_2)
    : tensor_like_(   tranpose_operation.tensor_),
      dimension_(     tensor_like_.getShape()),      // Get Untramnsposed dimension
      untranpose_map_(tensor_like_.getOrder()),
      tranpose_map_(  tensor_like_.getOrder()) {
    // fill from 0 to order
    std::iota(untranpose_map_.begin(), untranpose_map_.end(), 0);
    std::iota(tranpose_map_.begin(),   tranpose_map_.end(),   0);
    // Transpose in set order. 
    this->Transpose(tranpose_operation.axis_1_, tranpose_operation.axis_2_);
    this->Transpose(axis_1, axis_2);
  }
// End of Constructor ----------------------------------
  
// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    return dimension_;
  }
  inline int getDimension(int axis) const {
    return dimension_[SumIfNegative(axis, getOrder())];
  }
  inline size_t getCapacity() const noexcept {
    return tensor_like_.getCapacity();
  }
  inline int getOrder() const noexcept {
    return dimension_.size();
  }  
  const T getElement(const std::vector<int>& indices) const {
    // convert tranpose_indices to untransposed_indices by passing axis through map
    std::vector<int> untranposed_indices(getOrder());
    for (int axis = 0; axis < getOrder(); ++axis) {
      // TODO: both statements below are equivalent. Keep them both for now
      //    as either might be more beneficial for futher iterator implementation
      untranposed_indices[axis] = indices[tranpose_map_[axis]];
      // untranposed_indices[untranpose_map_[axis]] = indices[axis];
    }
    return tensor_like_.getElement(untranposed_indices);
  }
  // Constant default transpose
  inline const TransposeOperation<T, MultiTransposeOperation<T, HeldOperation>> 
               Transpose(int axis1 = -2, 
                         int axis2 = -1) const {
    return {*this, axis1, axis2};
  }
  //  Tranpose on MultiTranspose is in-place operation, as
  //    each transpose is to collapse onto this single Operation
  //  Therefore returns reference to self
  MultiTransposeOperation<T, HeldOperation>& 
         Transpose(int axis_1 = -1, 
                   int axis_2 = -2) {
    // normalized to positive
    axis_1 = SumIfNegative(axis_1, getOrder());
    axis_2 = SumIfNegative(axis_2, getOrder());

    std::swap(dimension_[axis_1], dimension_[axis_2]);
    // TODO: As mentioned in getElement, not both maps may be needed
    int& effective_axis_1 = untranpose_map_[axis_1];
    int& effective_axis_2 = untranpose_map_[axis_2];
    std::swap(effective_axis_1, 
              effective_axis_2);
    std::swap(tranpose_map_[effective_axis_1], 
              tranpose_map_[effective_axis_2]);
    return *this;
  }
// End of Tensor-Behaviours ----------------------------

// Iterator --------------------------------------------
  typedef typename HeldOperation::ConstIterator HeldIterator;

  class ConstIterator : public Parent::DefaultConstIterator {
   private:
    typedef typename Parent::DefaultConstIterator Parent;
  // Members ---------------------------------------------
    HeldIterator it_;
    size_t curr_address_;

    std::vector<int> old_chunk_sizes_;
    size_t capacity_;

    const std::vector<int>& held_shape_;
  // End of Members --------------------------------------

  // Housekeeping ----------------------------------------
  /** Updates current_address to given address */
    void IncrementAddressTo(size_t address_to) {
      size_t diff;
      if (curr_address_ > address_to) {
        diff = curr_address_ - address_to;
        it_ -= diff;;
      } else {
        diff = address_to - curr_address_;
        it_ += diff;;
      }
      curr_address_ = address_to;
    }
  // End of Housekeeping ---------------------------------

   public:
    ConstIterator(const Self* transpose_ptr, 
                  std::vector<int> idx, 
                  bool is_end,
                  HeldIterator it,
                  size_t address)
        : Parent::DefaultConstIterator(transpose_ptr, 
                                       idx, 
                                       is_end),
          it_(it),
          curr_address_(address),
          old_chunk_sizes_(transpose_ptr->getOrder(), 1),
          capacity_(1),
          held_shape_(transpose_ptr->tensor_like_.getShape()) {
      // compute chunk sizes from old, then swap
      ComputeCapacityAndChunkSizes(held_shape_,
                                   old_chunk_sizes_,
                                   capacity_);
      std::vector<int> transposed_chunk_size(transpose_ptr->getOrder());
      for (int axis = 0; axis < transpose_ptr->getOrder(); ++axis) {
        // transposed_chunk_size[axis] = old_chunk_sizes_[transpose_ptr->untranpose_map_[axis]];
        transposed_chunk_size[transpose_ptr->tranpose_map_[axis]] = old_chunk_sizes_[axis];
      }
      old_chunk_sizes_ = std::move(transposed_chunk_size);
    }

    T operator*() const override {
      return *it_;
    }

    ConstIterator& operator+=(int increment) override {
      // Early exits
      if (increment < 0) {
        return operator+=(-increment);
      } else if (increment == 0) {
        return *this;
      }
      // Rely on default incrementation
      Parent::operator+=(increment);

      // when at end, address cannot be deduced from idx alone
      size_t address;
      if (Parent::at_end_) {
        address = Parent::tensor_like_->getCapacity();
      } else {
        address = IndicesToAddress(held_shape_, 
                                   old_chunk_sizes_,
                                   Parent::current_indices_);
      }
      IncrementAddressTo(address);
      return *this;
    }
    ConstIterator& operator-=(int decrement) override {
      // Early exits
      if (decrement < 0) {
        return operator+=(-decrement);
      } else if (decrement == 0) {
        return *this;
      }
      // Rely on default decrementation
      Parent::operator-=(decrement);

      // as -= always sets to expected idx, can call address immediately 
      IncrementAddressTo(IndicesToAddress(held_shape_, 
                                          old_chunk_sizes_,
                                          Parent::current_indices_));
      return *this;
    }
  }; // End of ConstIterator

  ConstIterator begin() const {
    return {this, std::vector<int>(getOrder(), 0), false, tensor_like_.begin(), 0};
  }
  ConstIterator end() const {
    return {this, std::vector<int>(getOrder(), 0), true, tensor_like_.end(), getCapacity()};
  }
// End of Iterator -------------------------------------
}; // End of MultiTransposeOperation

/**
 *  sums two tensors with broadcasting applied. 
 */
template<typename T, typename HeldOperation1, typename HeldOperation2>
class SummationOperation : public TensorLike<T, SummationOperation<T, HeldOperation1, HeldOperation2>> {
  typedef SummationOperation<T, HeldOperation1, HeldOperation2> Self;
  typedef TensorLike<T, Self> Parent;

// Members ---------------------------------------------
    const std::vector<int> broadcast_shape_;
    const size_t broadcast_capacity_;

    const BroadcastOperation<T, HeldOperation1> first_;
    const BroadcastOperation<T, HeldOperation2> second_;
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  SummationOperation(const TensorLike<T, HeldOperation1>& first, 
                     const TensorLike<T, HeldOperation2>& second) 
      // Broadcasting also checks for compatiblity.
      : broadcast_shape_(Broadcast(first.getShape().begin(),  first.getShape().end(),
                                   second.getShape().begin(), second.getShape().end())),
        broadcast_capacity_(std::accumulate(broadcast_shape_.begin(), broadcast_shape_.end(),
                            1, std::multiplies<int>())),
        first_( BroadcastOperation(first, 
                                   broadcast_shape_, 
                                   broadcast_capacity_)), 
        second_(BroadcastOperation(second, 
                                   broadcast_shape_, 
                                   broadcast_capacity_)) {}
// End of Constructor ----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    return broadcast_shape_;
  } 
  inline int getDimension(int axis) const {
    return broadcast_shape_[SumIfNegative(axis, getOrder())];
  }
  inline size_t getCapacity() const noexcept {
    return broadcast_capacity_;
  }
  inline int getOrder() const noexcept {
    return broadcast_shape_.size();
  }
  inline const T getElement(const std::vector<int>& indices) const {
    return first_.getElement(indices) + second_.getElement(indices);
  }
// End of Tensor-Behaviours ----------------------------

  class ConstIterator : public Parent::ConstIterator {
    typedef typename BroadcastOperation<T, HeldOperation1>::ConstIterator BroadcastIterator1;
    typedef typename BroadcastOperation<T, HeldOperation2>::ConstIterator BroadcastIterator2;
    BroadcastIterator1 first_it_;
    BroadcastIterator2 second_it_;
   public:
    ConstIterator(BroadcastIterator1 first_it, 
                  BroadcastIterator2 second_it)
        : first_it_(first_it),
          second_it_(second_it) {}

    T operator*() const override {
      return *first_it_ + *second_it_;
    }

    ConstIterator& operator+=(int increment) override {
      first_it_ += increment;
      second_it_ += increment;
      return *this;
    }
    ConstIterator& operator-=(int decrement) override {
      first_it_ -= decrement;
      second_it_ -= decrement;
      return *this;
    }

    bool operator==(const ConstIterator& other) const override {
      return first_it_ == other.first_it_ && 
             second_it_ == other.second_it_;
    }
  };

  ConstIterator begin() const {
    return {first_.begin(), second_.begin()};
  }
  ConstIterator end() const {
    return {first_.end(), second_.end()};
  }
}; // End of SummationOperation

/**
 *  multiplies two tensors with broadcasting applied. 
 *  That is, broadcasting is applied to first order-2 dimensions. 
 *    Last two are treated as matrix.
 */
template<typename T, typename HeldOperation1, typename HeldOperation2>
class MultiplicationOperation : public TensorLike<T, MultiplicationOperation<T, HeldOperation1, HeldOperation2>> {
  typedef MultiplicationOperation<T, HeldOperation1, HeldOperation2> Self;
  typedef TensorLike<T, Self> Parent;

// Members ---------------------------------------------
  // product is resolved upon construct,
  //  so that other multiplcation algorithms can be used.
  Tensor<T>* product_tensor_;
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  // TODO, with this design, a special construcor for Tensor may be prefered for Multiplcation Holder, where a simple move is prefered
  // Product is handled upon constrcution
  MultiplicationOperation(const TensorLike<T, HeldOperation1>& A, 
                          const TensorLike<T, HeldOperation2>& B)
      // tensor is build after all checks
      : product_tensor_(nullptr) {
    // Broadcasting checks for compatiblity 
    std::vector<int> broad_cast_shape(Broadcast(A.getShape().begin(), A.getShape().end() - 2,
                                                B.getShape().begin(), B.getShape().end() - 2)); 
    if (A.getOrder() < 2) {
      throw std::invalid_argument("Multiplication- Insufficient Dimension");
    }
    if (A.getDimension(-1) != B.getDimension(-2)) {
      throw std::invalid_argument("Multiplication- Multiplcation Dimension Mismatch");
    }

    const int rows = A.getDimension(-2);
    const int cols = B.getDimension(-1);
    const int interm = A.getDimension(-1);

    // For A
    broad_cast_shape.push_back(rows);
    broad_cast_shape.push_back(interm);
    size_t capacity = std::accumulate(broad_cast_shape.begin(), broad_cast_shape.end(), 
                                      1, std::multiplies<int>());
    BroadcastOperation<T, HeldOperation1> A_broadcast(A, 
                                                      broad_cast_shape, 
                                                      capacity);
    // For B
    broad_cast_shape[broad_cast_shape.size() - 2] = interm;
    broad_cast_shape[broad_cast_shape.size() - 1] = cols;
    capacity = std::accumulate(broad_cast_shape.begin(), broad_cast_shape.end(), 
                               1, std::multiplies<int>());
    BroadcastOperation<T, HeldOperation2> B_broadcast(B, 
                                                      broad_cast_shape, 
                                                      capacity);
    // For product
    broad_cast_shape[broad_cast_shape.size() - 2] = rows;
    // col is readily set 
    product_tensor_ = new Tensor<T>(broad_cast_shape);

    // iterate through each matrix blocks
    std::vector<int> indices(getOrder(), 0);
    do {
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          indices[getOrder() - 2] = r;
          indices[getOrder() - 1] = c;

          T& element = product_tensor_->getElement(indices);
          element = T();

          for (int k = 0; k < interm; ++k) {
            indices[getOrder() - 2] = r;
            indices[getOrder() - 1] = k;
            T a_element = A_broadcast.getElement(indices);

            indices[getOrder() - 2] = k;
            indices[getOrder() - 1] = c;
            T b_element = B_broadcast.getElement(indices);

            element += a_element * b_element;
          }
        }
      }
    // increments for block-indices only, which are first order-2 axes
    } while (IncrementIndicesByShape(broad_cast_shape.begin(), broad_cast_shape.end() - 2,
                                     indices.begin(),          indices.end()          - 2));
  }
// End of Constructor ----------------------------------

// Destructor ------------------------------------------
  ~MultiplicationOperation() {
    delete product_tensor_;
  }
// End of Destructor -----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    return product_tensor_->getShape();
  } 
  inline int getDimension(int axis) const {
    return product_tensor_->getDimension(axis);
  }
  inline size_t getCapacity() const noexcept {
    return product_tensor_->getCapacity();
  }
  inline int getOrder() const noexcept {
    return product_tensor_->getOrder();
  }
  inline const T getElement(const std::vector<int>& indices) const {
    return product_tensor_->getElement(indices);
  }
// End of Tensor-Behaviours ----------------------------

// ! ConstTerator of Tensor moved to represent ConstIterator of this
  typedef typename Tensor<T>::ConstIterator ConstIterator;

  ConstIterator begin() const {
    return product_tensor_->begin();
  }
  ConstIterator end() const {
    return product_tensor_->end();
  }
// friends ---------------------------------------------
  friend Tensor<T>;   // For specialized constructor
// end of friedns --------------------------------------
}; // End of MultiplicationOperation

/**
 *  changes shape of tensor without altering capacity or order of elements.
 *  Chaning multiple Reshapes collapses to a single reshape.
 *  When reshaped as const, results in another layer of reshape, thus little less efficient.
 */
template<typename T, typename HeldOperation>
class ReshapeOperation : public TensorLike<T, ReshapeOperation<T, HeldOperation>> {
  typedef ReshapeOperation<T, HeldOperation> Self;
  typedef TensorLike<T, Self> Parent;

// Members ---------------------------------------------
  const HeldOperation& tensor_like_;
  std::vector<int> reshaped_dimension_;

  // below are members to be used for index altering
  std::vector<int> reshaped_chunk_size_; // to be computed
  size_t capacity_;                      // must be equal to tensor_like
// End of Members --------------------------------------

// Housekeeping ----------------------------------------
/**
 *  given that dimension is correctly set and that capacity and chunk_sizes are rest to 1
 *    updates the latter two according to dimensions.
 *  Checks and throws if newly set capacity does not match tensor_like_'s capacity
 */
  inline void UpdateAndCheck() {
    cpp_nn::util::ComputeCapacityAndChunkSizes(reshaped_dimension_, 
                                               reshaped_chunk_size_, 
                                               capacity_);
    if (capacity_ != tensor_like_.getCapacity()) {
      throw std::invalid_argument("TensorReshape- Capacity mismatch");
    }
  }
// End of Housekeeping ---------------------------------

 public:
// Constructor -----------------------------------------
  ReshapeOperation(const TensorLike<T, HeldOperation>& tensor_like, 
                   const std::vector<int>& new_shape)
      : tensor_like_(tensor_like.getRef()),
        reshaped_dimension_(new_shape),
        reshaped_chunk_size_(new_shape.size(), 1),
        capacity_(1) {
    UpdateAndCheck();
  }
// End of Constructor ----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    return reshaped_dimension_;
  } 
  inline int getDimension(int axis) const {
    return reshaped_dimension_[SumIfNegative(axis, getOrder())];
  }
  inline size_t getCapacity() const noexcept {
    return capacity_;
  }
  inline int getOrder() const noexcept {
    return reshaped_dimension_.size();
  }
  const T getElement(const std::vector<int>& indices) const {
    // reshape_indices -> address -> old_indices
    // !This again does not check for input validity.
    const size_t address = IndicesToAddress(reshaped_dimension_,
                                            reshaped_chunk_size_,
                                            indices);
    const std::vector<int> old_indices = AddressToIndices(tensor_like_.getShape(), 
                                                          address);
    return tensor_like_.getElement(old_indices);
  }

  // Non-const reshape
  inline ReshapeOperation<T, HeldOperation>&
         Reshape(const std::vector<int>& new_dimensions) {
    // Reset members
    reshaped_dimension_ = new_dimensions;
    reshaped_chunk_size_ = std::vector<int>(reshaped_dimension_.size(), 1);
    capacity_ = 1;
    // Compute members
    UpdateAndCheck();
    return *this;
  }
// End of Tensor-Behaviours ----------------------------

  typedef typename HeldOperation::ConstIterator ConstIterator;

  ConstIterator begin() const {
    return tensor_like_.begin();
  }
  ConstIterator end() const {
    return tensor_like_.end();
  }
// friend  ---------------------------------------------
  friend Tensor<T>;   // For specialized constructor
// end of friend  --------------------------------------
}; // End of ReshapeOperation

// TODO: Make FRont Padding.
/** 
 *  crops/pads the tensor to new shape. 
 *  The order must be same, but capacity may be differrent. 
 *  When pading is greater than old-shape, the values are padded to padded_val.
 *  When is less, becomes cropped and elements are lost.
 */
template<typename T, typename HeldOperation>
class PaddingOperation : public TensorLike<T, PaddingOperation<T, HeldOperation>> {
  typedef PaddingOperation<T, HeldOperation> Self;
  typedef TensorLike<T, Self> Parent;

// Members ---------------------------------------------
  const HeldOperation& tensor_like_;
  std::vector<int> padded_shape_;
  T padded_value_;

  size_t padded_capacity_;
// End of Members --------------------------------------

// Housekeeping ----------------------------------------
/** 
 *  given that members are validly set,
 *    checks and throws shape is correctly set
 */
  void CheckValidPadding() {
    if (padded_shape_.size() != tensor_like_.getOrder()) {
      throw std::invalid_argument("Padding- Order mismatch");
    }
    // dimension only need define valid dimension
    for (int axis = 0; axis < tensor_like_.getOrder(); ++axis) {
      if (padded_shape_[axis] > 0) {
        continue;
      }
      else {
        throw std::invalid_argument("Padding- Non-positive dimension");
      }
    }
  }
// End of Housekeeping ---------------------------------
 
 public:
// Constructor -----------------------------------------
  PaddingOperation(const TensorLike<T, HeldOperation>& tensor_like, 
                   const std::vector<int>& padded_shape, 
                   T padded_value = T()) 
      : tensor_like_(tensor_like.getRef()),
        padded_shape_(padded_shape),
        padded_value_(padded_value),
        padded_capacity_(std::accumulate(padded_shape_.begin(), padded_shape_.end(), 
                                         1, std::multiplies<int>())) {
    CheckValidPadding();
  }
// End of Constructor ----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    return padded_shape_;
  } 
  inline int getDimension(int axis) const {
    return padded_shape_[SumIfNegative(axis, getOrder())];
  }
  inline size_t getCapacity() const noexcept {
    return padded_capacity_;
  }
  inline int getOrder() const noexcept {
    return padded_shape_.size();
  }
  const T getElement(const std::vector<int>& indices) const {
    // forward order would work too, but we would most likely padd the inner dimension
    for (int axis = getOrder() - 1; axis >= 0; --axis) {
      if (indices[axis] >= tensor_like_.getDimension(axis)) {
        // if beyond old shape, return padded val
        return padded_value_;
      }
      // TODO, consider making crop validity too?
    }
    return tensor_like_.getElement(indices);
  }

  // Non-const padding
  PaddingOperation<T, HeldOperation>&
         Padding(const std::vector<int>& padded_dimensions, 
                 T padded_value = T()) {
    padded_shape_ = padded_dimensions;
    padded_value_ = padded_value;
    padded_capacity_ = std::accumulate(padded_shape_.begin(), padded_shape_.end(), 
                                       1, std::multiplies<int>());
    CheckValidPadding();
    return *this;
  }
// End of Tensor-Behaviours ----------------------------

  // type def instead of making another inner class
  typedef typename Parent::DefaultConstIterator ConstIterator;

  ConstIterator begin() const {
    return {this, std::vector<int>(getOrder(), 0), false};
  }
  ConstIterator end() const {
    return {this, std::vector<int>(getOrder(), 0), true};
  }
}; // End of PaddingOperation

/** 
 *  broadcasts tensor to desired shape. 
 *  Allows for indices to be called to broadcast shape, which 
 *    will repeat index when valid
 * 
 *  TODO: as it is implemented now, broadcasting is prepared and only passed
 *    onto this operation holder.
 *    That is, this class does little of actual broadcasting and just holds 
 *    the membvers.
 *    We may want top change the implementation so that broadcasting is done in here,
 *    or change the name to refelct what this is actually doing. 
 */
template<typename T, typename HeldOperation>
class BroadcastOperation : public TensorLike<T, BroadcastOperation<T, HeldOperation>> {
  typedef BroadcastOperation<T, HeldOperation> Self;
  typedef TensorLike<T, Self> Parent;

// Members ---------------------------------------------
  const HeldOperation& tensor_like_;
  const std::vector<int> broadcast_shape_;

  size_t broadcast_capacity_;
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  // passed broadcast shape is assumed to be validly formed, and so does no checks in construction
  BroadcastOperation(const TensorLike<T, HeldOperation>& tensor_like, 
                     const std::vector<int>& broadcast_shape,
                     const size_t broadcast_capacity) noexcept
      : tensor_like_(tensor_like.getRef()),
        broadcast_shape_(broadcast_shape),
        broadcast_capacity_(broadcast_capacity)  {}
// End of Constructor ----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    return broadcast_shape_;
  } 
  inline int getDimension(int axis) const {
    return broadcast_shape_[SumIfNegative(axis, getOrder())];
  }
  inline size_t getCapacity() const noexcept {
    return broadcast_capacity_;
  }
  inline int getOrder() const noexcept {
    return broadcast_shape_.size();
  }
  const T getElement(const std::vector<int>& indices) const {
    // TODO: Broadcast shape is not used here... should we have all axis check?

    // get last tensor_like.order indices and set all that is beyond dim as 0
    std::vector<int> passing_indices(indices.end() - tensor_like_.getOrder(), indices.end());
    int axis = passing_indices.size() - 1;
    std::vector<int>::const_iterator original_dim  = tensor_like_.getShape().end() - 1;    
    // with the assumption that broadcast shape is correct, only need to check if original dim is not 1
    while (axis >= 0) {
      if (*original_dim != 1) {
        // do noting
      } else {
        passing_indices[axis] = 0; 
      }
      --axis;
      original_dim--;
    }
    return tensor_like_.getElement(passing_indices);
  }
// End of Tensor-Behaviours ----------------------------

  // type def instead of making another inner class
  typedef typename Parent::DefaultConstIterator ConstIterator;

  // TODO: a clever iterator is iminnent, 
  class Temp : public Parent::ConstIterator {
    // maybe the only bezt option is same as DefaultConstIter
    /*
    
    size_t current_address_;
    vector<size_t> broadcast_checkmarks_; when address_ = checkmark, return to 0
    vector<size_t> check_mark_count;
    vector<size_t> current_checkmark_iteration;

    do"
    addres += increment;
    if (address >= checkmark_ && current_checkmark_iteration != check_mark_count){
      address -= cehckmark_
      current_checkmark_iteration ++
    }

    // Some recursive stuff like this.

    // Will need to come up with decrementation as well.
    
    */
  };

  // TODO
  /*
    if the two shapes are the same, then we would ideally want to simply make the default iterator occur.
    oitherwise there is iterator break in this step, making everyhting slow again

    OR 
    add in summation, a check for if broadcasting is needed if not then handle it without it?
  */

  ConstIterator begin() const {
    return {this, std::vector<int>(getOrder(), 0), false};
  }
  ConstIterator end() const {
    return {this, std::vector<int>(getOrder(), 0), true};
  }
}; // End of BroadcastOperation

} // unnamed namespace 

} // util
} // cpp_nn


#endif // CPP_NN_R_TENSOR_OPERATIONS
