/** 
 * Handles operations on tensor as a whole. Often manipulating the shape.
 * 
 * Includes axis sum as well
 */

#ifndef CPP_NN_TENSORWISE_OPERATIONS
#define CPP_NN_TENSORWISE_OPERATIONS

#include "CPPNeuralNet/Utils/utils.h"
#include "CPPNeuralNet/Utils/tensor_like.h"

#include <numeric> // to fill from 0 to n

#include <iostream>

namespace cpp_nn {
namespace util {

// Forward Declaration ======================================
template<typename T>
class Tensor;

namespace {
template<typename T, typename HeldOperation>
class MultiTransposeOperation;
}
// End of Forward Declaration ===============================

namespace {
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
  // axis_1 is the further in of the two, meaning chunk1 < chunk2
  const int axis_1_;
  const int axis_2_;

  std::vector<int> dimension_;    // dimension post-tranposing

  const int chunk_size_1_;
  const int chunk_size_2_;
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  TransposeOperation(const TensorLike<T, HeldOperation>& tensor_like, 
                     const int axis_1, const int axis_2)
      : tensor_(tensor_like.getRef()), 
        // axis are stored as positive int, though it can be given as negative
        axis_1_(SumIfNegative(std::max(axis_1, axis_2), tensor_.getOrder())), 
        axis_2_(SumIfNegative(std::min(axis_1, axis_2), tensor_.getOrder())), 
        dimension_(tensor_like.getShape()),
        chunk_size_1_(std::accumulate(dimension_.begin() + axis_1_ + 1, dimension_.end(), 
                                      1, std::multiplies<int>())),
        chunk_size_2_(std::accumulate(dimension_.begin() + axis_2_ + 1, dimension_.end(), 
                                      1, std::multiplies<int>())) {
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

  class ConstIterator : public Parent::template ConstIterator<ConstIterator> {
   private:
  // Members ---------------------------------------------
    HeldIterator it_;
    size_t curr_address_;
    size_t inner_address_;

    // with chunk1 < chunk2
    const int dim1_;
    const int chunk1_;

    const int dim2_;
    const int chunk2_;

    const size_t capacity_;
  // End of Members --------------------------------------

  // Housekeeping ----------------------------------------
  /** Updates current_address to given address */
    void IncrementAddressTo(size_t address_to) {
      size_t diff;
      if (inner_address_ > address_to) {
        diff = inner_address_ - address_to;
        it_ -= diff;;
      } else {
        diff = address_to - inner_address_;
        it_ += diff;;
      }
      inner_address_ = address_to;
    }
  // End of Housekeeping ---------------------------------

   public:
    ConstIterator(HeldIterator it,
                  size_t address,
                  size_t inner_address,
                  int dim1, int chunk1,
                  int dim2, int chunk2,
                  size_t capacity)
        : it_(it),
          curr_address_(address),
          inner_address_(inner_address),
          dim1_(dim1), 
          chunk1_(chunk1),
          dim2_(dim2), 
          chunk2_(chunk2),
          capacity_(capacity) {
      // TODO: should inner iterator not be set at construction?
    }

    T operator*() const {
      return *it_;
    }

    ConstIterator& operator+=(int increment) {
      if (increment < 0) return operator-=(-increment);
      if (increment == 0) return *this;

      curr_address_ += increment;
      if (curr_address_ >=  capacity_) {
        // not expected to yield anything anyway, do nto incement
        curr_address_ = capacity_;
        return *this;
      }

      IncrementAddressTo(TranposedAddressToOriginalAddress(curr_address_,
                                                           dim1_, chunk1_, 
                                                           dim2_, chunk2_));
      return *this;
    }
    ConstIterator& operator-=(int decrement) {
      if (decrement < 0) return operator+=(-decrement);
      if (decrement == 0) return *this;

      curr_address_ -= decrement;
      if (curr_address_ < 0) {
        // not expected to yield anything anyway, do nto incement
        curr_address_ = 0;
      }

      IncrementAddressTo(TranposedAddressToOriginalAddress(curr_address_,
                                                           dim1_, chunk1_, 
                                                           dim2_, chunk2_));
      return *this;
    }

    bool operator==(const ConstIterator& other) const {
      // with the assumption that we are pointing to the same data,
      return curr_address_ == other.curr_address_;
    }

  }; // End of ConstIterator

  ConstIterator begin() const {
    return {tensor_.begin(), 
            0, 
            0, // inner can just be 0
            tensor_.getDimension(axis_1_), chunk_size_1_,
            tensor_.getDimension(axis_2_), chunk_size_2_,
            getCapacity()};
  }
  ConstIterator end() const {
    return {tensor_.begin(), 
            getCapacity(), 
            0, // inner can just be 0
            tensor_.getDimension(axis_1_), chunk_size_1_,
            tensor_.getDimension(axis_2_), chunk_size_2_,
            getCapacity()};  
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

  class ConstIterator : public Parent::template ConstIterator<ConstIterator>,
                        private Parent::DefaultConstIterator {
   private:
    typedef typename Parent::DefaultConstIterator DefaultConstIterator;
    typedef typename Parent::template ConstIterator<ConstIterator> Parent; // order matter
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
        : DefaultConstIterator::DefaultConstIterator(transpose_ptr, 
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

    T operator*() const {
      return *it_;
    }

    ConstIterator& operator+=(int increment) {
      // Early exits
      if (increment < 0) {
        return operator+=(-increment);
      } else if (increment == 0) {
        return *this;
      }
      // Rely on default incrementation
      DefaultConstIterator::operator+=(increment);

      // when at end, address cannot be deduced from idx alone
      size_t address;
      if (DefaultConstIterator::at_end_) {
        address = DefaultConstIterator::tensor_like_->getCapacity();
      } else {
        address = IndicesToAddress(held_shape_, 
                                   old_chunk_sizes_,
                                   DefaultConstIterator::current_indices_);
      }
      IncrementAddressTo(address);
      return *this;
    }
    ConstIterator& operator-=(int decrement) {
      // Early exits
      if (decrement < 0) {
        return operator+=(-decrement);
      } else if (decrement == 0) {
        return *this;
      }
      // Rely on default decrementation
      DefaultConstIterator::operator-=(decrement);

      // as -= always sets to expected idx, can call address immediately 
      IncrementAddressTo(IndicesToAddress(held_shape_, 
                                          old_chunk_sizes_,
                                          DefaultConstIterator::current_indices_));
      return *this;
    }

    using Parent::operator++;
    using Parent::operator--;
    using Parent::operator!=;
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

  typedef typename HeldOperation::ConstIterator HeldIterator;

  class ConstIterator : public Parent::template ConstIterator<ConstIterator>,
                        private Parent::DefaultConstIterator {
   private:
    typedef typename Parent::DefaultConstIterator DefaultConstIterator;
    typedef typename Parent::template ConstIterator<ConstIterator> Parent; // order matter
  // Members ---------------------------------------------
    HeldIterator it_;
    size_t inner_address_; // address of inner

    const T& padded_value_;
    const std::vector<int>& inner_shape_;
    std::vector<int> inner_chunk_sizes_;


    // axies where padding has happened. when no padding is found, is empty
    std::vector<int> padded_axes_;
    std::vector<int> original_axes_dimensions_;

    bool in_bounds_;
  // End of Members --------------------------------------

    void SetIsInBound() noexcept {
      // if not in bound for any of thre padded axes, then not in bound
      std::vector<int>::const_iterator dim_it = original_axes_dimensions_.begin();
      for (int axis : padded_axes_) {
        if (DefaultConstIterator::current_indices_[axis] >= *dim_it) {
          in_bounds_ = false;
          return;
        }
        ++dim_it;
      }
      // otherwise is in bound
      in_bounds_ = true;
    }
 /** Updates current_address to given address */
    void IncrementAddressTo(size_t address_to) noexcept {
      size_t diff;
      if (inner_address_ > address_to) {
        diff = inner_address_ - address_to;
        it_ -= diff;;
      } else {
        diff = address_to - inner_address_;
        it_ += diff;;
      }
      inner_address_ = address_to;
    }

    void SetInner() noexcept {
      // compute initial position
      SetIsInBound();

      // If out of bounds, ignore update
      if (DefaultConstIterator::at_end_) {
        return;
      }

      if (in_bounds_) {
        // need to update the position
        size_t address_to = IndicesToAddress(inner_shape_,
                                             inner_chunk_sizes_,
                                             DefaultConstIterator::current_indices_);
        IncrementAddressTo(address_to);
      }
    }

   public:
    ConstIterator(const Self* transpose_ptr, 
                  std::vector<int> idx, 
                  bool is_end,
                  HeldIterator it,
                  size_t inner_address) 
        : DefaultConstIterator::DefaultConstIterator(transpose_ptr, 
                                                     idx, 
                                                     is_end),
          it_(it),
          inner_address_(inner_address),
          padded_value_(transpose_ptr->padded_value_),
          inner_shape_(transpose_ptr->tensor_like_.getShape()),
          inner_chunk_sizes_(transpose_ptr->getOrder(), 1) {
      // compute axes where padding has happened,
      // we only care for axis where curr dimension is stictly larger
      //  if it is lesser, we can address it simply skipping values 

      padded_axes_.reserve(inner_shape_.size());
      original_axes_dimensions_.reserve(inner_shape_.size());

      for (int axis = 0; axis < transpose_ptr->getOrder(); ++axis) {
        if (transpose_ptr->getDimension(axis) > inner_shape_[axis]) {
          padded_axes_.push_back(axis);
          original_axes_dimensions_.push_back(inner_shape_[axis]);
        }
      }

      padded_axes_.shrink_to_fit();
      original_axes_dimensions_.shrink_to_fit();

      // Compute chunk_sizes for address conversion
      size_t inner_cap = 1;
      ComputeCapacityAndChunkSizes(inner_shape_,
                                   inner_chunk_sizes_,
                                   inner_cap);
      

      SetInner();
    }

    T operator*() const {
      if (in_bounds_) {
        return *it_;
      }
      return padded_value_;
    }

    ConstIterator& operator+=(int increment) {
      // Early exits
      if (increment < 0) {
        return operator+=(-increment);
      } else if (increment == 0) {
        return *this;
      }
      // Rely on default incrementation
      DefaultConstIterator::operator+=(increment);

      SetInner();
      return *this;
    }
    ConstIterator& operator-=(int decrement) {
      // Early exits
      if (decrement < 0) {
        return operator+=(-decrement);
      } else if (decrement == 0) {
        return *this;
      }
      // Rely on default decrementation
      DefaultConstIterator::operator-=(decrement);

      SetInner();
      return *this;
    }

    using Parent::operator++;
    using Parent::operator--;
    using Parent::operator!=;
  };

  ConstIterator begin() const {
    return {this, std::vector<int>(getOrder(), 0), false, tensor_like_.begin(), 0};
  }
  ConstIterator end() const {
    return {this, std::vector<int>(getOrder(), 0), true, tensor_like_.end(), tensor_like_.getCapacity()};
  }
}; // End of PaddingOperation

/** 
 * Summs over axes, preversing major and minor shapes. When order is 1, SPECIAL CASE
 */
template<typename T, typename HeldOperation>
class AxisSummationOperation : public TensorLike<T, AxisSummationOperation<T, HeldOperation>> {
  typedef AxisSummationOperation<T, HeldOperation> Self;
  typedef TensorLike<T, Self> Parent;

// Members ---------------------------------------------
  const HeldOperation& tensor_like_;
  int collapse_axis_; 
  std::vector<int> collapsed_shape_;

  size_t collapse_count_; // := equal to dimension of axis that collapses
  size_t minor_jump_; // := chunksize of axis      // from last comment, is equal to chunk size at computed's[0]
  size_t major_jump_; // := chunksize of axis-1    // we can use computeChunkSize from util, where major_jump is capacity of subarray from [axis: end]

  size_t capacity_;
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  AxisSummationOperation(const TensorLike<T, HeldOperation>& tensor_like,
                         const int axis = 0)
      : tensor_like_(tensor_like.getRef()),
        collapse_axis_(SumIfNegative(axis, tensor_like_.getOrder())),
        collapsed_shape_(tensor_like_.getOrder() - 1),
        collapse_count_(tensor_like_.getDimension(collapse_axis_)),
        minor_jump_(std::accumulate(tensor_like_.getShape().begin() + collapse_axis_ + 1, 
                                    tensor_like_.getShape().end(), 
                                    1, std::multiplies<int>())),
        major_jump_(minor_jump_ * collapse_count_),
        capacity_(tensor_like_.getCapacity() / collapse_count_) {
    if (collapse_axis_ >= tensor_like_.getOrder()) {
      throw std::invalid_argument("AxisSummationOperation- Axis out of bounds");
    }
    // Move over other non-collapsed dimensions. When none, becomes 1
    if (collapsed_shape_.size() == 0) {
      collapsed_shape_.push_back(1);
    } else {
      int collpased_idx = 0;
      for (int i = 0; i < collapse_axis_; ++i) {
        collapsed_shape_[collpased_idx] = tensor_like_.getDimension(i);
        ++collpased_idx;
      }
      for (int i = collapse_axis_ + 1; i < tensor_like_.getOrder(); ++i) {
        collapsed_shape_[collpased_idx] = tensor_like_.getDimension(i);
        ++collpased_idx;
      }
    }
  }
// End of Constructor ----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept { 
    return collapsed_shape_;
  }
  inline int getDimension(int axis) const {
    return collapsed_shape_[SumIfNegative(axis, getOrder())];
  }
  inline size_t getCapacity() const noexcept {
    return capacity_;
  }
  inline int getOrder() const noexcept {
    return collapsed_shape_.size();
  }
  // returns by value as operation returns are temp
  inline const T getElement(std::vector<int> indices) const {
    // TODO: try implement iterator?
    //    Potential speed up, but will need to then store chunk_size of old. 
    //    Classic Speed vs Space balance. 
    //  When non-default Iterator is implemented, the benefit may be much reduced.

    indices.insert(indices.begin() + collapse_axis_, 0);
    T sum = 0;
    for (int i = 0; i < collapse_count_; ++i) {
      sum += tensor_like_.getElement(indices);
      indices[collapse_axis_] += 1;
    }
    return sum;
  }
// End of Tensor-Behaviours ----------------------------
  
// Iterator --------------------------------------------
  typedef typename HeldOperation::ConstIterator HeldIterator;

  class ConstIterator : public Parent::template ConstIterator<ConstIterator> {
    HeldIterator iter; // when at iter end, then is at AxisSumEnd as well
    size_t collapse_count_;
    size_t minor_jump_; 
    size_t major_jump_; 

    // Position within the minor shape. BEgin and End both start at 0.
    int minor_idx_;
   public:
    ConstIterator(HeldIterator iter,
                  size_t collapse_count,
                  size_t minor_jump,
                  size_t major_jump,
                  size_t minor_idx)
        : iter(iter),
          collapse_count_(collapse_count),
          minor_jump_(minor_jump),
          major_jump_(major_jump),
          minor_idx_(minor_idx) {}

    T operator*() const {
      // Ideally would not 
      HeldIterator curr_iter = this->iter;
      // Jump and sum 
      T sum = 0;
      for (int i = 0; i < collapse_count_ - 1; ++i) {
        sum += *curr_iter;
        curr_iter += minor_jump_;
      }
      sum += *curr_iter;
      return sum;
    }

    ConstIterator& operator+=(int increment) {
      if (increment < 0) {
        return operator-=((-increment));
      } 
      if (increment == 0) {
        return *this;
      }

      int incr_within_minor_block = increment % minor_jump_;
      increment -= incr_within_minor_block;

      if (minor_idx_ + incr_within_minor_block >= minor_jump_) {
        // Carry down
        increment += minor_jump_;
        incr_within_minor_block -= minor_jump_;
      }
      // update minor-position
      minor_idx_ += incr_within_minor_block;

      increment *= collapse_count_;
      increment += incr_within_minor_block;

      iter += increment;
      return *this;
    }

    ConstIterator& operator-=(int decrement) {
      // Early exits
      if (decrement < 0) {
        return operator+=(-decrement);
      } else if (decrement == 0) {
        return *this;
      }

      int decr_within_minor_block = decrement % minor_jump_;
      decrement -= decr_within_minor_block;

      if (minor_idx_ - decr_within_minor_block < 0) {
        // Carry down
        decrement += minor_jump_;
        decr_within_minor_block -= minor_jump_;
      }
      // update minor-position
      minor_idx_ -= decr_within_minor_block;

      decrement *= collapse_count_; // decrement is associated with major jump, as minor travel is cut out
      decrement += decr_within_minor_block; // add bck the minor travel

      iter -= decrement;
      return *this;
    }

    bool operator==(const ConstIterator& other) const {
      return iter == other.iter;
    }
  }; // End of ConstIterator

// Iterator Callers --------------------------------
  ConstIterator begin() const {
    return {tensor_like_.begin(),
            collapse_count_,
            minor_jump_,
            major_jump_,
            0};
  }
  ConstIterator end() const {
    return {tensor_like_.end(),
            collapse_count_,
            minor_jump_,
            major_jump_,
            0};
  }
// End of Iterator Callers -------------------------
// End of Iterator -------------------------------------

}; // End of AxisSummationOperation

} // unnamed namespace
} // util
} // cpp_nn

#endif // CPP_NN_TENSORWISE_OPERATIONS