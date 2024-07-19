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

// Not an Operation
template<typename T, typename HeldOperation1, typename HeldOperation2>
class BroadcastedPairHolder;
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

  class ConstIterator : public Parent::ConstIterator {
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

    T operator*() const override {
      return *it_;
    }

    ConstIterator& operator+=(int increment) override {
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
    ConstIterator& operator-=(int decrement) override {
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

    bool operator==(const ConstIterator& other) const override {
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
    // const std::vector<int> broadcast_shape_;
    // const size_t broadcast_capacity_;

    // const BroadcastOperation<T, HeldOperation1> first_;
    // const BroadcastOperation<T, HeldOperation2> second_;
  BroadcastedPairHolder<T, HeldOperation1, HeldOperation2> broadcast_pair_;
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  SummationOperation(const TensorLike<T, HeldOperation1>& first, 
                     const TensorLike<T, HeldOperation2>& second) 
      // Broadcasting also checks for compatiblity.
      : broadcast_pair_(first, second) {}
// End of Constructor ----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    return broadcast_pair_.getShape();
  } 
  inline int getDimension(int axis) const {
    return getShape()[SumIfNegative(axis, getOrder())];
  }
  inline size_t getCapacity() const noexcept {
    return broadcast_pair_.getCapacity();
  }
  inline int getOrder() const noexcept {
    return getShape().size();
  }
  inline const T getElement(const std::vector<int>& indices) const {
    return broadcast_pair_.getFirst().getElement(indices) +
           broadcast_pair_.getSecond().getElement(indices);
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
    return {broadcast_pair_.getFirst().begin(),
            broadcast_pair_.getSecond().begin()};
  }
  ConstIterator end() const {
    return {broadcast_pair_.getFirst().end(), 
            broadcast_pair_.getSecond().end()};
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

  typedef BroadcastOperation<T, HeldOperation1> BroadcastFirst;
  typedef BroadcastOperation<T, HeldOperation2> BroadcastSecond;

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
    // Multiplication possiblity check
    if (A.getOrder() < 2 || B.getOrder() < 2) {
      throw std::invalid_argument("Multiplication- Insufficient Dimension");
    }
    if (A.getDimension(-1) != B.getDimension(-2)) {
      throw std::invalid_argument("Multiplication- Multiplcation Dimension Mismatch");
    }

    // Broadcasting checks for compatiblity 
    std::vector<int> broad_cast_shape(Broadcast(A.getShape().begin(), A.getShape().end() - 2,
                                                B.getShape().begin(), B.getShape().end() - 2)); 

    const int rows = A.getDimension(-2);
    const int cols = B.getDimension(-1);
    const int interm = A.getDimension(-1);

    // !! Does not use conventional Broadcast. Need to have broadcast separaelty

    // First set product_shape
    broad_cast_shape.push_back(rows);
    broad_cast_shape.push_back(cols);

    const size_t order = broad_cast_shape.size();

    // Broadcast A
    std::vector<int> A_broadcast_shape(broad_cast_shape);
    A_broadcast_shape[order - 1] = interm;
    // TODO for this context these might be unnecessary to compute
    // we might as well pass broken chunk sizes
    std::vector<int> A_broadcast_chunk(order, 1);
    size_t A_capacity = 1;
    ComputeCapacityAndChunkSizes(A_broadcast_shape,
                                 A_broadcast_chunk,
                                 A_capacity);
    // but for now, use proper, then reevaluate when doing strassen TODO
    BroadcastFirst A_broadcast(A, 
                               A_broadcast_shape, 
                               A_broadcast_chunk,
                               A_capacity);
    // Broadcast B
    std::vector<int> B_broadcast_shape(broad_cast_shape);
    B_broadcast_shape[order - 2] = interm;
    // TODO for this context these might be unnecessary to compute
    // we might as well pass broken chunk sizes
    std::vector<int> B_broadcast_chunk(order, 1);;
    size_t B_capacity = 1;
    ComputeCapacityAndChunkSizes(B_broadcast_shape,
                                 B_broadcast_chunk,
                                 B_capacity);
    // but for now, use proper, then reevaluate when doing strassen TODO
    BroadcastSecond B_broadcast(B, 
                                B_broadcast_shape, 
                                B_broadcast_chunk,
                                B_capacity);

    // Final storage for C = A*B
    product_tensor_ = new Tensor<T>(broad_cast_shape);

    typename Tensor<T>::Iterator it = product_tensor_->begin();
    typename Tensor<T>::Iterator fin = product_tensor_->end();

    typename BroadcastFirst::ConstIterator  A_it = A_broadcast.begin();
    typename BroadcastSecond::ConstIterator B_it = B_broadcast.begin();

    while (it != fin) {
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          T& element = *it;

          element = T();

          for (int k = 0; k < interm; ++k) {
            element += (*A_it) * (*B_it);

            // Increment A to next column
            ++A_it;
            // increment to next row if only not at the last col
            if (k != interm - 1) {
              B_it += cols;
            }
          } /// k loop --

          // Reset A to start of row
          A_it -= interm;
          // rest to first row
          B_it -= cols * (interm - 1)
                  - 1; // Increment B to next col
          ++it;
        } /// c Loop --

        // Increment A to next row
        A_it += interm;
        // Reset B to first column
        B_it -= cols;
      }/// r Loop --

      // A naturally increments to next matrix by row incr.
      // B Move to next matrix
      B_it += interm * cols;
    } 
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
    return static_cast<const Tensor<T>*>(product_tensor_)->begin();
  }
  ConstIterator end() const {
    return  static_cast<const Tensor<T>*>(product_tensor_)->end();
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
  typedef typename Parent::DefaultConstIterator ConstIteratorTEMP;

  // Padding barrows from transpose iterator idea, where
  //    we keep running default vector indexer
  //    WHEN IN INCREMENTATION, it is deemed we are WITHIN PADDING, make a bool for that
  //    we increment the inner indexer too (using previous address)
  //  in accessor, if bool for INBOUND is true, return iterator, 
  //    if not, return padding value

  typedef typename HeldOperation::ConstIterator HeldIterator;

  class ConstIterator : public Parent::DefaultConstIterator {
    typedef typename Parent::DefaultConstIterator Parent;
   private:
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
        if (Parent::current_indices_[axis] >= *dim_it) {
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
      if (Parent::at_end_) {
        return;
      }

      if (in_bounds_) {
        // need to update the position
        size_t address_to = IndicesToAddress(inner_shape_,
                                             inner_chunk_sizes_,
                                             Parent::current_indices_);
        IncrementAddressTo(address_to);
      }
    }

   public:
    ConstIterator(const Self* transpose_ptr, 
                  std::vector<int> idx, 
                  bool is_end,
                  HeldIterator it,
                  size_t inner_address) 
        : Parent::DefaultConstIterator(transpose_ptr, 
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

    T operator*() const override {
      if (in_bounds_) {
        return *it_;
      }
      return padded_value_;
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

      SetInner();
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

      SetInner();
      return *this;
    }
  };

  ConstIterator begin() const {
    return {this, std::vector<int>(getOrder(), 0), false, tensor_like_.begin(), 0};
  }
  ConstIterator end() const {
    return {this, std::vector<int>(getOrder(), 0), true, tensor_like_.end(), tensor_like_.getCapacity()};
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

  const std::vector<int>& broadcasted_shape_;
  // To be compatible with BroadcastPairHolder, must be reference
  const std::vector<int>& broadcasted_chunk_sizes_;
  const size_t& broadcasted_capacity_;
// End of Members --------------------------------------

 public:
// Constructor -----------------------------------------
  // passed broadcast shape is assumed to be validly formed, and so does no checks in construction
  BroadcastOperation(const TensorLike<T, HeldOperation>& tensor_like, 
                     const std::vector<int>& broadcasted_shape,
                     const std::vector<int>& broadcasted_chunk_sizes,
                     const size_t& broadcasted_capacity) noexcept
      : tensor_like_(tensor_like.getRef()),
        broadcasted_shape_(broadcasted_shape),
        broadcasted_chunk_sizes_(broadcasted_chunk_sizes),
        broadcasted_capacity_(broadcasted_capacity)  {}
// End of Constructor ----------------------------------

// Tensor-Behaviours -----------------------------------
  inline const std::vector<int>& getShape() const noexcept {
    return broadcasted_shape_;
  } 
  inline int getDimension(int axis) const {
    return broadcasted_shape_[SumIfNegative(axis, getOrder())];
  }
  inline size_t getCapacity() const noexcept {
    return broadcasted_capacity_;
  }
  inline int getOrder() const noexcept {
    return broadcasted_shape_.size();
  }
  const T getElement(const std::vector<int>& indices) const {
    return tensor_like_.getElement(CutToShape(indices, tensor_like_.getShape()));
  }
// End of Tensor-Behaviours ----------------------------

  typedef typename HeldOperation::ConstIterator HeldIterator;

  class ConstIterator : public Parent::ConstIterator {
    const Self* const reference_;
    HeldIterator it_;

    size_t curr_address_;

    size_t original_capacity_;

    // To be constructed in const. body
    std::vector<int> detected_dimensions_;
    std::vector<int> detected_chunk_sizes_;

    size_t curr_inner_address_;
   public:
    ConstIterator(const Self* const ref, 
         HeldIterator it,
         size_t address) 
        : reference_(ref),
          it_(it),
          curr_address_(address),
          original_capacity_(reference_->tensor_like_.getCapacity()) {
      DetectBroadcastAxes(reference_->broadcasted_shape_,
                          reference_->broadcasted_chunk_sizes_,
                          reference_->tensor_like_.getShape(),
                          detected_dimensions_,
                          detected_chunk_sizes_);
      
      curr_inner_address_ = ConvertToUnbroadcastAddress(detected_dimensions_,
                                                        detected_chunk_sizes_,
                                                        original_capacity_,
                                                        curr_address_);
    }

    T operator*() const override {
      return *it_;
    }
    
    ConstIterator& operator+=(int increment) override {
      curr_address_ += increment;
      if (curr_address_ >= reference_->getCapacity()) {
        curr_address_ = reference_->getCapacity();
      }

      size_t new_inner_address = ConvertToUnbroadcastAddress(detected_dimensions_,
                                                             detected_chunk_sizes_,
                                                             original_capacity_,
                                                             curr_address_);
      
      if (new_inner_address > curr_inner_address_) {
        size_t diff = new_inner_address - curr_inner_address_;
        it_ += diff;
      } else {
        size_t diff = curr_inner_address_ - new_inner_address;
        it_ -= diff;
      }
      curr_inner_address_ = new_inner_address;

      return *this;
    }
    ConstIterator& operator-=(int decrement) override {
      if (decrement >= curr_address_) {
        curr_address_ = 0;
      } else {
        curr_address_ -= decrement;
      }

      size_t new_inner_address = ConvertToUnbroadcastAddress(detected_dimensions_,
                                                             detected_chunk_sizes_,
                                                             original_capacity_,
                                                             curr_address_);
      
      if (new_inner_address > curr_inner_address_) {
        size_t diff = new_inner_address - curr_inner_address_;
        it_ += diff;
      } else {
        size_t diff = curr_inner_address_ - new_inner_address;
        it_ -= diff;
      }
      curr_inner_address_ = new_inner_address;

      return *this;
    }

    bool operator==(const ConstIterator& other) const override {
      return reference_ == other.reference_ && 
              curr_address_ == other.curr_address_;
    }

  };

  ConstIterator begin() const {
    return {this, tensor_like_.begin(), 0};
  }
  ConstIterator end() const {
    return {this, tensor_like_.begin(), getCapacity()};
  }
}; // End of BroadcastOperation


// NOT A OPERATIONHOLDER, but instead a holder for operations
// internally holds two broadcastOperations
template<typename T, typename HeldOperation1, typename HeldOperation2>
class BroadcastedPairHolder {
  typedef BroadcastOperation<T, HeldOperation1> BroadcastFirst;
  typedef BroadcastOperation<T, HeldOperation2> BroadcastSecond;

  std::vector<int> broadcasted_shape_;

  // WILL PASS AS REFERENCE FIRST, THEN SET IN CONSTRUCT
  std::vector<int> broadcasted_chunk_sizes_;
  size_t broadcasted_capacity_;

  BroadcastFirst first_;
  BroadcastSecond second_;

  // above are to be shared to broadcast operation
  // as those will store reference to this only
 public:
  BroadcastedPairHolder(const TensorLike<T, HeldOperation1>& first_operation,
                        const TensorLike<T, HeldOperation2>& second_operation)
      : broadcasted_shape_(Broadcast(first_operation.getShape().begin(),  first_operation.getShape().end(),
                                     second_operation.getShape().begin(), second_operation.getShape().end())),
        broadcasted_chunk_sizes_(broadcasted_shape_.size(), 1),
        broadcasted_capacity_(1),
        // The broadcasts are passed unset vector and capacity, because they will now stroe them as rference
        first_ (first_operation,  broadcasted_shape_, broadcasted_chunk_sizes_, broadcasted_capacity_),
        second_(second_operation, broadcasted_shape_, broadcasted_chunk_sizes_, broadcasted_capacity_) {
    ComputeCapacityAndChunkSizes(broadcasted_shape_,
                                 broadcasted_chunk_sizes_,
                                 broadcasted_capacity_);
  }

  inline const BroadcastFirst& getFirst() const noexcept {
    return first_;
  }
  inline const BroadcastSecond& getSecond() const noexcept {
    return second_;
  }

  inline const std::vector<int>& getShape() const noexcept {
    return broadcasted_shape_;
  }
  inline size_t getCapacity() const noexcept {
    return broadcasted_capacity_;
  }
};

} // unnamed namespace 

} // util
} // cpp_nn


#endif // CPP_NN_R_TENSOR_OPERATIONS
