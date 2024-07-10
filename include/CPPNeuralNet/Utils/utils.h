#ifndef CPP_NN_UTIL
#define CPP_NN_UTIL


#include <vector>
#include <utility>

namespace cpp_nn {
namespace util {

// Increment/Decrement =========================================
// Increments Once ----------------------------------------
/**
 *  increments indices vector by one according to the shape vector.
 *  If overflow, returns False and sets idx to 0s.
 */
bool IncrementIndicesByShape(const std::vector<int>::const_iterator shape_begin, 
                             std::vector<int>::const_iterator       shape_end,
                             const std::vector<int>::const_iterator idx_begin,
                             std::vector<int>::iterator             idx_end) noexcept;
/**
 *  dencrements indices vector by one according to the shape vector.
 *  If underflow, returns False and sets idx to maximum index, which is dimension-1.
 */
bool DecrementIndicesByShape(const std::vector<int>::const_iterator shape_begin, 
                             std::vector<int>::const_iterator       shape_end,
                             const std::vector<int>::const_iterator idx_begin,
                             std::vector<int>::iterator             idx_end) noexcept;
// End of Increments Once ---------------------------------

// Multiple Incrementatins --------------------------------
/** 
 *  increments count number of times. 
 *  If overflows before count, returns False and sets idx to 0.
 */
bool IncrementIndicesByShape(const std::vector<int>::const_iterator shape_begin, 
                             std::vector<int>::const_iterator       shape_end,
                             const std::vector<int>::const_iterator idx_begin,
                             std::vector<int>::iterator             idx_end,
                             int count) noexcept;
/** 
 *  decrements count number of times. 
 *  If underflows before count, 
 *    returns False and sets idx to maximum index, which is dimension-1.
 */
bool DecrementIndicesByShape(const std::vector<int>::const_iterator shape_begin, 
                             std::vector<int>::const_iterator       shape_end,
                             const std::vector<int>::const_iterator idx_begin,
                             std::vector<int>::iterator             idx_end,
                             int count) noexcept;
// End of Multiple Incrementatins -------------------------


// Recursive Increment Helpers ----------------------------
bool MultipleIncrementIndicesByShape(const std::vector<int>::const_iterator shape_begin, 
                                     std::vector<int>::const_iterator       shape_end,
                                     const std::vector<int>::const_iterator idx_begin,
                                     std::vector<int>::iterator             idx_end,
                                     int count) noexcept;
bool MultipleDecrementIndicesByShape(const std::vector<int>::const_iterator shape_begin, 
                                     std::vector<int>::const_iterator       shape_end,
                                     const std::vector<int>::const_iterator idx_begin,
                                     std::vector<int>::iterator             idx_end,
                                     int count) noexcept;
// End of Recursive Increment Helpers ---------------------
// End of Increment/Decrement ==================================

/**
 *  given sub-vectors defining shapes two tensors to be broadcasted,
 *    returns shape of broadcasted tensor.
 *  
 *  throws "Broadcast- Incompatible Shapes" when two are not compatible by broadcasting
 * 
 *  Broadcast on empty will result in correct shapes too,
 *    when both are empty result is empty as well
 */
std::vector<int> Broadcast(const std::vector<int>::const_iterator first_shape_begin, 
                           std::vector<int>::const_iterator       first_shape_end,
                           const std::vector<int>::const_iterator second_shape_begin,
                           std::vector<int>::const_iterator       second_shape_end);

/** 
 *  adds the two values if first arguement if negative.
 *  Returns first value if it is non-negative.
 */
inline size_t SumIfNegative(int index, int capacity) noexcept {
  return index < 0 ? index + capacity
                   : index;
}

/**
 *  computes and updates chunk_size and capacity from given shape. 
 *  
 *  Assumes: 
 *    shape and chnnk_size are same order
 *    chunk_size are all set to 1
 *    capacity is set to 1
 * 
 *  Chunk Size and Capacity, therefore, are passed by reference 
 */
void ComputeCapacityAndChunkSizes(const std::vector<int>& shape,
                                  std::vector<int>& chunk_sizes,
                                  size_t& capacity);

/**
 *  computes the indices corresponding to the address on the given dimension shape.
 * 
 *  Skip bound-check for efficiency. 
 */
std::vector<int> AddressToIndices(const std::vector<int>& shape, size_t address) noexcept;

/**
 *  computes address from indices.
 *  Requires both shape and chunk sizes pre-computed from the shape. 
 *  
 *  Skip bound-check for efficiency. 
 */
size_t IndicesToAddress(const std::vector<int>& shape,
                        const std::vector<int>& chunk_sizes,
                        const std::vector<int>& indices) noexcept;


// TODO: move somewhere else more appropriate?
template<typename T, typename Derived>
struct IteratorInterface {
 public:
  virtual const T& operator*() const {
    return const_cast<const T&>(const_cast<IteratorInterface<T, Derived>&>(*this).operator*());
  }
  virtual T& operator*() = 0;

  virtual Derived& operator+=(int increment) = 0;
  virtual Derived& operator-=(int decrement) = 0;

  Derived& operator++() {
    return (*this) += 1;
  }
  Derived& operator--() {
    return (*this) -= 1;
  }

  virtual bool operator==(const Derived& other) const = 0;
  bool operator!=(const Derived& other) const {
    return !(*this == other);
  }
};

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