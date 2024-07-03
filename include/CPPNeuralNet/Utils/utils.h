#ifndef CPP_NN_UTIL
#define CPP_NN_UTIL


#include <vector>
#include <utility>

namespace cpp_nn {
namespace util {
/** 
 *  increments given vector defining shape of a tensor and vector defining 
 *    indicies to be traversed, increments the indicies vector to the next 
 *    within the defined shape. 
 *  If next index does not exist, meaning iteration is terminated,
 *    returns false. Otherwise returns true.
 * 
 *  Iterators are given defining begin and end of vector. 
 *    Incrementation is only performed within given order.
 *    If either vector is shroter than the other,
 *      they are right-aligned and only considered until smaller
 *      terminates.
 */
bool IncrementIndicesByShape(const std::vector<int>::const_iterator shape_begin, 
                             std::vector<int>::const_iterator       shape_end,
                             const std::vector<int>::const_iterator idx_begin,
                             std::vector<int>::iterator             idx_end) noexcept;
/**
 *  given sub-vectors defining shapes two tensors to be broadcasted,
 *    returns shape of broadcasted tensor.
 *  
 *  throws "Broadcast- Incompatible Shapes" when two are not compatible by broadcasting
 */
std::vector<int> Broadcast(const std::vector<int>::const_iterator first_shape_begin, 
                           std::vector<int>::const_iterator       first_shape_end,
                           const std::vector<int>::const_iterator second_shape_begin,
                           std::vector<int>::const_iterator       second_shape_end);

/** 
 *  adds the two values if first arguement if negative.
 *  Returns first value if it is non-negative.
 */
inline size_t SumIfNegative(int index, int capacity) {
  return index < 0 ? index + capacity
                   : index;
}

// computes chunk size and capacity
//  updates chunk_size and capacity in place
// Will assume capacity and chunk_size is all set to one and correvt shape
void ComputeCapacityAndChunkSizes(const std::vector<int>& shape,
                                  std::vector<int>& chunk_size,
                                  size_t& capacity);


std::vector<int> AddressToIndices(const std::vector<int>& shape, int address);

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