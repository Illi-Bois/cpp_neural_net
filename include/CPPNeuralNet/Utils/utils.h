#ifndef CPP_NN_UTIL
#define CPP_NN_UTIL


#include <vector>
#include <initializer_list>
#include <utility>

namespace cpp_nn {
namespace util {

/** increments index by given shape. 
 * Index will be matched at given shape's end
 *  and increment forth until shape's begi
 */
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
                             std::vector<int>::iterator             idx_end);


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