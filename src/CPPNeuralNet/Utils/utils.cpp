#include "CPPNeuralNet/Utils/utils.h"

#include <iostream>

namespace cpp_nn {
namespace util {


// Increment/Decrement =========================================
// Increments Once ----------------------------------------
bool IncrementIndicesByShape(const std::vector<int>::const_iterator shape_begin, 
                             std::vector<int>::const_iterator       shape_end,
                             const std::vector<int>::const_iterator idx_begin,
                             std::vector<int>::iterator             idx_end) noexcept {
  while (shape_end != shape_begin &&
         idx_end   != idx_begin) {
    shape_end--;
    idx_end--;
    if (++(*idx_end) >= *shape_end) {
      *idx_end = 0;
    } else {
      return true;
    }
  };
  return false;
}
bool DecrementIndicesByShape(const std::vector<int>::const_iterator shape_begin, 
                             std::vector<int>::const_iterator       shape_end,
                             const std::vector<int>::const_iterator idx_begin,
                             std::vector<int>::iterator             idx_end) noexcept {
  while (shape_end != shape_begin &&
         idx_end   != idx_begin) {
    shape_end--;
    idx_end--;
    if (*idx_end <= 0) {
      *idx_end = (*shape_end) - 1;
    } else {
      --*idx_end;
      return true;
    }
  };
  return false;
}
// End of Increments Once ---------------------------------

// Multiple Incrementatins --------------------------------
bool IncrementIndicesByShape(const std::vector<int>::const_iterator shape_begin, 
                             std::vector<int>::const_iterator       shape_end,
                             const std::vector<int>::const_iterator idx_begin,
                             std::vector<int>::iterator             idx_end,
                             int count) noexcept {
  if (count == 0) {
    return true;
  } else if (count == 1) {
    return IncrementIndicesByShape(shape_begin, 
                                   shape_end,
                                   idx_begin,
                                   idx_end);
  } else if (count == -1) {
    return DecrementIndicesByShape(shape_begin, 
                                   shape_end,
                                   idx_begin,
                                   idx_end);
  } else if (count < 0) {
    return MultipleDecrementIndicesByShape(shape_begin, 
                                           shape_end,
                                           idx_begin,
                                           idx_end,
                                           -count);
  }
  return MultipleIncrementIndicesByShape(shape_begin, 
                                         shape_end,
                                         idx_begin,
                                         idx_end,
                                         count);
}
bool DecrementIndicesByShape(const std::vector<int>::const_iterator shape_begin, 
                             std::vector<int>::const_iterator       shape_end,
                             const std::vector<int>::const_iterator idx_begin,
                             std::vector<int>::iterator             idx_end,
                             int count) noexcept {
  if (count == 0) {
    return true;
  } else if (count == 1) {
    return DecrementIndicesByShape(shape_begin, 
                                   shape_end,
                                   idx_begin,
                                   idx_end);
  } else if (count == -1) {
    return IncrementIndicesByShape(shape_begin, 
                                   shape_end,
                                   idx_begin,
                                   idx_end);
  } else if (count < 0) {
    return MultipleIncrementIndicesByShape(shape_begin, 
                                           shape_end,
                                           idx_begin,
                                           idx_end,
                                           -count);
  }
  return MultipleDecrementIndicesByShape(shape_begin, 
                                         shape_end,
                                         idx_begin,
                                         idx_end,
                                         count);
}
// End of Multiple Incrementatins -------------------------

// Recursive Increment Helpers ----------------------------
bool MultipleIncrementIndicesByShape(const std::vector<int>::const_iterator shape_begin, 
                                     std::vector<int>::const_iterator       shape_end,
                                     const std::vector<int>::const_iterator idx_begin,
                                     std::vector<int>::iterator             idx_end,
                                     int count) noexcept {
  // count is never given as negative or 0
  //   can assume count > 0

  if (shape_end == shape_begin ||
      idx_end   == idx_begin) {
    // BASE CASE
    return false; // End is reached and so is overflow

  } else { // ITERATING CASE
    --idx_end;
    --shape_end;

    *idx_end += count; // Try adding everything to current
    if (*idx_end >= *shape_end) { // If overflows in box
      // Need to carry over remaining
      count = *idx_end / *shape_end;  // count > 0 necessarily
      // Recurvively increment
      if (IncrementIndicesByShape(shape_begin, shape_end,
                                  idx_begin, idx_end,
                                  count)) {
        // If carry was sucessful
        // set current to the left-over
        *idx_end %= *shape_end;  
        return true;
      } else {  // If carry failed
        // set current to 0
        *idx_end = 0;
        return false;
      }

    } else { // If count fit in current, 
      return true; // incrementation was succ
    }
  }
}
bool MultipleDecrementIndicesByShape(const std::vector<int>::const_iterator shape_begin, 
                                     std::vector<int>::const_iterator       shape_end,
                                     const std::vector<int>::const_iterator idx_begin,
                                     std::vector<int>::iterator             idx_end,
                                     int count) noexcept {
  // count is never given as negative or 0
  //   can assume count > 0                               

  if (shape_end == shape_begin ||
      idx_end   == idx_begin) {
    // BASE CASE
    return false;  // End if reached meaning underflow

  } else { // ITERATING CASE
    --idx_end;
    --shape_end;

    // take away what can be taken away at current dimension
    *idx_end -= count % *shape_end;
    // remaining take-away to be carried
    count /= *shape_end;

    if (*idx_end < 0) {
      // IF current-take away is negative, means we will have to 
      //  carry down from next axis down
      // This is equivalent to incrementing count by 1
      *idx_end += *shape_end;
      ++count; 
    }

    if (count == 0) {
      // Takeaway at current dimension was successful, 
      // no need to carry
      return true;
    } else {
      // recursive call
      if (DecrementIndicesByShape(shape_begin, shape_end,
                                  idx_begin, idx_end,
                                  count)) {
        // If carry was successful, return true
        return true;
      } else {
        // Carry failed and thus underflowed
        // Set current to underflow-roll-over
        *idx_end = *shape_end - 1;
        return false;
      }
    }
  }
}
// End of Recursive Increment Helpers ---------------------
// End of Increment/Decrement ==================================



std::vector<int> Broadcast(const std::vector<int>::const_iterator first_shape_begin, 
                           std::vector<int>::const_iterator       first_shape_end,
                           const std::vector<int>::const_iterator second_shape_begin,
                           std::vector<int>::const_iterator       second_shape_end) {
  const size_t first_length = std::distance(first_shape_begin, first_shape_end);
  const size_t second_length = std::distance(second_shape_begin, second_shape_end);

  const size_t broadcast_length = std::max(first_length, second_length);

  std::vector<int> res(broadcast_length, 0);

  std::vector<int>::iterator curr = res.end();

  while (first_shape_begin  != first_shape_end &&
         second_shape_begin != second_shape_end) {
    first_shape_end--;
    second_shape_end--;
    curr--; // curr is always valid as length is <=

    // curr is set to the dimension if they are equal, or either equals 1
    // else incompatible
    if (*first_shape_end == 1 ||
        *first_shape_end == *second_shape_end) {
      *curr = *second_shape_end;
    } else if (*second_shape_end == 1) {
      *curr = *first_shape_end;
    } else {
      throw std::invalid_argument("Broadcast- Incompatible shapes");
    }
  }

  while (first_shape_begin != first_shape_end) {
    first_shape_end--;
    curr--; 

    *curr = *first_shape_end;
  }

  while (second_shape_begin != second_shape_end) {
    second_shape_end--;
    curr--; 

    *curr = *second_shape_end;
  }

  return res;
}


void ComputeCapacityAndChunkSizes(const std::vector<int>& shape,
                                  std::vector<int>& chunk_sizes,
                                  size_t& capacity) {
  for (int axis = shape.size() - 1; axis >= 1; --axis) {
    if (shape[axis] > 0) {
      chunk_sizes[axis - 1] = chunk_sizes[axis] * shape[axis];
    } else {
      throw std::invalid_argument("ChunkSize and Capacity- Non-positive dimension given");
    }
  }
  if (shape[0] > 0) {
    capacity = chunk_sizes[0] * shape[0];
  } else {
    throw std::invalid_argument("ChunkSize and Capacity- Non-positive dimension given");
  }
}

std::vector<int> AddressToIndices(const std::vector<int>& shape, size_t address) noexcept {
  std::vector<int> res(shape.size(), 0);
  int curr_dim;
  for (int axis = shape.size() - 1; axis >= 0; --axis) {
    curr_dim = shape[axis];
    res[axis] = address % curr_dim;
    address /= curr_dim;
    // quick exit
    if (address == 0) break;
  }
  return res;
}


size_t IndicesToAddress(const std::vector<int>& shape,
                        const std::vector<int>& chunk_sizes,
                        const std::vector<int>& indices) noexcept {
  size_t address = 0;
  for (int axis = indices.size() - 1; axis >= 0; --axis) {
    address += SumIfNegative(indices[axis], shape[axis]) * chunk_sizes[axis];
  }
  return address;
}

}
}