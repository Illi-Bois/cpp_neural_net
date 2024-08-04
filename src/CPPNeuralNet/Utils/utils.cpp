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



// Broadcasting ================================================
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

std::vector<int> CutToShape(const std::vector<int>& indices, const std::vector<int>& shape) noexcept {
  std::vector<int> cut_indices(indices.end() - shape.size(), indices.end());

  std::vector<int>::iterator it = cut_indices.begin();
  const std::vector<int>::iterator it_end = cut_indices.end();
  std::vector<int>::const_iterator it_shape = shape.begin();

  while (it != it_end) {
    if (*it_shape == 1) {
      *it = 0;
    }
    ++it;
    ++it_shape;
  }
  return cut_indices;
}

void DetectBroadcastAxes(const std::vector<int>& broadcast_shape,
                         const std::vector<int>& broadcast_chunk_sizes,
                         const std::vector<int>& original_shape,
                         std::vector<int>& ret_detected_dimensions,
                         std::vector<int>& ret_detected_chunk_size) noexcept {
  // assuming ret vectors are empty
  // at most, the detected axes are same as order of original_shape
  ret_detected_dimensions.reserve(original_shape.size());
  ret_detected_chunk_size.reserve(original_shape.size());

  //  all of the front-paddings are broadcasted, and for 
  //    intended computation can be ignored
  const int front_padding_size = broadcast_shape.size() - original_shape.size();
  for (int axis = 0; axis < original_shape.size(); ++axis) {
    if (broadcast_shape[front_padding_size + axis] != original_shape[axis]) {
      // assuming broadcast shape is valid, mismatch of dimension indicates broadcasting
      ret_detected_dimensions.push_back(broadcast_shape[front_padding_size + axis]);
      ret_detected_chunk_size.push_back(broadcast_chunk_sizes[front_padding_size + axis]);
    }
  }

  ret_detected_dimensions.shrink_to_fit();
  ret_detected_chunk_size.shrink_to_fit();
}

size_t ConvertToUnbroadcastAddress(const std::vector<int>& detected_broadcast_dim,
                                   const std::vector<int>& detected_broadcast_chunk_size,
                                   const size_t original_capacity,
                                   size_t broadcasted_address) noexcept {
  for (int axis = 0; axis < detected_broadcast_dim.size(); ++axis) {
    // see how many times we have repeated the chunksize
    const size_t repetition = broadcasted_address / detected_broadcast_chunk_size[axis];
    // every final index in dimension gets a 'pass'
    const size_t repetition_off_setter = repetition / detected_broadcast_dim[axis]; 

    // need to remove address by this count
    const size_t reset_count = repetition - repetition_off_setter;
    broadcasted_address -= detected_broadcast_chunk_size[axis] * reset_count;

    // TODO: can we summarize this into one line?
    // broadcasted_address -= detected_broadcast_chunk_size[axis] * ((broadcasted_address / detected_broadcast_chunk_size[axis]) - ((broadcasted_address / detected_broadcast_chunk_size[axis]) / detected_broadcast_dim[axis]));
  }
  // Underset, ensure is in bound
  //   needed because broadcast does not consider front-broadcasting
  broadcasted_address %= original_capacity;

  return broadcasted_address;
}
// End of Broadcasting =========================================


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

void ComputeCapacityAndChunkSizes(std::vector<int>::const_iterator shape_begin,
                                  std::vector<int>::const_iterator shape_end,
                                  std::vector<int>& chunk_sizes,
                                  size_t& capacity) {
  std::vector<int>::iterator chunk_sizes_iter = chunk_sizes.end();
  while (shape_end != (shape_begin + 1)) {
    --shape_end;
    --chunk_sizes_iter;
    if (*shape_end > 0) {
      *(chunk_sizes_iter - 1) = *chunk_sizes_iter * *shape_end;
    } else {
      throw std::invalid_argument("ChunkSize and Capacity- Non-positive dimension given");
    }
  }

  --shape_end;
  --chunk_sizes_iter;
  if (*shape_end > 0) {
    capacity = *chunk_sizes_iter * *shape_end;
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


// Tranpose Address Operations =================================
/* Two Axis only Transpose */
/** converts address on tranposed shape to address on untraposed shape */
size_t TranposedAddressToOriginalAddress(size_t transposed_address,
                                         const int dim1, const int chunk1,
                                         const int dim2, const int chunk2) noexcept {
  /* when tranposed, the index can be thought as 5-order shape

    indices name:
      AFTER,        idx1,             BETWEEN,      idx2,     BEFORE
      chunk2*dim2   chunk2*dim2/dim1  chunk1*dim2   chunk1    1
    with associated chunk-sizes as above

    we conver this to:
      AFTER,        idx2,             BETWEEN,      idx1,     BEFORE
      chunk2*dim2   chunk2            chunk1*dim1   chunk1    1

    Note that we can take advantage of those indices with identical chunk-sizes
      to skip some conversion
   */

  // when both axes are the same, address do not change
  if (dim1   == dim2 &&
      chunk1 == chunk2) {
    return transposed_address;
  }
  // order matters when tranpose is none
  const int idx1_chunk_size = chunk2  * dim2 / dim1;

  /** these are full-value, meaning idx * chunk size  */
  const size_t before_fv = transposed_address % chunk1;
  transposed_address -= before_fv;
  const size_t idx2_fv = transposed_address % (chunk1 * dim2);
  transposed_address -= idx2_fv;
  const size_t between_fv = transposed_address % (idx1_chunk_size);
  transposed_address -= between_fv;
  const size_t idx1_fv = transposed_address % (chunk2 * dim2);
  transposed_address -= idx1_fv;
  const size_t after_fv = transposed_address;

  return before_fv + 
         (idx1_fv    / idx1_chunk_size * chunk1) +
         (between_fv / dim2            * dim1) +
         (idx2_fv    / chunk1          * chunk2) +
         after_fv;
}
/** converts address on untranposed shape to address on traposed shape */
size_t OriginalAddressToTransposedAddress(size_t transposed_address,
                                          const int dim1, const int chunk1,
                                          const int dim2, const int chunk2) noexcept {
  /* when tranposed, the index can be thought as 5-order shape

    indices name:
      AFTER,        idx2,             BETWEEN,      idx1,     BEFORE
      chunk2*dim2   chunk2            chunk1*dim1   chunk1    1
    with associated chunk-sizes as above

    we conver this to:
      AFTER,        idx1,             BETWEEN,      idx2,     BEFORE
      chunk2*dim2   chunk2*dim2/dim1  chunk1*dim2   chunk1    1

    Note that we can take advantage of those indices with identical chunk-sizes
      to skip some conversion
   */
  /** these are full-value, meaning idx * chunk size  */
  const size_t before_fv = transposed_address % chunk1;
  transposed_address -= before_fv;
  const size_t idx1_fv = transposed_address % (chunk1 * dim1);
  transposed_address -= idx1_fv;
  const size_t between_fv = transposed_address % (chunk2);
  transposed_address -= between_fv;
  const size_t idx2_fv = transposed_address % (chunk2 * dim2);
  transposed_address -= idx2_fv;
  const size_t after_fv = transposed_address;

  return before_fv + 
         (idx2_fv    / chunk2          * chunk1) +
         (between_fv / dim1            * dim2) +
         (idx1_fv    / chunk1          * chunk2 * dim2 / dim1) +
         after_fv;
}
// End of Tranpose Address Operations ==========================

}
} 