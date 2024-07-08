#include "CPPNeuralNet/Utils/utils.h"

#include <iostream>

namespace cpp_nn {
namespace util {

bool IncrementIndicesByShape(const std::vector<int>::const_iterator shape_begin, 
                             std::vector<int>::const_iterator shape_end,
                             const std::vector<int>::const_iterator idx_begin,
                             std::vector<int>::iterator idx_end) noexcept {
  while (shape_end != shape_begin &&
         idx_end != idx_begin) {
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