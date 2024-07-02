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

  while (first_shape_begin != first_shape_end &&
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

}
}