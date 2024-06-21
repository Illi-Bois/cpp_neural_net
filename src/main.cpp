#include <iostream>
#include <initializer_list>
#include <vector>
#include <functional>
#include <stdexcept>

#include "CPPNeuralNet/Utils/utils.h"



class A {
 public:
  int* aptr_;
  int& b;
  A(int* aptr, int& c) : aptr_(aptr), b(c) {
  }

  void change() const {
    this->b = 13;
    *this->aptr_ = 13;
  }
};
class B {
 public:
  int* bptr_;
  B(const A& a) : bptr_(a.aptr_) {
    *a.aptr_ = 12;
  }
};

void test(int* const ptr) {
  *ptr = 100;
};


class AAAA {
 public:
  AAAA(const std::vector<int>& a) {
    std::cout << a[0];
  }
  AAAA(const std::initializer_list<int>& a) : AAAA(std::vector<int>(a)) {
  }
};


int conv(const std::vector<int>& indices, const std::vector<int>& dims, const std::vector<int> tm) {

  // Transpose is handled by the fact that Dimension is accessed in Transposed order
  int array_index = 0;
  // block_size is chuck-size that i-th index jumps each time.
  // Bottom-Up apporoach. By multiplying up the block_size, division is avoided
  for (int i = dims.size() - 1, block_size = 1; i >= 0; --i) {
    if (indices[tm[i]] >= 0 && indices[i] < dims[i]) {
      array_index += block_size * indices[tm[i]];
      block_size *=  dims[i]; // By passing through trans._map_, we can transpose in place
    } else {
      throw std::invalid_argument("TensorElement ElementGetter- Index Out of Bounds"); 
    }
  }

  return array_index;
};

int inc(std::vector<int>& idx, const std::vector<int>& dim, const std::vector<int> tm) {
  for (int i = idx.size() - 1; i >= 0; --i) {
    ++idx[i];
    if (idx[i] >= dim[tm[i]]) {
      idx[i] = 0;
    } else {
      return 1;
    }
  }

  return 0;
};

std::string toString(std::vector<int> arr) {
  std::string str;
  for (int i : arr) {
    str += std::to_string(i);
    str += " ";
  }

  return str;
};

int main() {
  // AAAA a({1});
  std::vector<int> dim{2,3,2};
  int cap = 1;
  for (int i : dim) {
    cap *= i;
  }
  std::vector<int> arr(cap, 0);
  for (int i = 0; i < cap; ++i) {
    arr[i] = i;
  }

  std::vector<int> tm = arr; 

  std::vector<int> idx(dim.size(), 0);

  do {
    std::cout << arr[conv(idx, dim, tm)] << std::endl;
  } while(inc(idx, dim, tm));

  tm = {2, 1, 0}; 

  idx = std::vector<int>(dim.size(), 0);
  std::cout << std::endl;
    do {
    std::cout << arr[conv(idx, dim, tm)] << std::endl;
  } while(inc(idx, dim, tm));

}
