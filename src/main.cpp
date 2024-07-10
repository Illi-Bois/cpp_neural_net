#include <iostream>
#include <initializer_list>
#include <vector>
#include <functional>
#include <stdexcept>
#include <numeric>

#include "CPPNeuralNet/Utils/tensor.h"
#include "CPPNeuralNet/Utils/utils.h"


int main() {
  using namespace cpp_nn::util;

  Tensor<int> A({2, 3, 4});
  Tensor<int>::Iterator it = A.begin();
  int i = 0;
  while (it != A.end()) {
    *it = ++i;
    ++it;
  }

  auto nit = const_cast<const Tensor<int>&>(A).begin();
  auto nend = const_cast<const Tensor<int>&>(A).end();
  while (nit != nend) {
    std::cout << *nit << ", ";
    ++nit;
  }
  std::cout << std::endl;

  PrintTensor(A);

  // A = A.Transpose().Transpose() + (A.Transpose() * A);
  A = A.Transpose();
  PrintTensor(A);
}




