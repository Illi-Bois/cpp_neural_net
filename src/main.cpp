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

  // Tensor<int> A({2, 3, 4});
  // Tensor<int> B({4, 5});

  // int i = 0;
  // for (auto it = A.begin(); it != A.end(); ++it) {
  //   *it = ++i;
  // }
  // i = 0;
  // for (auto it = B.begin(); it != B.end(); ++it) {
  //   *it = ++i;
  // }

  // std::cout << "A" << std::endl;
  // PrintTensor(A);
  // std::cout << "B" << std::endl;
  // PrintTensor(B);

  // std::cout << "MULTIPLYING" << std::endl;
  // Tensor<int> C = A * B;
  // PrintTensor(C);



  auto generator = [val = 0]() mutable { return ++val; };
  Tensor<float> a({2, 3}, generator);
  Tensor<float> b({2, 3}, 2);

  PrintTensor(a);
  PrintTensor(b);

  
}