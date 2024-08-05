#include <iostream>
#include <initializer_list>
#include <vector>
#include <functional>
#include <stdexcept>
#include <numeric>

#include <cstdlib>

#include "CPPNeuralNet/Utils/tensor.h"
#include "CPPNeuralNet/Utils/utils.h"



int main() {
  using namespace cpp_nn::util;

  Tensor<float> A({3, 2, 3}, [val = 0]() mutable {return ++val;});
  std::cout << "A" << std::endl;
  PrintTensor(A);

  Tensor<float> B = Apply<float, decltype(A)>(A, [](float a) {return a * 2;});
  std::cout << "B" << std::endl;
  PrintTensor(B);


  Tensor<float> C = Apply<float, decltype(A), decltype(B)>(A, B, [](float a, float b)->float
                            {
                              return a > 5 ? b : 0;
                            });
  std::cout << "C" << std::endl;
  PrintTensor(C);
}