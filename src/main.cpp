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

  {
    Tensor<int> A({2, 3, 4}, [val=0]()mutable {return val++;});
    Tensor<int> B = A.Transpose().Transpose();


    PrintTensor(A);
    PrintTensor(B);

  }
}