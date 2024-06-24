#include <iostream>
#include <initializer_list>
#include <vector>
#include <functional>
#include <stdexcept>

#include "CPPNeuralNet/Utils/sanity_check.h"
#include "CPPNeuralNet/Utils/tensor.h"

int main() {
  std::cout << "Hello World!!" << std::endl;

  TesterClass<int> test(10);
  std::cout << test.getA() << std::endl;

  cpp_nn::util::Tensor<int> tens({1, 2, 3});
  std::cout << tens.getDimension(0) << std::endl;

}
