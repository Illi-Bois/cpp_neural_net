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

  Tensor A = Tensor({2, 3, 4}, 0);
  int val = 0; 
  auto ait = A.begin();
  auto aend = A.end();

  while (ait != aend) {
    *ait = ++val;
    ++ait;
  }

  PrintTensor(A);

  // std::cout << "This is what it shoudl look like" << std::endl;
  // Tensor C = A.Transpose(0, 1).Reshape({3, 2, 4}).Transpose(0, 2);

  std::cout << "On multiple lines" << std::endl;
  Tensor C = A.Transpose(0, 1).Reshape({3, 2, 4});
  C = C.Transpose(0, 2);
  PrintTensor(C);

  std::cout << "On single lines" << std::endl;
  C = A.Transpose(0, 1).Reshape({3, 2, 4}).Transpose(0, 2);
  PrintTensor(C);


  std::cout << "without reshape in the middle" << std::endl;
  C = A.Transpose(0, 1).Transpose(0, 2);
  PrintTensor(C);

  // std::cout << "With multi" << std::endl;
  // Tensor D = A.Transpose(0, 1).Transpose(0, 2);
  // PrintTensor(D);
  
}