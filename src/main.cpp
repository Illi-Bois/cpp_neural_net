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

  Tensor<int> A({1, 2, 3});
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

  // A = (A.Transpose().Transpose().Padding({2, 4, 4}) + (A.Transpose() * A)).Padding({2, 2, 2}).Reshape({2, 2, 2}).Padding({2, 3, 3});
  // A = A.Transpose();

  A = (A.Transpose() * A).Padding({1, 3, 3});

  PrintTensor(A);
  Tensor<int> B(A.getShape(), -100);

  A = A.Reshape({3, 1, 3}) + (A * A.Reshape({1, 3, 3}));
  PrintTensor(A);

  auto rev_it = A.end();
  auto rev_fin = A.begin();

  while (rev_it != rev_fin) {
    --rev_it;

    std::cout << *rev_it << " ";
  }
  std::cout << std::endl;

  A = A.Reshape(std::vector<int>{static_cast<int>(A.getCapacity()) }).Padding({2 * 3 * 4}).Reshape({2, 3, 4});
  PrintTensor(A);
  A = A.Transpose(0, -1);
  PrintTensor(A);


  A = Tensor({2, 3, 4}, 0);
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
  Tensor C = A.Transpose(0, 1).Reshape({3, 2, 4});
  PrintTensor(C);
  C = C.Transpose(0, 2);
  PrintTensor(C);

  // std::cout << "With multi" << std::endl;
  // Tensor D = A.Transpose(0, 1).Transpose(0, 2);
  // PrintTensor(D);
  
}