#include <iostream>
#include <initializer_list>
#include <vector>
#include <functional>
#include <stdexcept>
#include <numeric>

#include <cstdlib>

#include "CPPNeuralNet/Utils/tensor.h"
#include "CPPNeuralNet/Utils/utils.h"

#include "CPPNeuralNet/Layers/relu_layer.h"
#include "CPPNeuralNet/Layers/sigmoid_layer.h"
#include "CPPNeuralNet/Layers/linear_layer.h"



int main() {
  using namespace cpp_nn::util;
  using namespace cpp_nn;

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

  ReLU relu;

  Tensor<float> OK({2, 3, 4}, [val = 1]()mutable {
                                val *= -2;
                                return val;  
                              });

  Tensor<float> grad({2, 3, 4}, [val = 1]()mutable {
                                val *= 3;
                                return val;  
                              });
  PrintTensor(OK);
  Tensor<float> DK = relu.forward(OK);
  PrintTensor(DK);

  PrintTensor(grad);
  Tensor<float> PK = relu.backward(grad);
  PrintTensor(PK);

  Sigmoid sig;
  Tensor<float> sK = sig.forward(PK);
  std::cout << "SK" << std::endl;
  PrintTensor(sK);


  Tensor<float> one({2, 3, 4}, 1);
  Tensor<float> sKg = sig.backward(one);
  std::cout << "baK" << std::endl;
  PrintTensor(sKg);

  LinearLayer lin(20, 15);

  Tensor<float> vec({20}, [val=0]()mutable {
                            val++;
                            return (val > 5) ? 0 : 2;
                          });

  std::cout << "VEC" << std::endl;
  PrintTensor(vec);
  // Tensor<float> vecVec = AsVector(vec);
  // std::cout << "VECVEC" << std::endl;
  // PrintTensor(vecVec);
  // Tensor<float> vecVecVec = AsVector(vecVec);
  // std::cout << "vecVecVec" << std::endl;
  // PrintTensor(vecVecVec);

  // Tensor<float> un = CollapseEnd(vec);
  // Tensor<float> unun = CollapseEnd(vecVec);
  // Tensor<float> ununun = CollapseEnd(vecVecVec);

  // std::cout << "un" << std::endl;
  // PrintTensor(un);
  // std::cout << "unun" << std::endl;
  // PrintTensor(unun);
  // std::cout << "ununun" << std::endl;
  // PrintTensor(ununun);

  Tensor<float> out = lin.forward(vec);
  PrintTensor(out);

  Tensor<float> grad2 = lin.backward(out);
  PrintTensor(grad2);



}