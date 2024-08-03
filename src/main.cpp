#include <iostream>
#include <initializer_list>
#include <vector>
#include <functional>
#include <stdexcept>
#include <numeric>

#include <cstdlib>

#include "CPPNeuralNet/Utils/tensor.h"
#include "CPPNeuralNet/Utils/utils.h"

template<typename Der>
class Base {
 public:
  Der& getRef() {
    return static_cast<Der&>(*this);
  }
  void call() {
    getRef().Der::call();
  }

  void otherCall() {
    call();
  }
};

class DefBase : public Base<DefBase> {
 public:
  void call() {
    std::cout << "Def" << std::endl;
  }
};

class Act : public Base<Act>, 
            private DefBase {
              
 public:
  typedef Base<Act> Parent;
  void call() {
    DefBase::call();
    std::cout << "and then" << std::endl;
  }

    using Base<Act>::otherCall;

  
};

int main() {
  using namespace cpp_nn::util;

  {
    Tensor<int> A({2, 3, 4}, [val=0]()mutable {return val++;});
    Tensor<int> B = A.Transpose().Transpose();


    PrintTensor(A);
    PrintTensor(B);

  }
  {
    DefBase A;
    Base<DefBase> Ac = A;

    Act B;
    Base<Act> Bc = B;

    A.call();
    Ac.call();

    A.otherCall();
    Ac.otherCall();

    B.call();
    Bc.call();

    B.otherCall();
    Bc.otherCall();
  } {
    Tensor<float> A = AsTensor<float>({4, 5, 5,
                                       2, 3, 2,
                                       4, 4, 4}).Reshape({3, 3});
    Tensor<float> B = A * 0.5f;
    Tensor<float> C = 0.5f * A;

    PrintTensor(A);
    PrintTensor(B);
    PrintTensor(C);
  }

  {
    std::cout << "SUB" << std::endl;

    Tensor<int> A({2, 3, 4}, [val=0]()mutable {return val++;});
    Tensor<int> B({2, 3, 4}, [val=0]()mutable {return val++;});

    PrintTensor(A);
    PrintTensor(B);

    Tensor<int> C = A - B;
    PrintTensor(C);
  }

  {
    std::cout << "SUBSUB" << std::endl;
    Tensor<int> A({4, 3, 2}, [val=0]()mutable {return val++;});
    Tensor<int> B = AxisSummationOperation<int, Tensor<int>>({A, 2});

    std::cout << "A" << std::endl;
    PrintTensor(A);
    std::cout << "B" << std::endl;
    PrintTensor(B);
  }
}