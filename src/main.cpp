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
  }
}