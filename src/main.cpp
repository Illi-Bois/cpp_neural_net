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
    std::cout << "TESTING" << std::endl;
  // Assuming that once summing is correct we can easily test for chaining correctness
  using namespace cpp_nn::util;

  Tensor<int> A({2, 2, 3, 2}, [val = 0]() mutable {return val++;});

  Tensor<int> Afirst = A.SumAxis(1);  // 2 3 2
  Tensor<int> Asecond = Afirst.SumAxis(2); // 2 3
  Tensor<int> Athird = Asecond.SumAxis(0); // 3
  Tensor<int> Afourth = Athird.SumAxis(); // 1

  Tensor<int> Bsecond =  A.SumAxis(1).SumAxis(2);
  
  std::cout << "FOR B" << std::endl << std::endl;
  Tensor<int> Bthird =  A.SumAxis(1).SumAxis(2).SumAxis(0);
  // Tensor<int> Bfourth =  A.SumAxis(1).SumAxis(2).SumAxis(0).SumAxis();

  std::cout << std::endl;
  std::cout << "A" << std::endl;
  PrintTensor(A);
  std::cout << "Afirst" << std::endl;
  PrintTensor(Afirst);
  std::cout << "Asecond" << std::endl;
  PrintTensor(Asecond);
  std::cout << "Athird" << std::endl;
  PrintTensor(Athird);
  // std::cout << "Afourth" << std::endl;
  // PrintTensor(Afourth);


  std::cout << std::endl;
  std::cout << "Bsecond" << std::endl;
  PrintTensor(Bsecond);
  std::cout << "Bthird" << std::endl;
  PrintTensor(Bthird);
  // std::cout << "Bfourth" << std::endl;
  // PrintTensor(Bfourth);
  }


  {
    std::cout << "Mult!!!!!!!!!!!!" << std::endl;
    using namespace cpp_nn::util;

    Tensor<int> A({3, 2}, [val = 0]() mutable {return ++val;});
    PrintTensor(A);
    Tensor<int> B({2, 4}, [val = 0]() mutable {return 2 * ++val;});
    PrintTensor(B);

    Tensor<int> C = A*B;
    std::cout << "Fin" << std::endl;
    PrintTensor(C);

  }

  {
    std::cout << "As vector" << std::endl;
    using namespace cpp_nn::util;

    Tensor<float> A({3}, [val = 0]() mutable {return ++val;});
    PrintTensor(A);

    Tensor<float> B = AsVector(A);
    for (auto i : B.getShape()) {
      std::cout << i << ", ";
    }
    std::cout << std::endl;
    PrintTensor(B);

    Tensor<float> C = TransposeAsVector(A);
    for (auto i : C.getShape()) {
      std::cout << i << ", ";
    }
    std::cout << std::endl;
    PrintTensor(C);

    Tensor<float> D = B * C;
    for (auto i : D.getShape()) {
      std::cout << i << ", ";
    }
    std::cout << std::endl;
    PrintTensor(D);


    Tensor<float> Av = Average<float, Tensor<float>>(D, 0);
    // Tensor<float> Av = D.SumAxis(0) / 2.0f;
    PrintTensor(Av);
  }


  {
    std::cout << "relu exploreation" << std::endl;

    using namespace cpp_nn::util;

    Tensor<float> lastIn = AsTensor<float>({
      2.0f,
      0.0f,
      -1.0f,

      1.0f,
      1.0f,
      -2.0f
    }).Reshape({2, 3});
    Tensor<float> gradient = AsTensor<float>({
      1,
      2,
      3,

      4,
      5,
      6
    }).Reshape({2, 3});

    auto func = [inpIt = lastIn.begin(), gradIt = gradient.begin()]
                () mutable {
      float res = (*inpIt > 0) ? *gradIt : 0;
      ++inpIt;
      ++gradIt;
      return res;
    };

    Tensor<float> A(lastIn.getShape(), 
                    [inpIt = lastIn.begin(), gradIt = gradient.begin()] () mutable {
                      float res = (*inpIt > 0) ? *gradIt : 0;
                      ++inpIt;
                      ++gradIt;
                      return res;
                    });

    PrintTensor(A);
  }
}