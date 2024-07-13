#include <iostream>
#include <initializer_list>
#include <vector>
#include <functional>
#include <stdexcept>
#include <numeric>

#include <cstdlib>

#include "CPPNeuralNet/Utils/tensor.h"
#include "CPPNeuralNet/Utils/utils.h"

template<typename T>
struct Generator {
  std::function<T()> f;

  Generator(std::function<T()> F)
      : f(F) {}
  
  T operator()() {
    return f();
  }
};

int a= 0;
int genVal() {
  return ++a;
}

struct anon {
  double a = 10;
  double getA() {
    return a += 10;
  }
  double operator()() {
    return getA();
  }
} al;
struct randomer {
  double operator()() {
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }
} ra;

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


  int val = 0;
  auto gen1 = [&]() -> float { return ++val; };

  auto generator = [val = 0]() mutable { return ++val; };
  Tensor<float> a({2, 3}, generator);
  Tensor<float> b({2, 3}, 2);
  Tensor<float> c({2, 3}, &genVal);
  Tensor<float> d({2, 3}, al);
  Tensor<float> e({2, 3}, randomer());
  struct HERE {
    float operator()() {
      return 10;
    }
  };
  {
    Tensor<float> f({2, 3}, HERE());  
  }
  Tensor<float> f({2, 3}, HERE{});

  std::function<int()> thisWorks = al;

  PrintTensor(a);
  PrintTensor(b);
  PrintTensor(c);
  PrintTensor(d);
  PrintTensor(e);
  PrintTensor(f);

  std::cout << typeid(decltype(gen1)).name() << std::endl;
  std::cout << val << std::endl;

  std::cout << thisWorks() << std::endl;


  // Generator<float> g([&]() {return "NO";});

  // g();
}