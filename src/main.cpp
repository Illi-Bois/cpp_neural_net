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

  std::cout << "PADDING" << std::endl;
  Tensor<float> padded = a.Padding({3, 3});
  PrintTensor(padded);
  Tensor<float> cropped = a.Padding({3, 2});
  PrintTensor(cropped);


  std::cout << "Complex" << std::endl;

  Tensor<int> A = ((Tensor<int>({3, 150, 100}, [val = 0]() mutable {return ++val;})
                * Tensor<int>({150*100}, 2).Reshape({100, 150}))
                + (Tensor<int>({20, 10}, 10).Padding({150, 15}) * Tensor<int>({10, 15}, [val=0]() mutable {return +val;}).Padding({15, 150}) )).Padding({1, 10, 10});
  PrintTensor(A);


  std::cout << "TRANSPOSE ORDER optimize for 2" << std::endl;

  int ax1 = 1;
  int ax2 = 2;

  Tensor<int> tpOri({2, 3, 4}, [val=0]()mutable {return val++;});
  std::vector<int> chunk(tpOri.getOrder(), 1);
  size_t cap = 1;
  ComputeCapacityAndChunkSizes(tpOri.getShape(), chunk, cap);
  std::cout << "ORI shape " << std::endl;
  for (auto i : tpOri.getShape()) {
    std::cout << i << ", ";
  }
  std::cout << std::endl;
  std::cout << "ORI CZ " << std::endl;
  for (auto i : chunk) {
    std::cout << i << ", ";
  }
  std::cout << std::endl;

  Tensor<int> tp = tpOri.Transpose(ax1, ax2);
  std::swap(chunk[ax1], chunk[ax2]);
  std::cout << "tra shape " << std::endl;
  for (auto i : tp.getShape()) {
    std::cout << i << ", ";
  }
  std::cout << std::endl;
  std::cout << "swapped CZ " << std::endl;
  for (auto i : chunk) {
    std::cout << i << ", ";
  }
  std::cout << std::endl;

  // dim1 is the inner- of the two axes in original,
  //  that is chunk1 < chunk2
  int dim1 = tpOri.getDimension(ax2);
  int chunk1 = chunk[ax1];

  int dim2 = tpOri.getDimension(ax1);
  int chunk2 = chunk[ax2];

  int after2 = dim2 * chunk2;

  std::cout << dim1 << " " << chunk1 << std::endl;
  std::cout << dim2 << " " << chunk2 << std::endl;

  int idx = 0;
  auto it = tp.begin();
  while (it != tp.end()) {
    int temp = idx;
    int computed = 0;

    computed += temp - (temp % after2);
    temp %= after2;
    computed += (temp % dim2) * dim1;
    computed += (temp / dim2);
    // computed %= cap;


    // maybe we need to 

    std::cout << idx << "\t" << *it << ",\t" << computed <<std::endl;
    ++it;
    ++idx;
  }
}