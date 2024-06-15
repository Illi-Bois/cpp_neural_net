#include <iostream>
#include <initializer_list>
#include <vector>

#include "CPPNeuralNet/Utils/utils.h"


// TESTZONE, modify as you want to test out ideas.
template<typename T=int>
class A {
 public:
  A mult(A other);

  template<class Derived>
  Derived genMult(Derived d);
};


template<typename T=int>
class B : public A<T> {
};

template<typename T>
A<T> A<T>::mult(A<T> other) {
  std::cout << "okay" << std::endl;
}

template<typename T>
template<class D>
D A<T>::genMult(D d) {
  static_assert(std::is_base_of<A, D>::value, "Error: not a A");
  std::cout << "weird" << std::endl;
  return d;
}


void test(int arr...) {
  std::cout << arr << std::endl;
  std::cout << *(&arr + sizeof(int)*2) << std::endl;
}

// void a1(std::initializer_list<int> arr) {
//   std::cout << "This is init list" << std::endl;
//   std::cout << arr[0] << std::endl;
// }
void a1(std::vector<int> arr) {
  std::cout << "This is vec" << std::endl;
  std::cout << arr.size() << std::endl;
}


template<typename T, int kVal>
class AA {
 public:
  void p();
};

template<typename T, int kVal>
void AA<T, kVal>::p() {
  int a = kVal;
  std::cout << kVal << std::endl;
}


int main() {
  std::cout << "Hello World" << std::endl;
  // Model model;

  // model.addLayer(new Layer(255, 25))
  //      .addLayer(new Sogmoid())
  //      .addLayer(new Layer(255, 25))
  //      .addLayer(new Sogmoid())
  //      .addLayer(new Layer(255, 25));

  // A<> a;
  // B<> b;

  // A<> c = b.mult(b);
  // A<> d = a.mult(b);
  // B<> e = a.genMult(b);
  // A<> f = a.genMult(a);

  // int k = 0;
  // int g = a.genMult(k);


  // test(1, 2, 3);

  // a1({1,2,3});


  // AA<int, 3> al;
  // al.p();

  // for (int i = 0, j = 0; i < 10; ++i, ++j) {
  //   std::cout << i << std::endl;
  // }

  std::vector<int> ord{0,3,2,1};
  std::vector<int> val{11, 12, 13, 14};

  cpp_nn::util::reorder(ord.begin(), ord.end(), val.begin());

  for(auto a : val) {
    std::cout << a << std::endl;
  }
}
