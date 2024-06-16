#include <iostream>
#include <initializer_list>
#include <vector>

#include "CPPNeuralNet/Utils/utils.h"



class A {
 public:
  int* aptr_;
  A(int* aptr) : aptr_(aptr) {
  }
};
class B {
 public:
  int* bptr_;
  B(const A& a) : bptr_(a.aptr_) {
    *a.aptr_ = 12;
  }
};

void test(int* const ptr) {
  *ptr = 100;
}


int main() {
  int num = 10;

  A a(&num);

  B b(a);
  std::cout << *b.bptr_ << std::endl;

  test(&num);

  std::cout << num << std::endl;
}
