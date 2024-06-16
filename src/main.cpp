#include <iostream>
#include <initializer_list>
#include <vector>

#include "CPPNeuralNet/Utils/utils.h"



class A {
 public:
  int* aptr_;
  int& b;
  A(int* aptr, int& c) : aptr_(aptr), b(c) {
  }

  void change() const {
    this->b = 13;
    *this->aptr_ = 13;
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
  int num2 = 10;

  A a(&num, num2);

  const A c(&num, num2);
  c.b = 11;
  // c.aptr_ = &num2;

  c.change();


  std::cout << num2 << std::endl;
}
