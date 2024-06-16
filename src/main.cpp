#include <iostream>
#include <initializer_list>
#include <vector>
#include <functional>

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
};


class AAAA {
 public:
  AAAA(const std::vector<int>& a) {
    std::cout << a[0];
  }
  AAAA(const std::initializer_list<int>& a) : AAAA(std::vector<int>(a)) {
  }
};

int main() {
  AAAA a({1});
}
