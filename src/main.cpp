#include <iostream>
#include <initializer_list>
#include <vector>
#include <functional>
#include <stdexcept>

#include "sanity_check.h"

int main() {
  std::cout << "Hello World" << std::endl;

  TesterClass test(10);
  std::cout << test.getA() << std::endl;

}
