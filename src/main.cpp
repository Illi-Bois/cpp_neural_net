#include <iostream>
#include <initializer_list>
#include <vector>
#include <functional>
#include <stdexcept>

#include "Utils/sanity_check.h"

int main() {
  std::cout << "Hello World" << std::endl;

  TesterClass<int> test(10);
  std::cout << test.getA() << std::endl;

}
