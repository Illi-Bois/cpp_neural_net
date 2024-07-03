#include <iostream>
#include <initializer_list>
#include <vector>
#include <functional>
#include <stdexcept>
#include <numeric>

#include "CPPNeuralNet/Utils/sanity_check.h"
#include "CPPNeuralNet/Utils/re_tensor.h"
#include "CPPNeuralNet/Utils/utils.h"


namespace VarTempTemp {
/*
  Small lightweight example to figure it out
*/

template<typename T, typename Derived>
class Base {
 public:
  const Derived& getRef() const {
    return static_cast<const Derived&>(*this);
  }

  T getThing() {
    return T();
  }
};

template<typename T>
class FirstDer : public Base<T, FirstDer<T>> {
 public:
  FirstDer(int a) {
  }

  // Generic type... accepts any variadic
  template<typename... Other, template<typename...>typename Der>
  void test(Base<T, Der<T, Other...>> a);
};


template<typename T, typename Other>
class SecondDer : public Base<T, SecondDer<T, Other>> {
 public:
  SecondDer(int a, int b) {
  }
};

template<typename T, typename Other1, typename Other2>
class ThirdDer : public Base<T, ThirdDer<T, Other1, Other2>> {
 public:
  ThirdDer(int a, int b) {
  }
};



  // Generic type... accepts any variadic
template<typename T>
template<typename... Other, template<typename...>typename Der>
void FirstDer<T>::test(Base<T, Der<T, Other...>> a) {
  std::cout << "First" << std::endl;
  std::cout << typeid(a).name() << std::endl;
}

// // want to specify above 
// template<>
// template<typename T>
// template<typename... Other>
// void FirstDer<T>::test(Base<T, SecondDer<T, Other...>> a) {
//   std::cout << "First" << std::endl;
//   std::cout << typeid(a).name() << std::endl;
// }


// template<typename T>
// template<>
// template<typename Other>
// void FirstDer<T>::test(Base<T, SecondDer<T, Other>> a) {
//   std::cout << "Over" << std::endl;
//   std::cout << typeid(a).name() << std::endl;
// }

// template<typename T, typename Other>
// void test(Base<T, SecondDer<T, Other>> a) {
//   std::cout << "Over" << std::endl;
//   std::cout << typeid(a).name() << std::endl;
// }


} // Variadic Template Template


int main() {
  // cpp_nn::util::rTensor one({2,3,4}, 1);
  // cpp_nn::util::rTensor two({2,3,4}, 2);

  // cpp_nn::util::rTensor three = one + two;
  // std::cout << "Summed" << std::endl;
  // for (int k = 0; k < 2; ++k) {
  //   for (int i = 0; i < 3; ++i) {
  //     for (int j = 0; j < 4; ++j) {
  //       std::cout << three.getElement({k, i, j}) << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  //   std::cout << std::endl;
  // }


  // std::cout << "PRODT " << std::endl;

  // cpp_nn::util::rTensor first({2,3,3}, 0);
  // first.getElement({0, 0, 0}) = 1;
  // first.getElement({0, 1, 1}) = 1;
  // first.getElement({0, 2, 2}) = 1;

  // first.getElement({1, 0, 0}) = 2;
  // first.getElement({1, 1, 1}) = 2;
  // first.getElement({1, 2, 2}) = 2;
  // cpp_nn::util::rTensor second({2,3,4}, 2);

  // cpp_nn::util::rTensor third = first * second;

  // auto shape = third.getShape();
  // for (auto i : shape) {
  //   std::cout << i << " ";
  // }
  // std::cout << std::endl;
  // for (int k = 0; k < third.getDimension(0); ++k) {
  //   for (int i = 0; i < third.getDimension(1); ++i) {
  //     for (int j = 0; j < third.getDimension(2); ++j) {
  //       std::cout << third.getElement({k, i, j}) << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  //   std::cout << std::endl;
  // }


  // cpp_nn::util::rTensor<int> fin = three + third + third + three;
  // std::cout << std::endl;
  // for (int k = 0; k < fin.getDimension(0); ++k) {
  //   for (int i = 0; i < fin.getDimension(1); ++i) {
  //     for (int j = 0; j < fin.getDimension(2); ++j) {
  //       std::cout << fin.getElement({k, i, j}) << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  //   std::cout << std::endl;
  // }



  // // if( equal(vector1.begin(), vector1.end(), vector2.begin()) )
  //   // DoSomething();

  // std::vector<int> arr1 = {3, 2, 4};
  // std::vector<int> arr2 = {0, 0, 0};


  
  // for (int i = 0; i < 30; ++i) {
  //   bool res = cpp_nn::util::IncrementIndicesByShape(arr1.begin(), arr1.end() - 1,
  //                                         arr2.begin(), arr2.end() - 1 );
  //   for (int i : arr2) {
  //     std::cout << i << ", ";
  //   }
  //   std::cout << " with " << res << std::endl;
  // }



  // if (std::equal(arr1.begin(), arr1.end() - 2, arr2.begin())) {
  //   std::cout << "EQ" << std::endl;
  // }

  // if (true) {
  //   cpp_nn::util::rTensor<int> aTen({3, 4}, 0);
  //   /*
  //       1 2 0 0
  //       0 1 0 0
  //       0 0 0 1
  //   */
  //   aTen.getElement({0, 0}) = 1;
  //   aTen.getElement({0, 1}) = 2;

  //   aTen.getElement({1, 1}) = 1;
    
  //   aTen.getElement({2, 3}) = 1;

  //   cpp_nn::util::rTensor<int> bTen({4, 4}, 0);
  //   /*
  //     2 1 0 1
  //     0 0 0 0
  //     1 0 0 1
  //     0 1 0 1
  //   */
  //   bTen.getElement({0, 0}) = 2;
  //   bTen.getElement({0, 1}) = 1;
  //   bTen.getElement({0, 3}) = 1;

  //   bTen.getElement({2, 0}) = 1;
  //   bTen.getElement({2, 3}) = 1;
    
  //   bTen.getElement({3, 1}) = 1;
  //   bTen.getElement({3, 3}) = 1;

  //   cpp_nn::util::rTensor res = aTen * bTen;
  //   for (int r = 0; r < res.getDimension(0); ++r) { 
  //     for (int c = 0; c < res.getDimension(1); ++c) {
  //       std::cout << res.getElement({r, c});
  //     }
  //     std::cout << std::endl;
  //   }


  //   res.Reshape({3, 2, 2});
  //   for (int r = 0; r < res.getDimension(0); ++r) { 
  //     for (int c = 0; c < res.getDimension(1); ++c) {
  //       for (int k = 0; k < res.getDimension(2); ++k) {
  //         std::cout << res.getElement({r, c, k});
  //       }
  //       std::cout << std::endl;
  //     }
  //     std::cout << std::endl;
  //   }

  //   cpp_nn::util::rTensor<int> exRes = res.ExternalReshape({4, 3, 1});
  //   for (int r = 0; r < exRes.getDimension(0); ++r) { 
  //     for (int c = 0; c < exRes.getDimension(1); ++c) {
  //       for (int k = 0; k < exRes.getDimension(2); ++k) {
  //         std::cout << exRes.getElement({r, c, k});
  //       }
  //       std::cout << std::endl;
  //     }
  //     std::cout << std::endl;
  //   }


  //   cpp_nn::util::rTensor<int> exRes2 = (res + cpp_nn::util::rTensor<int>({3, 2, 2}, 2)).ExternalReshape({4, 3, 1}).ExternalReshape({2, 3, 2});
  //   for (int r = 0; r < exRes2.getDimension(0); ++r) { 
  //     for (int c = 0; c < exRes2.getDimension(1); ++c) {
  //       for (int k = 0; k < exRes2.getDimension(2); ++k) {
  //         std::cout << exRes2.getElement({r, c, k});
  //       }
  //       std::cout << std::endl;
  //     }
  //     std::cout << std::endl;
  //   }
  // }


  // Some Temp Temp exp.
  VarTempTemp::FirstDer<int> f(10);
  VarTempTemp::SecondDer<int, double> s(10, 11);
  VarTempTemp::ThirdDer<int, double, int> t(10, 11);

  f.test(f);
  f.test(s);
  f.test(t);
}




