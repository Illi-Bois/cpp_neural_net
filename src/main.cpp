#include <iostream>
#include <initializer_list>
#include <vector>
#include <functional>
#include <stdexcept>
#include <numeric>

#include "CPPNeuralNet/Utils/sanity_check.h"
// #include "CPPNeuralNet/Utils/tensor.h"
#include "CPPNeuralNet/Utils/re_tensor.h"

template<typename Der>
class Base {
 public:
  const Der& getRef() const {
    return static_cast<const Der&>(*this);
  }
};



template<typename T1, typename T2>
class SumGlue : public Base<SumGlue<T1, T2>> {
 public:
  const T1& first_;
  const T2& second_;

  SumGlue(const T1& one, const T2& two) 
      : first_(one), second_(two) {}
};

template<typename T>
struct lhs_depth {
  static const int depth = 0;
};
template<typename T1, typename T2>
struct lhs_depth< SumGlue<T1, T2> > {
  static const int depth = 1 + lhs_depth<T1>::depth;
};






template<typename T1, typename T2>
inline SumGlue<T1, T2> operator+(const Base<T1>& first,
                                         const Base<T2>& second) {
  std::cout << "Sum called" << std::endl;
  return SumGlue(first.getRef(), second.getRef());
}


class A : public Base<A> {
 public:
  int a_;

  A(int a) : a_(a) {}

  template<typename T1, typename T2>
  const A& operator=(const SumGlue<T1, T2>& glue);

  template<typename T1, typename T2>
  inline A(const SumGlue<T1, T2>& glue) {
    operator=<T1, T2>(glue); // use copy const
  }

};

template<typename T> 
struct ptr_arr {
  static const int idx = 0;
  
  inline static void get_ptr(const A** arr_p, const T& X) {
    // Store X as A
    arr_p[idx] = reinterpret_cast<const A*>(&X);
  }
};
template<typename T1, typename T2>
struct ptr_arr< SumGlue<T1, T2> > {
  static const int idx = 1 + ptr_arr<T1>::idx;
  
  inline static void get_ptr(const A** arr_p, const SumGlue<T1, T2>& X) {
    //add all to front of arr_p
    ptr_arr<T1>::get_ptr(arr_p, X.first_);
    // Store X as A at the end
    arr_p[idx] = reinterpret_cast<const A*>(&X.second_);
  }
};



template<typename T1, typename T2>
const A& A::operator=(const SumGlue<T1, T2>& glue) {
  // 1 + depth corresponds to total nbumber of matrices
  const int depth = 1 + lhs_depth<SumGlue<T1, T2>>::depth;
  std::cout << depth << std::endl;

  const A* arr_p[depth];

  ptr_arr< SumGlue<T1, T2> >::get_ptr(arr_p, glue);

  a_ = 0;

  for (int i = 0; i < depth; ++i) {
    std::cout << i << " : " << arr_p[i]->a_ << std::endl;
    a_ += arr_p[i]->a_;
  }

  std::cout << "total : " << a_ << std::endl;

  return *this;
}


struct Testing {
  std::vector<int> a;

  const std::vector<int>& get() const {
    return a;
  }
};

// class A : public Base<A> {

// };

// class B : public A {
//  public:
//   int a = 20;
// };


int main() {
  // std::cout << "Hello World!!" << std::endl;

  // TesterClass<int> test(10);
  // std::cout << test.getA() << std::endl;

  // cpp_nn::util::Tensor<int> tens({1, 2, 3});
  // std::cout << tens.getDimension(0) << std::endl;

  A a1(10);
  A a2(11);
  A a3(12);
  A a4(13);

  A b = a1 + a2 + a3 + a4;


  // std::vector<int> arr {1,2,3,4,5};
  // int sum = std::accumulate(arr.begin(), arr.end(), 2, [](int a, int b)->int {return a*b;});

  // std::cout << sum << std::endl;  
  cpp_nn::util::rTensor<int> tens({2,1,3});
  cpp_nn::util::rTensor<int> tensOther( tens );

  cpp_nn::util::rTensor<int> tensOtherOther({1});

  tensOtherOther = tens;
}



