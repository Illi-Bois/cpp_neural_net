#include <iostream>
#include <initializer_list>
#include <vector>
#include <functional>
#include <stdexcept>
#include <numeric>

#include "CPPNeuralNet/Utils/sanity_check.h"
// #include "CPPNeuralNet/Utils/tensor.h"
#include "CPPNeuralNet/Utils/re_tensor.h"


#define quote(x) #x


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


// TODO!!
/* 
 We need to test out behaviour of cons-cast magix

 let A have const-cast getter

 Let B have non constant A

 Let b be constant instance of B

 when b internally modifies A, will it be unable to??


 CONCLUSION:
 i dont think we have to worry about that, as const B will not be able to
  call any non-const members
  and in const member methods, its members cannot be modified as they are themselves const
*/

namespace ConstCastMagic {
class A {
 public:
  int a;

  const int& get() const {
    return a;
  }
  int& get() {
    return const_cast<int&>(const_cast<const A*>(this)->get());
  }

};

class B {
  A a_;

  B() {
    a_.get() = 12;
  }

  void change() {
    a_ = A();
  }
};
}


// namespace VariadicTemplateMagic {
// template<typename... Args>
// struct A;

// template<typename First, typename... Args>
// struct A<First, Args...> {

// }


// }


namespace tempMagic {


void doIt() {
  std::cout << std::endl;
}

template<typename T, typename... U, int COUNT = 0>
void doIt(T a, U... b) {
  doIt(b...);

  std::cout << a << " " << COUNT << ", ";
}


int sum(int a) {
  return a;
}
int sum(int a, int b...) {
  return a + sum(b);
}


std::vector<int> asVec(int a...) {
  return std::vector<int>{a};
}

// DOESNT WORK....
template<int OFFSET = 1>
int magick(int a) {
  std::cout << "TERMINAL " << OFFSET << " : " << a << std::endl;
  return OFFSET * a;
}

template<int OFFSET = 1>
int magick(int a, int b...) {
  std::cout << "MID " << OFFSET << " : " << a << std::endl;
  return magick<OFFSET>(a) + magick<OFFSET + 1>(b);
}


template<int Size>
struct A {
  int arr[Size];

  int get(int i) {
    std::cout << arr[i] << std::endl;
  }
  template<typename... T>
  int get(int i, T... t) {
    std::cout << arr[i] << " ";
    (*this).get(t...);
    // this->operator()(t...);
  }
};

}



namespace PrivatePublic {

class A {
 private:
  struct B {
    int b;
  };

  int a_;
 public:
  A(int a) {
    a_ = a;
  }

  A(B b) {
    // std::cout << "Making from " << quote(b) << std::endl;
    a_ = b.b;
  }

  int get() {
    return a_;
  }

  B getB() {
    return B{2 * a_};
  }
};


A makeA(A a) {
  return a;
}

}


namespace Anon {

namespace {

struct B {
  int b;
};

}

class A {
 private:
  int a_;
 public:
  A(int a) {
    a_ = a;
  }

  A(B b) {
    a_ = b.b;
  }

  B getB() {
    return B{2 * a_};
  }
};

}

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

  tensOtherOther = std::move(tens);

  const auto& ref = tensOtherOther;


  // std::cout << tens.getDimension(-2) << std::endl;
  // std::cout << tens.getDimension(10) << std::endl;
  // std::cout << tens.getOrder() << std::endl;

  int arr[100];

  int* p = arr + 10;

  arr[8] = 12;

  for (int i = 0; i < 100; ++i) {
    arr[i] = i;
  }

  std::cout << p[-2] << std::endl;
  std::cout << -2[p] << std::endl;
  std::cout << (-2)[p] << std::endl;




  std::cout << "ORder is " << tensOther.getOrder() << std::endl; 
  std::cout << "CAp is " << tensOther.getCapacity() << std::endl; 
  // tens.getElement({0, 0, 0}) = 12;

  std::cout << tensOther.getElement({1, 0, 1}) << std::endl;


  std::cout << ref.getElement({0,0,0}) << std::endl;
  tensOtherOther.getElement({0, 0, 0}) = 12;
  std::cout << ref.getElement({0,0,0}) << std::endl;

  // ConstCastMagic::A a;
  // a.get() = 11;

  tempMagic::doIt<int, int, int>(10, 11, 12);


  try {
    std::cout << "Doing osme things" << std::endl;
    throw 1;
    std::cout << "Doing osme things" << std::endl;
  } catch (int a) {
    std::cout << a << std::endl;
  }

  std::vector<int> vec = tempMagic::asVec(111,112,113);
  for (auto i : vec) {
    std::cout << i << " ";
  }

  std::cout << std::endl << std::endl;
  std::cout << tempMagic::magick(1, 2, 3) << std::endl;


  tempMagic::A<3> a = {1,2,3};

  a.get(0,0,1,0,2);


  PrivatePublic::A pa(10);
  PrivatePublic::A pb(pa.getB());

  PrivatePublic::A pc = pb.getB();
  auto pd = pc.getB(); // seems capturable but compiler cannot view anything from it?
  // treating as A works
  PrivatePublic::A pe = PrivatePublic::makeA(pc.getB());

  std::cout << pa.get() << std::endl;
  std::cout << pb.get() << std::endl;
  std::cout << pc.get() << std::endl;
  std::cout << typeid(pa).name() << std::endl;
  std::cout << typeid(pd).name() << std::endl;
  std::cout << pe.get() << std::endl;

  Anon::A aa(10);
  Anon::A ab(aa.getB());
  Anon::A ac = ab.getB();
  auto ad = ac.getB(); // seems capturable but compiler cannot view anything from it?
  // unnamed namespace will make it local and private
  // Anon::A::B bb = ac.getB();
  std::cout << typeid(ad).name() << std::endl;
  std::cout << ad.b << std::endl;

}



