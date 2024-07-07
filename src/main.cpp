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
  cpp_nn::util::rTensor one({2,3,4}, 1);
  cpp_nn::util::rTensor two({2,3,4}, 2);

  cpp_nn::util::rTensor three = one + two;
  std::cout << "Summed" << std::endl;
  for (int k = 0; k < 2; ++k) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 4; ++j) {
        std::cout << three.getElement({k, i, j}) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }


  std::cout << "PRODT " << std::endl;

  cpp_nn::util::rTensor first({2,3,3}, 0);
  first.getElement({0, 0, 0}) = 1;
  first.getElement({0, 1, 1}) = 1;
  first.getElement({0, 2, 2}) = 1;

  first.getElement({1, 0, 0}) = 2;
  first.getElement({1, 1, 1}) = 2;
  first.getElement({1, 2, 2}) = 2;
  cpp_nn::util::rTensor second({2,3,4}, 2);

  cpp_nn::util::rTensor third = first * second;

  auto shape = third.getShape();
  for (auto i : shape) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  for (int k = 0; k < third.getDimension(0); ++k) {
    for (int i = 0; i < third.getDimension(1); ++i) {
      for (int j = 0; j < third.getDimension(2); ++j) {
        std::cout << third.getElement({k, i, j}) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }


  cpp_nn::util::rTensor<int> fin = three + third + third + three;
  std::cout << std::endl;
  for (int k = 0; k < fin.getDimension(0); ++k) {
    for (int i = 0; i < fin.getDimension(1); ++i) {
      for (int j = 0; j < fin.getDimension(2); ++j) {
        std::cout << fin.getElement({k, i, j}) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }



  // if( equal(vector1.begin(), vector1.end(), vector2.begin()) )
    // DoSomething();

  std::vector<int> arr1 = {3, 2, 4};
  std::vector<int> arr2 = {0, 0, 0};


  
  for (int i = 0; i < 30; ++i) {
    bool res = cpp_nn::util::IncrementIndicesByShape(arr1.begin(), arr1.end() - 1,
                                          arr2.begin(), arr2.end() - 1 );
    for (int i : arr2) {
      std::cout << i << ", ";
    }
    std::cout << " with " << res << std::endl;
  }



  if (std::equal(arr1.begin(), arr1.end() - 2, arr2.begin())) {
    std::cout << "EQ" << std::endl;
  }

  if (true) {
    cpp_nn::util::rTensor<int> aTen({3, 4}, 0);
    /*
        1 2 0 0
        0 1 0 0
        0 0 0 1
    */
    aTen.getElement({0, 0}) = 1;
    aTen.getElement({0, 1}) = 2;

    aTen.getElement({1, 1}) = 1;
    
    aTen.getElement({2, 3}) = 1;

    cpp_nn::util::rTensor<int> bTen({4, 4}, 0);
    /*
      2 1 0 1
      0 0 0 0
      1 0 0 1
      0 1 0 1
    */
    bTen.getElement({0, 0}) = 2;
    bTen.getElement({0, 1}) = 1;
    bTen.getElement({0, 3}) = 1;

    bTen.getElement({2, 0}) = 1;
    bTen.getElement({2, 3}) = 1;
    
    bTen.getElement({3, 1}) = 1;
    bTen.getElement({3, 3}) = 1;

    cpp_nn::util::rTensor res = aTen * bTen;
    std::cout << "Product" << std::endl;
    for (int r = 0; r < res.getDimension(0); ++r) { 
      for (int c = 0; c < res.getDimension(1); ++c) {
        std::cout << res.getElement({r, c});
      }
      std::cout << std::endl;
    }


    std::cout << "Reshape No Call" << std::endl;
    res.Reshape({3, 2, 2});
    for (auto d : res.getShape()) {
      std::cout << d << ", ";
    }
    std::cout << std::endl;
    for (int r = 0; r < res.getDimension(0); ++r) { 
      for (int c = 0; c < res.getDimension(1); ++c) {
        std::cout << res.getElement({r, c});
      }
      std::cout << std::endl;
    }

    std::cout << "Reshape called" << std::endl;
    cpp_nn::util::rTensor<int> exRes = res.Reshape({4, 3, 1});
    for (int r = 0; r < exRes.getDimension(0); ++r) { 
      for (int c = 0; c < exRes.getDimension(1); ++c) {
        for (int k = 0; k < exRes.getDimension(2); ++k) {
          std::cout << exRes.getElement({r, c, k});
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }

    std::cout << "Reshape Chained" << std::endl;
    cpp_nn::util::rTensor<int> exRes2 = (res.Reshape({3, 2, 2}) + cpp_nn::util::rTensor<int>({3, 2, 2}, 2)).Reshape({4, 3, 1}).Reshape({2, 3, 2});
    for (int r = 0; r < exRes2.getDimension(0); ++r) { 
      for (int c = 0; c < exRes2.getDimension(1); ++c) {
        for (int k = 0; k < exRes2.getDimension(2); ++k) {
          std::cout << exRes2.getElement({r, c, k});
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }

    // some reshape exploration
    {
      cpp_nn::util::rTensor<int> big({3, 4, 6});
      std::vector<int> idx(big.getOrder(), 0);
      int i = 0;
      do {
        big.getElement(idx) = ++i;
      } while (cpp_nn::util::IncrementIndicesByShape(big.getShape().begin(), big.getShape().end(),
                                                      idx.begin(), idx.end()));
      

      cpp_nn::util::PrintTensor(big);

      // MATRIX CUTTING
      /*
        Cut at initial rows, 
        Tranpose,
        cut at columns,
        Tranpose
      */
     /** Better to reshape {dim, R<-r C<-c} then switch r, C */
      cpp_nn::util::rTensor<int> small = big.Reshape({3, 2, 2, 2, 3})
                                            .Transpose(-2, -3);
      // cpp_nn::util::rTensor<int> small = big.Reshape({3, 2, 2, 6})        // 3 2 2 6
      //                                       .Transpose()                  // 3 2 6 2
      //                                       .Reshape({3, 2, 2, 3, 2})     // 3 2 2 3 2
      //                                       .Transpose();                 // 3 2 2 2 3
      cpp_nn::util::PrintTensor(small);

      // Recombiner
      /*
        Tranpose, 
        Combine col,
        Tranpose,
        Combin
       */
      cpp_nn::util::rTensor<int> back = small                       // 3 2 2 2 3
                                             .Transpose()           // 3 2 2 3 2
                                             .Reshape({3, 2, 6, 2}) // 3 2 6 2
                                             .Transpose()           // 3 2 2 6
                                             .Reshape({3, 4, 6}) // 3 4 6
                                            ;
      cpp_nn::util::PrintTensor(back);


      // The issue on previously found on CutMatrix not running propely may be issue on needed data being destroyed thing is being built?
      // ^ no because we are always calling on constructor

      // TestGround
      if (true) {
        std::cout << "Test Ground" << std::endl;
        cpp_nn::util::rTensor<int> big({1, 2*4, 3*5});
        std::vector<int> idx(big.getOrder(), 0);
        int i = 0;
        do {
          big.getElement(idx) = ++i;
        } while (cpp_nn::util::IncrementIndicesByShape(big.getShape().begin(), big.getShape().end(),
                                                        idx.begin(), idx.end()));   

        std::cout << "Big Made" << std::endl;
        cpp_nn::util::PrintTensor(big);
        std::cout << "Big Printed" << std::endl;


        cpp_nn::util::rTensor<int> cut(cpp_nn::util::CutMatrix(big, 2, 3));
        cpp_nn::util::PrintTensor(cut);
        std::cout << "cut Printed" << std::endl;



        cpp_nn::util::rTensor<int> padded = cut.Padding({1, 4, 5, 3, 3});
        cpp_nn::util::PrintTensor(padded);        
        std::cout << "padded Printed" << std::endl;


        cpp_nn::util::rTensor<int> merg(cpp_nn::util::MergeCutMatrix(cut));
        cpp_nn::util::PrintTensor(merg);
        std::cout << "merg Printed" << std::endl;
      

        cpp_nn::util::rTensor<int> res(merg.Reshape({1, 8, 15}).Reshape({1, 2, 3, 4, 5}));
        cpp_nn::util::PrintTensor(res);
        std::cout << "res Printed" << std::endl;
      
        

        // The issue seems, as soon as
        /*
          when i call tranpose on Reshape, reshape is a temporary rvalue
          so I cannot, or should not make temp point to the refrence
          when I do, the data gets overriden and so can no longer point to correct term

          This issue would not be present in normal usage because each rvalue will still be in scope to be processed?

          POTENTIAL SOLUTION
          make return of cut be a tensor

          make a CapsulatorOperation
            which just holds the temporary internally so it can be passed down
        */

      //   if (true) {
      //     std::cout << "Doing all in place" << std::endl;

      //     std::cout << "big addres < " << &big << std::endl;

      //     int R = 2;
      //     int C = 3;

      //     int r = big.getDimension(-2) / R;
      //     int c = big.getDimension(-1) / C;

      //     std::vector<int> newShape(big.getShape());
      //     newShape[newShape.size() - 2] = r;
      //     newShape[newShape.size() - 1] = R;
      //     newShape.push_back(c);
      //     newShape.push_back(C);
      //     cpp_nn::util::TransposeOperation<int, cpp_nn::util::ReshapeOperation<int, cpp_nn::util::rTensor<int>>> temp =  big.Reshape(newShape).Transpose(-2, -3);
      //     cpp_nn::util::rTensor<int> cut = temp;

      //     cpp_nn::util::PrintTensor(cut);
      //     // This works, which further suggests this is all a scope issue?
      //     //  With destructor it seems indeed that descrutor is called before tensor is set, and so failed?
      //   }
        



      //   if (true) {
      //     std::cout << "Doing all in place In one line" << std::endl;

      //     std::cout << "big addres < " << &big << std::endl;

      //     int R = 2;
      //     int C = 3;

      //     int r = big.getDimension(-2) / R;
      //     int c = big.getDimension(-1) / C;

      //     std::vector<int> newShape(big.getShape());
      //     newShape[newShape.size() - 2] = r;
      //     newShape[newShape.size() - 1] = R;
      //     newShape.push_back(c);
      //     newShape.push_back(C);
      //     cpp_nn::util::rTensor<int> cut =  big.Reshape(newShape).Transpose(-2, -3);

      //     cpp_nn::util::PrintTensor(cut);
      //     // Here dest for reshape only occurs after tensor is set.
      //     // This further suggests that we should not let intermediates be exposed

      //   }
      // }

      // if (true) {
      //   std::cout << "Doing all in place but with a lot of intermedaite things happening" << std::endl;

      //   std::cout << "big addres < " << &big << std::endl;

      //   int R = 2;
      //   int C = 3;

      //   int r = big.getDimension(-2) / R;
      //   int c = big.getDimension(-1) / C;

      //   std::vector<int> newShape(big.getShape());
      //   newShape[newShape.size() - 2] = r;
      //   newShape[newShape.size() - 1] = R;
      //   newShape.push_back(c);
      //   newShape.push_back(C);
      //   cpp_nn::util::TransposeOperation<int, cpp_nn::util::ReshapeOperation<int, cpp_nn::util::rTensor<int>>> temp =  big.Reshape(newShape).Transpose(-2, -3);
      //   /** A lot oe interm */
      //   cpp_nn::util::rTensor<int> useless = big.Transpose();
      //   cpp_nn::util::rTensor<int> useless2 = useless.Transpose();
      //   cpp_nn::util::rTensor<int> useless3 = useless + useless2;
      //   cpp_nn::util::rTensor<int> useless4 = useless3.Reshape({2*3*4*5});


      //   cpp_nn::util::rTensor<int> cut = temp;

      //   cpp_nn::util::PrintTensor(cut);
      //   // AS EXPECTED FAILS AS THE RESHAPE WAS OVERWRITTEN
      // }

      // if (true) {
      //   std::cout << "When function called " << std::endl;

      //   std::cout << "big addres < " << &big << std::endl;

      //   cpp_nn::util::rTensor<int> cut = cpp_nn::util::Test(big, 2, 3);
      //   std::cout << "cut Made" << std::endl;
      //   cpp_nn::util::PrintTensor(cut);
      //   std::cout << "cut Printed" << std::endl;
        

        if(true) {
          // Tranpose and Multi Tranpose
          std::cout << "Test Ground" << std::endl;
          cpp_nn::util::rTensor<int> ori({1, 2, 3, 4});
          std::vector<int> idx(ori.getOrder(), 0);
          int i = 0;
          do {
            ori.getElement(idx) = ++i;
          } while (cpp_nn::util::IncrementIndicesByShape(ori.getShape().begin(), ori.getShape().end(),
                                                          idx.begin(), idx.end()));   

          std::cout << "printing ori" << std::endl;
          cpp_nn::util::PrintTensor(ori);

          std::cout << "Normally Tranposed" << std::endl;
          cpp_nn::util::rTensor<int> norm = ori.Transpose(2, 0).Transpose(1, 3).Transpose(0, 1).Transpose(0, 3);
          cpp_nn::util::PrintTensor(norm);
          std::cout << " shape is  " << std::endl;
          for (int d : norm.getShape()) {
            std::cout << d << ", ";
          }
          std::cout << std::endl;

          std::cout << "Multi Tranposed" << std::endl;
          cpp_nn::util::rTensor<int> mult = ori.Transpose(2, 0).MultiTranpose(1, 3).Transpose(0, 1).Transpose(0, 3);
          cpp_nn::util::PrintTensor(mult);
          std::cout << " shape is  " << std::endl;
          for (int d : mult.getShape()) {
            std::cout << d << ", ";
          }
          std::cout << std::endl;

        }
      }
    }
  }
  
}




