#include <iostream>
#include <initializer_list>
#include <vector>
#include <functional>
#include <stdexcept>
#include <numeric>

#include <cstdlib>

#include "CPPNeuralNet/Utils/tensor.h"
#include "CPPNeuralNet/Utils/utils.h"

template<typename T>
struct Generator {
  std::function<T()> f;

  Generator(std::function<T()> F)
      : f(F) {}
  
  T operator()() {
    return f();
  }
};

int a= 0;
int genVal() {
  return ++a;
}

struct anon {
  double a = 10;
  double getA() {
    return a += 10;
  }
  double operator()() {
    return getA();
  }
} al;
struct randomer {
  double operator()() {
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }
} ra;

int TranposedAddressToOriginalAddress(int tranposed_address,
                                      int dim1, int chunk1,
                                      int dim2, int chunk2) {
  // take the previous found Origina->Tranpose in main and work it backwards.
  // essentially doing the same computation but backwards, meaning
  //  mod to pick by what we multiply for computed...


  // FIRST cut indices base on tranposed-shape
  // Before first axes of tranpose, which is from 0-chunk1, remain the same
  int before = tranposed_address % chunk1;
  // after tranposing, the Second idx, which goes as first after tranposing
  //  is responsible for chunk1's size. The dim is dim2
  int idx2 =   (tranposed_address / chunk1) % dim2;
  // the inbetween is resposible for chunk1* dim2, because
  //    idx2 covered chunk1 and it itself covers dim2
  // The dimension of between is product of all axes between the two,
  //    which can be summarised as chunk2/chunk1 / dim1.
  int between = (tranposed_address / (chunk1 * dim2)) % ((chunk2 / (chunk1 * dim1)));
  /*
  // same proecess can be  thiswritten as below,
  //    where we cut by range-limit of between which is chunk2*dim/dim1, and divide down by its own chunk size
  int between = ((tranposed_address % (chunk2 * dim2 / dim1)) / (chunk1 * dim2));
  // the possible ebnefit is this process uses the same variable as for idx1
  */
  // By the same idea, divide down by its tranposed chunk size which is chunk2 * dim2/dim1, and mod by its dim
  int idx1 = (tranposed_address / (chunk2 * dim2 / dim1)) % dim1;
  // final chunk size is always the same, 
  // since division and multiplcation of dimensions cancel out at the end
  int after = tranposed_address / (chunk2 * dim2);

  // std::cout << after << " " << idx2 << " " << between << " " << idx1 << " " << before << std::endl;
  std::cout << after << " " << idx1 << " " << between << " " << idx2 << " " << before << std::endl;

  int sum = 0;
  sum += before;
  sum += idx1 * chunk1;
  sum += between * chunk1 * dim1;
  sum += idx2 * chunk2;
  sum += after * chunk2 * dim2;

  return sum;
}

int main() {
  using namespace cpp_nn::util;

  // Tensor<int> A({2, 3, 4});
  // Tensor<int> B({4, 5});

  // int i = 0;
  // for (auto it = A.begin(); it != A.end(); ++it) {
  //   *it = ++i;
  // }
  // i = 0;
  // for (auto it = B.begin(); it != B.end(); ++it) {
  //   *it = ++i;
  // }

  // std::cout << "A" << std::endl;
  // PrintTensor(A);
  // std::cout << "B" << std::endl;
  // PrintTensor(B);

  // std::cout << "MULTIPLYING" << std::endl;
  // Tensor<int> C = A * B;
  // PrintTensor(C);


  int val = 0;
  auto gen1 = [&]() -> float { return ++val; };

  auto generator = [val = 0]() mutable { return ++val; };
  Tensor<float> a({2, 3}, generator);
  Tensor<float> b({2, 3}, 2);
  Tensor<float> c({2, 3}, &genVal);
  Tensor<float> d({2, 3}, al);
  Tensor<float> e({2, 3}, randomer());
  struct HERE {
    float operator()() {
      return 10;
    }
  };
  {
    Tensor<float> f({2, 3}, HERE());  
  }
  Tensor<float> f({2, 3}, HERE{});

  std::function<int()> thisWorks = al;

  PrintTensor(a);
  PrintTensor(b);
  PrintTensor(c);
  PrintTensor(d);
  PrintTensor(e);
  PrintTensor(f);

  std::cout << typeid(decltype(gen1)).name() << std::endl;
  std::cout << val << std::endl;

  std::cout << thisWorks() << std::endl;


  // Generator<float> g([&]() {return "NO";});

  // g();

  std::cout << "PADDING" << std::endl;
  Tensor<float> padded = a.Padding({3, 3});
  PrintTensor(padded);
  Tensor<float> cropped = a.Padding({3, 2});
  PrintTensor(cropped);


  std::cout << "Complex" << std::endl;

  Tensor<int> A = ((Tensor<int>({3, 150, 100}, [val = 0]() mutable {return ++val;})
                * Tensor<int>({150*100}, 2).Reshape({100, 150}))
                + (Tensor<int>({20, 10}, 10).Padding({150, 15}) * Tensor<int>({10, 15}, [val=0]() mutable {return +val;}).Padding({15, 150}) )).Padding({1, 10, 10});
  PrintTensor(A);


  std::cout << "TRANSPOSE ORDER optimize for 2" << std::endl;

  // int ax1 = 3;
  // int ax2 = 4;
  // Tensor<int> tpOri({5, 2, 3, 3, 4}, [val=0]()mutable {return val++;});
  
  
  int ax1 = 4;
  int ax2 = 5;
  Tensor<int> tpOri({ 3, 4, 2, 3, 5, 2}, [val=0]()mutable {return val++;});

  std::vector<int> chunk(tpOri.getOrder(), 1);
  size_t cap = 1;
  ComputeCapacityAndChunkSizes(tpOri.getShape(), chunk, cap);
  std::cout << "ORI shape " << std::endl;
  for (auto i : tpOri.getShape()) {
    std::cout << i << ", ";
  }
  std::cout << std::endl;
  std::cout << "ORI CZ " << std::endl;
  for (auto i : chunk) {
    std::cout << i << ", ";
  }
  std::cout << std::endl;

  Tensor<int> tp = tpOri.Transpose(ax1, ax2);
  std::swap(chunk[ax1], chunk[ax2]);
  std::cout << "tra shape " << std::endl;
  for (auto i : tp.getShape()) {
    std::cout << i << ", ";
  }
  std::cout << std::endl;
  std::cout << "swapped CZ " << std::endl;
  for (auto i : chunk) {
    std::cout << i << ", ";
  }
  std::cout << std::endl;

  // dim1 is the inner- of the two axes in original,
  //  that is chunk1 < chunk2
  int dim1 = tpOri.getDimension(ax2);
  int chunk1 = chunk[ax1];

  int dim2 = tpOri.getDimension(ax1);
  int chunk2 = chunk[ax2];

  int SWAPPER_RATIO = chunk2 / chunk1; // this is the amount to swap by

  int after2 = dim2 * chunk2;
  int bef1 = dim2 * chunk1;

  std::cout << dim1 << " " << chunk1 << std::endl;
  std::cout << dim2 << " " << chunk2 << std::endl;

  int idx = 0;
  auto it = tp.begin();
  while (it != tp.end()) {
    /* isolation method,
        find specific indices that are to be tranposed, then apply tranpose only on those indices
    */
    int temp = idx;

    // idx of the splits before transpose
    int before = temp % chunk1;
    temp -= before;

    int idx1 = (temp / chunk1) % dim1;
    temp -= idx1 * chunk1;

    int between = (temp % chunk2) / (dim1 * chunk1);
    temp -= between * chunk1 * dim1;
   
    int idx2 = (temp / chunk2) % dim2;
    temp -= idx2 * chunk2;

    int after = temp / (chunk2 * dim2);
    temp -= after;

    // std::cout << after << " " << idx2 << " " << between << " " << idx1 << " " << before << std::endl;

    // int computed = 0;
    // computed += before * 1; // associated chunk size is 1
    // std::cout << "BEFORE " << before << std::endl;
    // computed += idx2 * chunk1; // associated chunk size is chunk1
    // std::cout << "FIRST " << (idx2 * chunk1) << std::endl;
    // computed += between * dim2 * chunk1; // associated chunk size is chunj1*dim2 because
    //                                       // the tranposed to later retains teh same chunk size, the the axis after
    //                                       // increments now by dim2 not dim1
    // std::cout << "BETWEEN " << (idx2 * chunk1) << std::endl;
    // computed += idx1 * dim2 * (chunk2 / dim1); // associated chunk size is dim2*chunk2/dim1
    //                                           // because associated chunk size is same as old, except that it should multiply dim2 istead of dim1
    // computed += after * chunk2 * dim2; // chunk size of previous times dim1, which cancels out to this                                         
    
    // // last commet was incorrect
    // // This converts Real-address back to Tranposed address
    // // The function we want for tranpose iterator will be inverse of this function.

    /**/
    int computed = TranposedAddressToOriginalAddress(idx,
                                                     dim1, chunk1, 
                                                     dim2, chunk2);
    /**/


    // // shift betweens
    // between *= dim1; // the dim in front gets pushed forwards
    // between /= dim2;

    // int computed = before + after;
    // // computed += idx1 * chunk2;
    // // computed += idx2 * chunk1 * chunk2;

    // computed %= cap;
   

    // int temp = idx;
    // int computed = 0;

    // int before = idx % chunk1
    //             +idx - (idx % (chunk2 * dim2));
    // // find middle too

    // // computed += temp - (temp % after2);
    // // temp %= after2;
    // // computed += temp % chunk1;
    // // temp /= chunk1;
    // // computed += temp * chunk2;

    // // temp /= dim2;
    // // computed += temp * chunk1;


    // // TRY DO IT AS WE NORMALLY DO THIS
    // // everything before the tranposed dim is kept as is
    // computed += temp % chunk1;
    // // everything after the tranpose should be the same too
    // computed += idx - (idx % (chunk2 * dim2));
    // // CORRECT UNTIL HERE

    // // We only need to tranpose tjings between ax1 and ax2
    // temp %= (chunk2 * dim2);
    // temp /= chunk1;


    // // accounts for jump from every small...
    // computed += (temp / dim2) * chunk1;
    // temp %= dim2;
    // computed += temp * chunk2;

    // // To do that...
    // // computed += temp * chunk2;
    // // temp /= dim2;


    // // we can treat the inner as regular end tranpose then
    // // handle above tranpose first

    // // temp /= chunk1;
    // // computed += temp * chunk2;
    // // temp /= dim2;
    // // // computed -= temp * chunk2 * dim2;

    // // temp -= temp % chunk2;
    // std::cout << temp << std::endl;

    // computed += (temp % dim2) * chunk2;
    // temp /= dim2;
    // temp /= dim2;
    // computed += temp * dim1 * chunk1;



    // computed += (temp - (temp % chunk1))  * dim1;

    // computed %= cap;

    // computed += (temp % dim1) * chunk1;
    // temp /= dim2;

    // computed += (temp % chunk2) * chunk1 * dim2;
    // temp /= chunk1;
    // computed += (temp)  * chunk1 * dim2 * chunk1;
    // computed += (temp % chunk2)  * chunk1 * dim1 * chunk2;


    // computed += ((temp ) % dim2) * chunk2;


    // computed %= cap;

    // computed += temp - (temp % after2);
    // temp %= after2;
    // computed += (temp % dim2) * dim1;
    // computed += (temp / dim2);

    // maybe we need to 

    if (idx1 == 0 && idx2 == 0) {
      std::cout << "HERE" << std::endl;

      if (*it != computed) {
        std::cout << "COMP OFF" << std::endl;
      }
    }
    std::cout << idx << "\t" << *it << ",\t" << computed << (*it != computed ? "\t X" : "") <<std::endl;
    if (*it != computed) {
      std::cout << "WRONGK RGV" << std::endl;
      return -1;
    }
    ++it;
    ++idx;
  }
}