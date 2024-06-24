#ifndef CPP_NN_UTIL_SANITY_CHECK
#define CPP_NN_UTIL_SANITY_CHECK


template<typename T>
class TesterClass {
 private:
  T a_;
 public:
  TesterClass(T a);
  inline int getA() const {
    return a_;
  }
};

#include "../src/CPPNeuralNet/Utils/sanity_check.tpp"

#endif // CPP_NN_UTIL_SANITY_CHECK