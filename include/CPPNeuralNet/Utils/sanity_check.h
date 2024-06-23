#ifndef CPP_NN_UTIL_SANITY_CHECK
#define CPP_NN_UTIL_SANITY_CHECK


class TesterClass {
 private:
  int a_;
 public:
  TesterClass(int a);
  inline int getA() const {
    return a_;
  }
};

#endif // CPP_NN_UTIL_SANITY_CHECK