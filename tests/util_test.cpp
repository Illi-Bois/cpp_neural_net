#include "gtest/gtest.h"

#include "CPPNeuralNet/Utils/utils.h"

TEST(Util, SanityCheck) {
  EXPECT_TRUE(1);
}

TEST(Util, Increment_Idx) {
  std::vector<int> shape{2,3,1,4};
  std::vector<int> idx{0, 0, 0, 0};

  for (int i = 0; i < 21; ++i){
    bool res = cpp_nn::util::IncrementIndicesByShape(shape.begin(), shape.end(),
                                                      idx.begin(), idx.end());
    EXPECT_TRUE(res);
  }

  EXPECT_EQ(idx, std::vector<int>({1, 2, 0, 1 }));

  cpp_nn::util::IncrementIndicesByShape(shape.begin(), shape.end(),
                                        idx.begin(), idx.end());
  cpp_nn::util::IncrementIndicesByShape(shape.begin(), shape.end(),
                                        idx.begin(), idx.end());
  bool res = cpp_nn::util::IncrementIndicesByShape(shape.begin(), shape.end(),
                                        idx.begin(), idx.end());
  EXPECT_FALSE(res);
  EXPECT_EQ(idx, std::vector<int>({0, 0, 0, 0 }));
}

TEST(Util, Increment_Idx_Subvector) {
  //                        |     |       {3, 2}
  std::vector<int> shape{2, 3, 2, 4, 5};
  //                         |     | 
  std::vector<int> idx{0, 0, 0, 0, 0};

  for (int i = 0; i < 5; ++i){
    bool res = cpp_nn::util::IncrementIndicesByShape(shape.begin() + 1, shape.end() - 2,
                                                      idx.begin() + 2, idx.end() - 1);
    EXPECT_TRUE(res);
  }
  EXPECT_EQ(idx, std::vector<int>({0, 0, 2, 1, 0}));
  bool res = cpp_nn::util::IncrementIndicesByShape(shape.begin() + 1, shape.end() - 2,
                                                    idx.begin() + 2, idx.end() - 1);
  EXPECT_FALSE(res);
  EXPECT_EQ(idx, std::vector<int>({0, 0, 0, 0, 0}));
}


TEST(Util, Increment_Idx_Unequal_Subvector_Longer_Idx) {
  //                        |     |       {3, 2}
  std::vector<int> shape{2, 3, 2, 4, 5};
  //                      |        | 
  std::vector<int> idx{0, 0, 0, 0, 0};

  for (int i = 0; i < 5; ++i){
    bool res = cpp_nn::util::IncrementIndicesByShape(shape.begin() + 1, shape.end() - 2,
                                                      idx.begin() + 1, idx.end() - 1);
    EXPECT_TRUE(res);
  }
  EXPECT_EQ(idx, std::vector<int>({0, 0, 2, 1, 0}));
  bool res = cpp_nn::util::IncrementIndicesByShape(shape.begin() + 1, shape.end() - 2,
                                                    idx.begin() + 1, idx.end() - 1);
  EXPECT_FALSE(res);
  EXPECT_EQ(idx, std::vector<int>({0, 0, 0, 0, 0}));
}

TEST(Util, Increment_Idx_Unequal_Subvector_Longer_Shape) {
  //                     |        |       {3, 2}
  std::vector<int> shape{2, 3, 2, 4, 5};
  //                         |     | 
  std::vector<int> idx{0, 0, 0, 0, 0};

  for (int i = 0; i < 5; ++i){
    bool res = cpp_nn::util::IncrementIndicesByShape(shape.begin(), shape.end() - 2,
                                                      idx.begin() + 2, idx.end() - 1);
    EXPECT_TRUE(res);
  }
  EXPECT_EQ(idx, std::vector<int>({0, 0, 2, 1, 0}));
  bool res = cpp_nn::util::IncrementIndicesByShape(shape.begin(), shape.end() - 2,
                                                      idx.begin() + 2, idx.end() - 1);
  EXPECT_FALSE(res);
  EXPECT_EQ(idx, std::vector<int>({0, 0, 0, 0, 0}));
}


TEST(Util, Increment_Idx_Subvector_Change) {
  std::vector<int> shape{2,3,1,4};
  std::vector<int> idx{0, 0, 0, 0};

  for (int i = 0; i < 19; ++i){
    bool res = cpp_nn::util::IncrementIndicesByShape(shape.begin(), shape.end(),
                                                      idx.begin(), idx.end());
    EXPECT_TRUE(res);
  }
  // EXPECT_EQ(idx, std::vector<int>({1, 1, 0, 3 }));

  bool res = cpp_nn::util::IncrementIndicesByShape(shape.begin(), shape.end() - 2,    // {2, 3}
                                                    idx.begin() + 1, idx.end() - 1);  // {1, 0}
  EXPECT_EQ(idx, std::vector<int>({1, 1, 1, 3 }));
  EXPECT_TRUE(res);

  cpp_nn::util::IncrementIndicesByShape(shape.begin(), shape.end() - 2,    // {2, 3}
                                                    idx.begin() + 1, idx.end() - 1);  // -> {1, 2}
  res = cpp_nn::util::IncrementIndicesByShape(shape.begin(), shape.end() - 2,    // {2, 3}
                                                    idx.begin() + 1, idx.end() - 1);  // -> {0, 0}

  EXPECT_FALSE(res);
  EXPECT_EQ(idx, std::vector<int>({1, 0, 0, 3}));
}