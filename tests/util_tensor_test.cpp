#include "gtest/gtest.h"

#include "CPPNeuralNet/Utils/tensor.h"

#include <numeric>


namespace {

// returns 0 when increment failed
int increment(std::vector<int>& idx, const std::vector<int>& dims) {
  for (int i = idx.size() - 1; i >= 0; --i) {
    ++idx[i];

    if (idx[i] >= dims[i]) {
      idx[i] = 0;
      continue;
    } else {
      return 1;
    }
  }

  return 0;
}

} // namespace  


TEST(UtilTensor, SanityCheck) {
  EXPECT_TRUE(1);
}

TEST(UtilTensor, Constructor_By_Dimension) {
  // checks if the contructoed has correct shape
  using namespace cpp_nn::util;

  std::vector<std::vector<int>> test_dims = {
                                              {4, 2, 4},
                                              {1},
                                              {1, 1, 1, 2}
                                            };

  for (auto& dim : test_dims) {
    rTensor<int> tens(dim);
  
    int order = tens.getOrder();

    for (int i = 0; i < order; ++i) {
      EXPECT_EQ(tens.getDimension(i), dim[i]);
    }

    EXPECT_EQ(tens.getShape(), dim);

    int cap = std::accumulate(dim.begin(), dim.end(), 1, [](int a, int b)->int {return a*b;});
    EXPECT_EQ(tens.getCapacity(), cap);
  }
}

TEST(UtilTensor, Constructor_By_Dimension_Negative_Dimension) {
  // checks if the contructoed has correct shape
  using namespace cpp_nn::util;

  std::vector<std::vector<int>> test_dims = {
                                              {2, -1, 3},
                                              {-1, -2},
                                              {0, 1}
                                            };

  for (auto& dim : test_dims) {
    EXPECT_THROW({
      try
      {
        rTensor<int> tens(dim);
      }
      catch( const std::invalid_argument& e )
      {
        // and this tests that it has the correct message
        EXPECT_STREQ( "ChunkSize and Capacity- Non-positive dimension given", e.what() );
        throw;
      }
    }, std::invalid_argument);
  }
}


TEST(UtilTensor, Construct_With_InitVal) {
  using namespace cpp_nn::util;

  std::vector<std::vector<int>> test_dims = {
                                              {4, 2, 4},
                                              {1},
                                              {1, 1, 1, 2}
                                            };
  
  for (auto& dim : test_dims) {
    rTensor<int> tens(dim);
    rTensor<int> tensInit(dim, 11);

    std::vector<int> idx (dim.size(), 0);

    do {
      EXPECT_EQ(tens.getElement(idx), 0);
      EXPECT_EQ(tensInit.getElement(idx), 11);
    } while (::increment(idx, dim));
  }
}

TEST(UtilTensor, MoveConstsructor) {
  using namespace cpp_nn::util;

  rTensor<int> tens({2, 2, 2});

  tens.getElement({1, 0, 1}) = 5;
  tens.getElement({1, 1, 1}) = 7;
  tens.getElement({0, 0, 1}) = 1;

  rTensor<int> copied(tens);

  EXPECT_EQ(copied.getElement({1, 0, 1}), 5);
  EXPECT_EQ(copied.getElement({1, 1, 1}), 7);
  EXPECT_EQ(copied.getElement({0, 0, 1}), 1);
}

TEST(UtilTensor, CopyConstsructor) {
  using namespace cpp_nn::util;

  rTensor<int> tens({2, 2, 2});

  tens.getElement({1, 0, 1}) = 5;
  tens.getElement({1, 1, 1}) = 7;
  tens.getElement({0, 0, 1}) = 1;

  rTensor<int> copied(tens);

  EXPECT_EQ(copied.getElement({1, 0, 1}), 5);
  EXPECT_EQ(copied.getElement({1, 1, 1}), 7);
  EXPECT_EQ(copied.getElement({0, 0, 1}), 1);

  // see is deep copy
  tens.getElement({1, 0, 1}) = 15;
  tens.getElement({1, 1, 1}) = 17;
  tens.getElement({0, 0, 1}) = 11;


  EXPECT_EQ(copied.getElement({1, 0, 1}), 5);
  EXPECT_EQ(copied.getElement({1, 1, 1}), 7);
  EXPECT_EQ(copied.getElement({0, 0, 1}), 1);
}

TEST(UtilTensor, Dimension_Negative_Index_Getter) {
  using namespace cpp_nn::util;

  rTensor<int> tensor_one({2, 3, 4});

  EXPECT_EQ(tensor_one.getDimension(-1), 4);
  EXPECT_EQ(tensor_one.getDimension(-2), 3);
  EXPECT_EQ(tensor_one.getDimension(-3), 2);
}



TEST(UtilTensor, Constant_Tensor_Element_Getter) {
  using namespace cpp_nn::util;

  rTensor<int> tensor_one({2, 3, 4}, 10);

  tensor_one.getElement({1, 1, 1}) = 11;
  tensor_one.getElement({1, 2, 1}) = 11;
  tensor_one.getElement({1, 2, 3}) = 11;

  // make constant copy
  const rTensor<int> tensor_two = tensor_one;

  EXPECT_EQ(tensor_two.getElement({1, 1, 1}), 11);
  EXPECT_EQ(tensor_two.getElement({1, 2, 1}), 11);
  EXPECT_EQ(tensor_two.getElement({1, 2, 3}), 11);
}