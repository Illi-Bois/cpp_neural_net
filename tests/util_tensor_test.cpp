#include "gtest/gtest.h"

#include "CPPNeuralNet/Utils/tensor.h"


namespace cpp_nn {
namespace util {

TEST(UtilTensor, SanityCheck) {
  EXPECT_TRUE(1);
}



TEST(UtilTensorConstructor, InitList) {
    Tensor<float> t1({2, 3, 4}, 0.0f);
    EXPECT_EQ(t1.getOrder(), 3);
    EXPECT_EQ(t1.getDimension(0), 2);
    EXPECT_EQ(t1.getDimension(1), 3);
    EXPECT_EQ(t1.getDimension(2), 4);

    Tensor<int> t2({2, 2}, 5);
    EXPECT_EQ(t2.getElement({0, 0}), 5);
    EXPECT_EQ(t2.getElement({1, 1}), 5);
}


TEST(UtilTensorConstructor, VectorDims) {
    std::initializer_list<int> dims = {2, 3, 4};
    Tensor<float> t(dims, 1.5f);
    EXPECT_EQ(t.getOrder(), 3);
    EXPECT_EQ(t.getDimension(0), 2);
    EXPECT_EQ(t.getDimension(1), 3);
    EXPECT_EQ(t.getDimension(2), 4);
    EXPECT_FLOAT_EQ(t.getElement({0, 0, 0}), 1.5f);
}



TEST(UtilTensorOperations, Multiplication) {
    Tensor<float> t1({2, 3}, 2.0f);
    Tensor<float> t2({3, 2}, 1.5f);
    auto t3 = t1 * t2;
    EXPECT_EQ(t3.getOrder(), 2);
    EXPECT_EQ(t3.getDimension(0), 2);
    EXPECT_EQ(t3.getDimension(1), 2);
    EXPECT_FLOAT_EQ(t3.getElement({0, 0}), 9.0f);
}


}
}