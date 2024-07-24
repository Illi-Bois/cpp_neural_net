#include "gtest/gtest.h"

#include "CPPNeuralNet/model.h"
#include "CPPNeuralNet/Utils/tensor.h"


TEST(ModelTest, SanityCheck) {
  EXPECT_TRUE(true);
}

TEST(ModelTest, ModelConstruction_And_Passes) {
  using namespace cpp_nn;
  using namespace cpp_nn::util;

  Model model;

  Tensor<float> input = AsTensor<float>({1, 2, 3, 4, 5}).Reshape({5, 1});

  Tensor<float> forward = model.forward(input);
  auto it = forward.begin();
  auto end = forward.end();
  auto ori = input.begin();

  while (it != end) {
    EXPECT_EQ(*it, *ori);
    
    ++it;
    ++ori;
  }
}

