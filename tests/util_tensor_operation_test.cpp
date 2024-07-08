#include "gtest/gtest.h"

#include "CPPNeuralNet/Utils/re_tensor.h"

TEST(UtilTensorOperation, SanityCheck) {
  EXPECT_TRUE(true);
}

// TRANSPOSE =============================================
TEST(UtilTensorOperation, Transpose_Operation) {
  using namespace cpp_nn::util;
  //Large Dimensions
  rTensor<int> a({1000,1005}, 1.0f);
  auto b = a.Transpose(0,1);
  EXPECT_EQ(b.getShape(), std::vector<int>({1005,1000}));

  //1D tensor
  rTensor<float> c({5}, 1.0f); 
  auto d = c.Transpose(0,0);
  EXPECT_EQ(d.getShape(), std::vector<int>({5}));
  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(d.getElement({i}), 1.0f);
  }
}
TEST(UtilTensorOperation, Undoing_Transpose) {
  using namespace cpp_nn::util;
  rTensor<float> a({3,4,5});
  auto b = a.Transpose(0,1).Transpose(0,1);
  EXPECT_EQ(a.getShape(), b.getShape());
}
TEST(UtilTensorOperation, Chained_Transpose) {
  using namespace cpp_nn::util;
  rTensor<float> a({2,3,4,5});
  //{2,3,4,5} -> {3,2,4,5} -> {3,4,2,5} -> {3,4,5,2}
  auto b = a.Transpose(0,1).Transpose(1,2).Transpose(2,3);
  EXPECT_EQ(b.getShape(), std::vector<int>({3,4,5,2}));
}
TEST(UtilTensorOperation, Negative_Indexing) {
  using namespace cpp_nn::util;
  rTensor<float> a({2,3,4});
  auto b = a.Transpose(-3,-1);
  EXPECT_EQ(b.getShape(), std::vector<int>({4,3,2}));
  auto c = a.Transpose(-3,1);
  EXPECT_EQ(c.getShape(), std::vector<int>({3,2,4}));
}
TEST(UtilTensorOperation, Transpose_Self) {
  using namespace cpp_nn::util;
  rTensor<float> a({5,4,2});
  auto b = a.Transpose(1,1);
  auto c = a.Transpose(0,0);
  EXPECT_EQ(a.getShape(), b.getShape());
  EXPECT_EQ(a.getShape(), c.getShape());
}
// End of TRANSPOSE ======================================

// SUMMATION =============================================
TEST(UtilTensorOperation, Summation_Of_Two) {
  
}
TEST(UtilTensorOperation, Summation_Of_Three) {
  
}
TEST(UtilTensorOperation, Summation_Of_many_with_parenthesis) {
  
}
TEST(UtilTensorOperation, SelfSumming) {
  
}
TEST(UtilTensorOperation, Summing_To_Self) {
  
}
TEST(UtilTensorOperation, Summing_To_Self_with_others_inbetween) {
  
}
TEST(UtilTensorOperation, Summing_Incorrect_Dimensions) {
  
}
TEST(UtilTensorOperation, Summing_broadcast) {
  
}
TEST(UtilTensorOperation, Summing_broadcast_with_diff_order) {
  
}
// End of SUMMATION ======================================

// MULTIPLICATION ========================================
TEST(UtilTensorOperation, Multiplication_of_two) {
  
}
TEST(UtilTensorOperation, Multiplication_of_three) {
  
}
TEST(UtilTensorOperation, Multiplication_of_many_with_parenthesis) {
  
}
TEST(UtilTensorOperation, Multiplication_to_self) {
  
}
TEST(UtilTensorOperation, Multiplication_of_many_involving_to_self) {
  
}
TEST(UtilTensorOperation, Mult_Incorrect_matrix_dim) {
  
}
TEST(UtilTensorOperation, Mult_Incorrect_tensor_dim) {
  
}
TEST(UtilTensorOperation, Multiplying_broadcast) {
  
}
TEST(UtilTensorOperation, Multiplying_broadcast_with_diff_order) {
  
}
// END OF MULTIPLICATION =================================

// RESHAPE ===============================================
TEST(UtilTensorOperation, Rehsaping) {
  
}
TEST(UtilTensorOperation, Rehsaping_to_self) {
  
}
TEST(UtilTensorOperation, Reshaping_chained) {
  
}
TEST(UtilTensorOperation, Reshape_incorrect_capacity) {
  
}
TEST(UtilTensorOperation, Reshape_with_non_post_dim) {
  
}
// END OF RESHAPE ========================================

// PADDING ===============================================
TEST(UtilTensorOperation, Padding) {
  
}
TEST(UtilTensorOperation, Padding_to_self) {
  
}
TEST(UtilTensorOperation, Multiple_Padding) {
  
}
TEST(UtilTensorOperation, Reductive_padding) {
  
}
TEST(UtilTensorOperation, Padding_incorrect_order) {
  
}
TEST(UtilTensorOperation, Padding_non_postive_dim) {
  
}
// END OF PADDING ========================================



// MULTIPLE OPERATION CHAINED ============================
TEST(UtilTensorOperation, Chaining_Different_Operations) {
  // do while and go ham, 
}
// End of MULTIPLE OPERATION CHAINED =====================