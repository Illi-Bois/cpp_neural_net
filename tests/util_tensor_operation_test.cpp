#include "gtest/gtest.h"

#include "CPPNeuralNet/Utils/tensor.h"


TEST(UtilTensorOperation, SanityCheck) {
  EXPECT_TRUE(true);
}


// TRANSPOSE =============================================
TEST(UtilTensorOperation, Transpose_Operation) {
  using namespace cpp_nn::util;
  //Large Dimensions
  Tensor<float> a({30,24}, 1.0f);
  std::vector<int> idx(a.getOrder(), 0);
  int val = 0;
  for (auto it = a.begin(); it != a.end(); ++it) {
    *it = ++val;
  }                                     
  val = 0;
  Tensor<float> b = a.Transpose(0,1);
  for (int j = 0; j < 30; ++j) {
    for (int i = 0; i < 24; ++i) {
      EXPECT_EQ(b.getElement({i, j}), ++val);
    }
  }
  EXPECT_EQ(b.getShape(), std::vector<int>({24,30}));
  //1D tensor
  Tensor<float> c({5}, 1.0f); 
  Tensor<float> d = c.Transpose(0, 0);
  EXPECT_EQ(d.getShape(), std::vector<int>({5}));
  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(d.getElement({i}), 1.0f);
  }
}
TEST(UtilTensorOperation, Undoing_Transpose) {
  using namespace cpp_nn::util;
  Tensor<float> a({3,4,5});
  std::vector<int> idx(a.getOrder(), 0);
  int val = 0;
  for (auto it = a.begin(); it != a.end(); ++it) {
    *it = ++val;
  }      
  Tensor<float> b = a.Transpose(0,1).Transpose(0,1).Transpose(1,2).Transpose(1,2);
  EXPECT_EQ(a.getShape(), b.getShape());
  EXPECT_EQ(a.getElement({2,2,1}), b.getElement({2,2,1}));
}
TEST(UtilTensorOperation, Chained_Transpose) {
  using namespace cpp_nn::util;
  Tensor<float> a({2,3,4,5});
  //{2,3,4,5} -> {3,2,4,5} -> {3,4,2,5} -> {3,4,5,2}
  std::vector<int> idx(a.getOrder(), 0);
  int val = 0;
  for (auto it = a.begin(); it != a.end(); ++it) {
    *it = ++val;
  }
  Tensor<float> b = a.Transpose(0,1).Transpose(1,2).Transpose(2,3);
  EXPECT_EQ(b.getShape(), std::vector<int>({3,4,5,2}));
  std::vector<int> shapeB = b.getShape();
  std::vector<int> shapeA = {2, 3, 4, 5};
  for (int i = 0; i < shapeB[0]; ++i) {
    for (int j = 0; j < shapeB[1]; ++j) {
      for (int k = 0; k < shapeB[2]; ++k) {
        for (int l = 0; l < shapeB[3]; ++l) {
          int expectedValue = a.getElement({l, i, j, k});
          EXPECT_EQ(b.getElement({i, j, k, l}), expectedValue);
        }
      }
    }
  }
}
TEST(UtilTensorOperation, Transpose_Negative_Indexing) {
  using namespace cpp_nn::util;
  Tensor<float> a({2, 3, 4});
  std::vector<int> idx(a.getOrder(), 0);
  int val = 0;
  for (auto it = a.begin(); it != a.end(); ++it) {
    *it = ++val;
  }
  Tensor<float> b = a.Transpose(-3, -1);
  EXPECT_EQ(b.getShape(), std::vector<int>({4, 3, 2}));
  Tensor<float> c = b.Transpose(-3 ,1);
  EXPECT_EQ(c.getShape(), std::vector<int>({3, 4, 2}));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 2; ++k) {
        int expectedValue = b.getElement({j, i, k});
        EXPECT_EQ(c.getElement({i, j, k}), expectedValue);
      }
    }
  }
}
TEST(UtilTensorOperation, Transpose_Self) {
  using namespace cpp_nn::util;
  Tensor<float> a({2, 3, 2});
  std::vector<int> idx(a.getOrder(), 0);
  int val = 0;
  for (auto it = a.begin(); it != a.end(); ++it) {
    *it = ++val;
  }
  a = a.Transpose();
  EXPECT_EQ(a.getShape(), std::vector<int>({2, 2, 3}));
  val = 0;
  for (int i = 0; i < 2; ++i) {
    for (int k = 0; k < 3; ++k) {
      for (int j = 0; j < 2; ++j) {
        EXPECT_EQ(a.getElement({i, j, k}), ++val);
      }
    }
  }
}
// End of TRANSPOSE ======================================

// SUMMATION =============================================
TEST(UtilTensorOperation, Summation_Of_Two) {
  using namespace cpp_nn::util;
  Tensor<float> a({2, 3});
  Tensor<float> b({2, 3}, 3);
  int val = 0;
  std::vector<int> idx(a.getOrder(), 0);
  for (auto it = a.begin(); it != a.end(); ++it) {
    *it = ++val;
  }
  Tensor<float> c = a + b;
  val = 0;
  for (auto it = c.begin(); it != c.end(); ++it) {
    EXPECT_EQ(*it, ++val + 3 );
  }     
  EXPECT_EQ(a.getShape(), c.getShape());
}
TEST(UtilTensorOperation, Summation_Of_Three) {
  using namespace cpp_nn::util;
  Tensor<float> a({3, 4});
  int val = 0;
  std::vector<int> idx(a.getOrder(), 0);
  for (auto it = a.begin(); it != a.end(); ++it) {
    *it = ++val;
  }
  Tensor<float> b({3, 4}, 3);
  Tensor<float> c({3, 4}, 5);
  Tensor<float> d = a + b + c;
  val = 0;
  for (auto it = d.begin(); it != d.end(); ++it) {
    EXPECT_EQ(*it, ++val + 3 + 5);
  }
}
TEST(UtilTensorOperation, Summation_Of_many_with_parenthesis) {
  
}
TEST(UtilTensorOperation, SelfSumming) {
  using namespace cpp_nn::util;
  Tensor<float> a({4, 3});
  std::vector<int> idx(a.getOrder(), 0);
  int val = 0;
  for (auto it = a.begin(); it != a.end(); ++it) {
    *it = ++val;
  }
  Tensor<float> b = a + a;
  val = 0;
  for (auto it = b.begin(); it != b.end(); ++it) {
    EXPECT_EQ(*it, 2 * ++val);
  }
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