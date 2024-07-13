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
  //checks if element in a hasn't changed
  val = 0;
  for (auto it = a.begin(); it != a.end(); ++it) {
    EXPECT_EQ(*it, ++val);
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
  auto it1 = a.begin();
  auto it2 = a.begin();
  auto fin = a.end();
  while (it1 != fin) {
    EXPECT_EQ(*it1, *it2);
    ++it1;
    ++it2;
  }
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
TEST(UtilTensorOperation, Summation_Of_Two_Constructor) {
  using namespace cpp_nn::util;
  auto generator = [val = 0]() mutable {return ++val; };
  auto generator2 = [val = 0]() mutable {return ++val * 2; };
  std::vector<int> expected_elements3 = 
  {
    3,    6,    9,
    12,    15,    18,
  };
  Tensor<float> c = Tensor<float>({2, 3}, generator) + Tensor<float>({2, 3}, generator2);
  auto it3 = c.begin();
  for (auto exp : expected_elements3) {
    EXPECT_EQ(*it3, exp);
    ++it3;
  }
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
  using namespace cpp_nn::util;
    Tensor<int> a({2, 2}, 1);
    Tensor<int> b({2, 2}, -1);
    Tensor<int> c({2, 2}, 3);
    Tensor<int> d({2, 2}, 4);
    Tensor<int> e = (a + b) + (c + d);
    
    EXPECT_EQ(e.getElement({0, 0}), 7);
    EXPECT_EQ(e.getElement({1, 1}), 7);
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
  using namespace cpp_nn::util;
  Tensor<float> a({4, 3, 1, 2});
  std::vector<int> idx(a.getOrder(), 0);
  int val = 0;
  for (auto it = a.begin(); it != a.end(); ++it) {
    *it = ++val;
  }
  a = a + a;
  val = 0;
  for (auto it = a.begin(); it != a.end(); ++it) {
    EXPECT_EQ(*it, ++val * 2);
  }
}
TEST(UtilTensorOperation, Summing_To_Self_with_others_inbetween) {
  using namespace cpp_nn::util;
  Tensor<float> a({2, 2}, 1);
  Tensor<float> b({2, 2}, -2);
  a = a + b + a;
  EXPECT_EQ(a.getElement({0, 0}), 0);
  EXPECT_EQ(a.getElement({1, 1}), 0);
}
TEST(UtilTensorOperation, Summing_Incorrect_Dimensions) {
  using namespace cpp_nn::util;
  Tensor<int> A({2, 2}, 1);
  Tensor<int> B({2, 3}, 2);
  EXPECT_THROW(A + B, std::invalid_argument);
}
// TEST(UtilTensorOperation, Summing_broadcast) {
// //| 1  2  3  4  5 |   -broadcast-> | 1  2  3  4  5 |
// //                                 | 1  2  3  4  5 |
// //                                 | 1  2  3  4  5 | ...
//   using namespace cpp_nn::util;
//   Tensor<float> a({1, 5});
//   int val = 0;
//   for (auto it = a.begin(); it != a.end(); ++it) {
//     *it = ++val;
//   }
//   Tensor<float> b({6,5},3);
//   Tensor<float> c = a + b;
//   val = 0;
//   for (int i = 0; i < 6; ++i) {
//     for (int j = 0; j < 5; ++j) {
//       EXPECT_EQ(c.getElement({i, j}), (j + 1) + 3);
//     }
//   }
// }
TEST(UtilTensorOperation, Summing_broadcast) {
//| 1  2  3  4  5 |   -broadcast-> | 1  2  3  4  5 |
//                                 | 1  2  3  4  5 |
//                                 | 1  2  3  4  5 | ...
  using namespace cpp_nn::util;
  Tensor<float> a({1, 5});
  int val = 0;
  for (auto it = a.begin(); it != a.end(); ++it) {
    *it = ++val;
  }
  Tensor<float> b({6,5},3);
  Tensor<float> c = a + b;
  val = 0;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 5; ++j) {
      EXPECT_EQ(c.getElement({i, j}), (j + 1) + 3);
    }
  }
  //Reset after 5 elements
  std::vector<int> resetPositions = {5};
  int iterationCount = 0;
  int resetIndex = 0;
  std::vector<int> cIndex(c.getOrder(), 0);
  do {
    //iterationCount 0,5,10 -> aIndex = 0
    //iterationCount 2,7,12 -> aIndex = 2
    int aIndex = iterationCount % a.getCapacity();
    std::vector<int> aIdx {0, aIndex};
    EXPECT_EQ(c.getElement(cIndex), a.getElement(aIdx) + b.getElement(cIndex));
    ++iterationCount;
    if (resetIndex < resetPositions.size() && iterationCount == resetPositions[resetIndex]) {
        iterationCount = 0;
        ++resetIndex;
    }
  } while (IncrementIndicesByShape(c.getShape().begin(), c.getShape().end(), cIndex.begin(), cIndex.end()));
}
TEST(UtilTensorOperation, Summing_broadcast_with_diff_order) {
  using namespace cpp_nn::util;
  Tensor<float> a(   {2, 3});
  Tensor<float> b({4, 1, 3}, 2);
  int val = 0;
  for (auto it = a.begin(); it != a.end(); ++it) {
    *it = ++val;
  }
  Tensor<float> c = a + b; //resulting dimension would be {4, 2, 3}
  EXPECT_EQ(c.getShape(), std::vector<int>({4, 2, 3}));
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 3; ++k) {
        // a{0,j,k} as a's 0 position has been broadcastest, same for b
        EXPECT_EQ(c.getElement({i, j, k}), a.getElement(/*0*/{j, k}) + b.getElement({i, 0, k}));
      }
    }
  }

}
// End of SUMMATION ======================================

// MULTIPLICATION ========================================
TEST(UtilTensorOperation, Multiplication_of_two) {
  using namespace cpp_nn::util;
  //2D
  int val = 0;
  Tensor<float> a({2, 3});
  for (auto it = a.begin(); it != a.end(); ++it) {
    *it = ++val;
  }
  Tensor<float> b({3, 2});
  val = 16;
  for (auto it = b.begin(); it != b.end(); ++it) {
    *it = --val;
  }
  Tensor<float> c1 = a * b;
  Tensor<float> expected1({2, 2});
  expected1.getElement({0, 0}) = 74;
  expected1.getElement({0, 1}) = 68;
  expected1.getElement({1, 0}) = 191;
  expected1.getElement({1, 1}) = 176;
  Tensor<float> c2 = b * a;
  for(int i = 0; i < 2; ++i){
    for(int j = 0; j < 2; ++j){
      EXPECT_EQ(c1.getElement({i, j}), expected1.getElement({i, j}));
    }
  }
  Tensor<float> expected2({3, 3});
  expected2.getElement({0, 0}) = 71;
  expected2.getElement({0, 1}) = 100;
  expected2.getElement({0, 2}) = 129;
  expected2.getElement({1, 0}) = 61;
  expected2.getElement({1, 1}) = 86;
  expected2.getElement({1, 2}) = 111;
  expected2.getElement({2, 0}) = 51;
  expected2.getElement({2, 1}) = 72;
  expected2.getElement({2, 2}) = 93;
  for(int i = 0; i < 3; ++i){
    for(int j = 0; j < 3; ++j){
      EXPECT_EQ(c2.getElement({i, j}), expected2.getElement({i, j}));
    }
  }
}
TEST(UtilTensorOperation, Multiplication_of_three) {
  using namespace cpp_nn::util;
  Tensor<float> a({3, 4});
  Tensor<float> b({4, 2});
  Tensor<float> c({2, 3});
  int val = 0;
  for (auto it = a.begin(); it != a.end(); ++it) {
    *it = ++val;
  }
  val = 14;
  for (auto it = b.begin(); it != b.end(); ++it) {
    *it = --val;
  }
  val = 1;
  for (auto it = c.begin(); it != c.end(); ++it) {
    *it = ++val * 2;
  }
  Tensor<float> d = a * b * c;
  EXPECT_EQ(d.getElement({0, 0}), 1160);
  EXPECT_EQ(d.getElement({0, 1}), 1500);
  EXPECT_EQ(d.getElement({0, 2}), 1840);
  EXPECT_EQ(d.getElement({1, 0}), 3240);
  EXPECT_EQ(d.getElement({1, 1}), 4188);
  EXPECT_EQ(d.getElement({1, 2}), 5136);
  EXPECT_EQ(d.getElement({2, 0}), 5320);
  EXPECT_EQ(d.getElement({2, 1}), 6876);
  EXPECT_EQ(d.getElement({2, 2}), 8432);
  Tensor<float> e = c * a * b;
  EXPECT_EQ(e.getElement({0, 0}), 5140);
  EXPECT_EQ(e.getElement({0, 1}), 4608);
  EXPECT_EQ(e.getElement({1, 0}), 9640);
  EXPECT_EQ(e.getElement({1, 1}), 8640);
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
  using namespace cpp_nn::util;
  
  Tensor<int> A({2, 3, 4});
  Tensor<int> B({4, 5});

  int i = 0;
  for (auto it = A.begin(); it != A.end(); ++it) {
    *it = ++i;
  }
  i = 0;
  for (auto it = B.begin(); it != B.end(); ++it) {
    *it = ++i;
  }

  Tensor<int> C = A * B;
  

  std::vector<int> expected_elements = 
    {
      110,    120,    130,    140,    150,
      246,    272,    298,    324,    350,
      382,    424,    466,    508,    550,
      
      518,    576,    634,    692,    750,
      654,    728,    802,    876,    950,
      790,    880,    970,    1060,   1150,
    };
  
  PrintTensor(C);
  
  EXPECT_EQ(C.getShape(), std::vector<int>({2, 3, 5}));
  auto it = C.begin();
  for (auto exp : expected_elements) {
    EXPECT_EQ(*it, exp);
    ++it;
  }
}
TEST(UtilTensorOperation, Multiplying_broadcast_with_diff_order) {
  
}

TEST(UtilTensorOperation, Multiplying_Incredibly_large) {
  // checking that it doesnt crash. also running to see runtime
  using namespace cpp_nn::util;

  // THESE ARE TOO BIG,
  // Tensor<float> A(   {3, 4, 5, 6, 1, 100, 150});
  // Tensor<float> B({2, 3, 1, 5, 6, 3, 150, 200});

  Tensor<float> A({3, 100, 150});
  Tensor<float> B(   {150, 200});
  float val = 0;

  Tensor<float>::Iterator ait = A.begin();
  Tensor<float>::Iterator aend = A.end();
  while (ait != aend) {
    *ait = val;
    ++val;
    val /= 2;

    ++ait;
  }

  val = 0;
  Tensor<float>::Iterator bit = B.begin();
  Tensor<float>::Iterator bend = B.end();
  while (bit != bend) {
    *bit = val;
    ++val;
    ++val;
    val /= 3;

    ++bit;
  }

  std::cout << "READY" << std::endl;

  Tensor<float> C = A * B;
  EXPECT_TRUE(true); // simply a sanity check
  // want only to see that this does not fail
}
TEST(UtilTensorOperation, Multiplying_Incredibly_large2) {
  // checking that it doesnt crash. also running to see runtime
  using namespace cpp_nn::util;

  // THESE ARE TOO BIG,
  // Tensor<float> A(   {3, 4, 5, 6, 1, 100, 150});
  // Tensor<float> B({2, 3, 1, 5, 6, 3, 150, 200});

  Tensor<float> A({3, 200, 150});
  Tensor<float> B(   {150, 200});
  float val = 0;

  Tensor<float>::Iterator ait = A.begin();
  Tensor<float>::Iterator aend = A.end();
  while (ait != aend) {
    *ait = val;
    ++val;
    val /= 2;

    ++ait;
  }

  val = 0;
  Tensor<float>::Iterator bit = B.begin();
  Tensor<float>::Iterator bend = B.end();
  while (bit != bend) {
    *bit = val;
    ++val;
    ++val;
    val /= 3;

    ++bit;
  }

  std::cout << "READY" << std::endl;

  Tensor<float> C = A * B;
  EXPECT_TRUE(true); // simply a sanity check
  // want only to see that this does not fail
}
TEST(UtilTensorOperation, Multiplying_Incredibly_large3) {
  // checking that it doesnt crash. also running to see runtime
  using namespace cpp_nn::util;

  // THESE ARE TOO BIG,
  // Tensor<float> A(   {3, 4, 5, 6, 1, 100, 150});
  // Tensor<float> B({2, 3, 1, 5, 6, 3, 150, 200});

  Tensor<float> A({2, 1, 3, 200, 150});
  Tensor<float> B(   {2, 1, 150, 200});
  float val = 0;

  Tensor<float>::Iterator ait = A.begin();
  Tensor<float>::Iterator aend = A.end();
  while (ait != aend) {
    *ait = val;
    ++val;
    val /= 2;

    ++ait;
  }

  val = 0;
  Tensor<float>::Iterator bit = B.begin();
  Tensor<float>::Iterator bend = B.end();
  while (bit != bend) {
    *bit = val;
    ++val;
    ++val;
    val /= 3;

    ++bit;
  }

  std::cout << "READY" << std::endl;

  Tensor<float> C = A * B;
  EXPECT_TRUE(true); // simply a sanity check
  // want only to see that this does not fail
}

TEST(UtilTensorOperation, Multiplying_Incredibly_large4) {
  // checking that it doesnt crash. also running to see runtime
  using namespace cpp_nn::util;

  // THESE ARE TOO BIG,
  // Tensor<float> A(   {3, 4, 5, 6, 1, 100, 150});
  // Tensor<float> B({2, 3, 1, 5, 6, 3, 150, 200});

  Tensor<float> A({500, 300});
  Tensor<float> B({300, 500});
  float val = 0;

  Tensor<float>::Iterator ait = A.begin();
  Tensor<float>::Iterator aend = A.end();
  while (ait != aend) {
    *ait = val;
    ++val;
    val /= 2;

    ++ait;
  }

  val = 0;
  Tensor<float>::Iterator bit = B.begin();
  Tensor<float>::Iterator bend = B.end();
  while (bit != bend) {
    *bit = val;
    ++val;
    ++val;
    val /= 3;

    ++bit;
  }

  std::cout << "READY" << std::endl;

  Tensor<float> C = A * B;
  EXPECT_TRUE(true); // simply a sanity check
  // want only to see that this does not fail
}
// END OF MULTIPLICATION =================================

// RESHAPE ===============================================
TEST(UtilTensorOperation, Reshaping) {
  
}
TEST(UtilTensorOperation, Reshaping_to_self) {
  
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



TEST(UtilTensorOperation, Chaining_Different_Operations_SPECIFICALLY_Tranpose_Reshape) {
  using namespace cpp_nn::util;

  Tensor A({2, 3, 4}, 0);
  int val = 0; 
  auto ait = A.begin();
  auto aend = A.end();

  while (ait != aend) {
    *ait = ++val;
    ++ait;
  }

  Tensor C = A.Transpose(0, 1).Reshape({3, 2, 4});

  // with assumptinog that individual operation works fine
  Tensor D = A.Transpose(0, 1);
  D = D.Reshape({3, 2, 4});

  auto it = C.begin();
  auto endit = C.end();

  auto oit = D.begin();

  while (it != endit) {
    EXPECT_EQ(*it, *oit);
    ++it;
    ++oit;
  }
}

TEST(UtilTensorOperation, Chaining_Different_Operations_SPECIFICALLY_Reshape_Tranpose) {
  using namespace cpp_nn::util;

  Tensor A({2, 3, 4}, 0);
  int val = 0; 
  auto ait = A.begin();
  auto aend = A.end();

  while (ait != aend) {
    *ait = ++val;
    ++ait;
  }

  Tensor C = A.Reshape({3, 2, 4}).Transpose(0, 1);

  // with assumptinog that individual operation works fine
  Tensor D = A.Reshape({3, 2, 4});
  D = D.Transpose(0, 1);

  auto it = C.begin();
  auto endit = C.end();

  auto oit = D.begin();

  while (it != endit) {
    EXPECT_EQ(*it, *oit);
    ++it;
    ++oit;
  }
}

TEST(UtilTensorOperation, Chaining_Different_Operations_SPECIFICALLY_Tranpose_Reshape_Transpose) {
  using namespace cpp_nn::util;

  Tensor A({2, 3, 4}, 0);
  int val = 0; 
  auto ait = A.begin();
  auto aend = A.end();

  while (ait != aend) {
    *ait = ++val;
    ++ait;
  }

  Tensor C = A.Transpose(0, 1).Reshape({3, 2, 4}).Transpose(0, 2);

  // with assumptinog that individual operation works fine
  Tensor D = A.Transpose(0, 1);
  D = D.Reshape({3, 2, 4});
  D = D.Transpose(0, 2);

  auto it = C.begin();
  auto endit = C.end();

  auto oit = D.begin();

  while (it != endit) {
    EXPECT_EQ(*it, *oit);
    ++it;
    ++oit;
  }
}


TEST(UtilTensorOperation, Chaining_Different_Operations_SPECIFICALLY_Tranpose_Reshape_Transpose_and_so_on) {
  using namespace cpp_nn::util;

  Tensor A({2, 3, 4}, 0);
  int val = 0; 
  auto ait = A.begin();
  auto aend = A.end();

  while (ait != aend) {
    *ait = ++val;
    ++ait;
  }

// Ending with Tranpose
  Tensor C = A.Transpose()
              .Reshape({3, 2, 4})
              .Transpose()
              .Reshape({3, 2, 4})
              .Transpose()
              .Reshape({3, 2, 4})
              .Transpose();

  // with assumptinog that individual operation works fine
  Tensor D = A.Transpose();
  D = D.Reshape({3, 2, 4});
  D = D.Transpose();
  D = D.Reshape({3, 2, 4});
  D = D.Transpose();
  D = D.Reshape({3, 2, 4});
  D = D.Transpose();

  {
  auto it = C.begin();
  auto endit = C.end();

  auto oit = D.begin();

  while (it != endit) {
    EXPECT_EQ(*it, *oit);
    ++it;
    ++oit;
  }
  }


// Ending with Reshape
  C = A.Transpose()
              .Reshape({3, 2, 4})
              .Transpose()
              .Reshape({3, 2, 4})
              .Transpose()
              .Reshape({3, 2, 4})
              .Transpose()
              .Reshape({3, 2, 4});

  // with assumptinog that individual operation works fine
  D = A.Transpose();
  D = D.Reshape({3, 2, 4});
  D = D.Transpose();
  D = D.Reshape({3, 2, 4});
  D = D.Transpose();
  D = D.Reshape({3, 2, 4});
  D = D.Transpose();
  D = D.Reshape({3, 2, 4});

  {
  auto it = C.begin();
  auto endit = C.end();

  auto oit = D.begin();

  while (it != endit) {
    EXPECT_EQ(*it, *oit);
    ++it;
    ++oit;
  }
  }
}



TEST(UtilTensorOperation, Chaining_Different_Operations_SPECIFICALLY_MultiTranspose_and_Reshape_hard) {
  using namespace cpp_nn::util;

  Tensor A({2, 3, 4}, 0);
  int val = 0; 
  auto ait = A.begin();
  auto aend = A.end();

  while (ait != aend) {
    *ait = ++val;
    ++ait;
  }

// Ending with Tranpose
  Tensor C = A.Transpose(0, 1)
              .Transpose(0, 2)
              .Reshape({3, 2, 4})
              .Transpose(0, 1)
              .Transpose(1, 2)
              .Reshape({3, 2, 4})
              .Transpose(0, 2)
              .Transpose(1, 2)
              .Transpose(1, 0)
              .Reshape({3, 2, 4})
              .Transpose(0, 2)
              .Transpose(1, 2);

  // with assumptinog that individual operation works fine
  Tensor D = A;
  D = D.Transpose(0, 1);
  D = D.Transpose(0, 2);
  D = D.Reshape({3, 2, 4});
  D = D.Transpose(0, 1);
  D = D.Transpose(1, 2);
  D = D.Reshape({3, 2, 4});
  D = D.Transpose(0, 2);
  D = D.Transpose(1, 2);
  D = D.Transpose(1, 0);
  D = D.Reshape({3, 2, 4});
  D = D.Transpose(0, 2);
  D = D.Transpose(1, 2);

  {
  auto it = C.begin();
  auto endit = C.end();

  auto oit = D.begin();

  while (it != endit) {
    EXPECT_EQ(*it, *oit);
    ++it;
    ++oit;
  }
  }


// Ending with Reshape
  C = A.Transpose(0, 1)
              .Transpose(0, 2)
              .Reshape({3, 2, 4})
              .Transpose(0, 1)
              .Transpose(1, 2)
              .Reshape({3, 2, 4})
              .Transpose(0, 2)
              .Transpose(1, 2)
              .Transpose(1, 0)
              .Reshape({3, 2, 4})
              .Transpose(0, 2)
              .Transpose(1, 2)
              .Reshape({3, 2, 4});

  // with assumptinog that individual operation works fine
  D = A;
  D = D.Transpose(0, 1);
  D = D.Transpose(0, 2);
  D = D.Reshape({3, 2, 4});
  D = D.Transpose(0, 1);
  D = D.Transpose(1, 2);
  D = D.Reshape({3, 2, 4});
  D = D.Transpose(0, 2);
  D = D.Transpose(1, 2);
  D = D.Transpose(1, 0);
  D = D.Reshape({3, 2, 4});
  D = D.Transpose(0, 2);
  D = D.Transpose(1, 2);
  D = D.Reshape({3, 2, 4});

  {
  auto it = C.begin();
  auto endit = C.end();

  auto oit = D.begin();

  while (it != endit) {
    EXPECT_EQ(*it, *oit);
    ++it;
    ++oit;
  }
  }
}
// End of MULTIPLE OPERATION CHAINED =====================