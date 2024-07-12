#include "gtest/gtest.h"

#include "CPPNeuralNet/Utils/utils.h"

TEST(Util, SanityCheck) {
  EXPECT_TRUE(1);
}


// INCREMENT TESTS ----------------------------------------------------------------------------
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
// END OF INCREMENT TESTS ---------------------------------------------------------------------


// DECREMENT TEST -----------------------------------------------------------------------------
TEST(Util, Decrement_Test) {
  using namespace cpp_nn::util;

  std::vector<int> shape{2,3,1,4};
  std::vector<int> idx{0, 2, 0, 3};

  EXPECT_TRUE(DecrementIndicesByShape(shape.begin(), shape.end(),
                                      idx.begin(), idx.end()));
                      
  EXPECT_EQ(idx, std::vector<int>({0, 2, 0, 2}));
}

TEST(Util, Decrement_Test_Carry_under) {
  using namespace cpp_nn::util;

  std::vector<int> shape{2,3,2,4};
  std::vector<int> idx{0, 2, 0, 2};

  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(DecrementIndicesByShape(shape.begin(), shape.end(),
                                        idx.begin(), idx.end()));
  }
                      
  EXPECT_EQ(idx, std::vector<int>({0, 1, 1, 3}));
}

TEST(Util, Decrement_Fail) {
  using namespace cpp_nn::util;

  std::vector<int> shape{2,3,2,4};
  std::vector<int> idx{0, 0, 1, 2};

  for (int i = 0; i < 6; ++i) {
    EXPECT_TRUE(DecrementIndicesByShape(shape.begin() + 2, shape.end(),
                                        idx.begin() + 2, idx.end()));
  }
  EXPECT_EQ(idx, std::vector<int>({0, 0, 0, 0}));

  EXPECT_FALSE(DecrementIndicesByShape(shape.begin() + 2, shape.end(),
                                      idx.begin() + 2, idx.end()));
  EXPECT_EQ(idx, std::vector<int>({0, 0, 1, 3}));
}
// END OF DECREMENT TEST ----------------------------------------------------------------------


// MULTI INCREMENT TESTING --------------------------------------------------------------------
TEST(Util, Multi_Increment) {
  using namespace cpp_nn::util;

  std::vector<int> shape{2,3,2,4};
  std::vector<int> idx{0, 0, 1, 2};
  
  EXPECT_TRUE(IncrementIndicesByShape(shape.begin(), shape.end(), idx.begin(), idx.end(), 6));

  EXPECT_EQ(idx, std::vector<int>({0, 1, 1, 0}));
}
TEST(Util, Multi_Increment_Zero) {
  using namespace cpp_nn::util;

  std::vector<int> shape{2,3,2,4};
  std::vector<int> idx{0, 0, 1, 2};
  
  EXPECT_TRUE(IncrementIndicesByShape(shape.begin(), shape.end(), idx.begin(), idx.end(), 0));

  EXPECT_EQ(idx, std::vector<int>({0, 0, 1, 2}));
}
TEST(Util, Multi_Increment_Overflow) {
  using namespace cpp_nn::util;

  std::vector<int> shape{2,3,2,4};
  std::vector<int> idx{0, 0, 1, 2};
  

  EXPECT_FALSE(IncrementIndicesByShape(shape.begin() + 2, shape.end(), 
                                      idx.begin() + 2, idx.end(), 
                                      4));
  EXPECT_EQ(idx, std::vector<int>({0, 0, 0, 0}));
}
TEST(Util, Multi_Increment_Overflow_2) {
  using namespace cpp_nn::util;

  std::vector<int> shape{2,3,2,4};
  std::vector<int> idx{0, 0, 1, 2};
  

  EXPECT_FALSE(IncrementIndicesByShape(shape.begin(), shape.end(), 
                                      idx.begin(), idx.end(), 
                                      1000));
  EXPECT_EQ(idx, std::vector<int>({0, 0, 0, 0}));
}
TEST(Util, Multi_Increment_Right_to_the_end) {
  using namespace cpp_nn::util;

  std::vector<int> shape{2,3,2,4};
  std::vector<int> idx{0, 0, 1, 2};
  

  EXPECT_TRUE(IncrementIndicesByShape(shape.begin() + 2, shape.end(), 
                                      idx.begin() + 2, idx.end(), 
                                      1));
  EXPECT_EQ(idx, std::vector<int>({0, 0, 1, 3}));
}
TEST(Util, Multi_Increment_Right_past_end) {
  using namespace cpp_nn::util;

  std::vector<int> shape{2,3,2,4};
  std::vector<int> idx{0, 0, 1, 2};
  

  EXPECT_FALSE(IncrementIndicesByShape(shape.begin() + 2, shape.end(), 
                                      idx.begin() + 2, idx.end(), 
                                      2));
  EXPECT_EQ(idx, std::vector<int>({0, 0, 0, 0}));
}
TEST(Util, Multi_Decrement_by_Neagtive) {
  using namespace cpp_nn::util;

  std::vector<int> shape{2,3,2,4};
  std::vector<int> idx{0, 0, 1, 2};
  

  EXPECT_TRUE(DecrementIndicesByShape(shape.begin(), shape.end(), 
                                      idx.begin(), idx.end(), 
                                      -4));
  EXPECT_EQ(idx, std::vector<int>({0, 1, 0, 2}));
}
// End of  MULTI INCREMENT TESTING ------------------------------------------------------------

// MULTI DECREMENT TESTING --------------------------------------------------------------------
TEST(Util, Multi_Decrement) {
  using namespace cpp_nn::util;

  std::vector<int> shape{2,3,1,2};
  std::vector<int> idx{1, 2, 0, 0}; // at 10th

  EXPECT_TRUE(DecrementIndicesByShape(shape.begin(), shape.end(),
                                      idx.begin(), idx.end(),
                                      8));

  EXPECT_EQ(idx, std::vector<int>({0, 1, 0, 0}));
}
TEST(Util, Multi_Decrement_zero) {
  using namespace cpp_nn::util;

  std::vector<int> shape{2,3,1,2};
  std::vector<int> idx{1, 2, 0, 0}; // at 10th

  EXPECT_TRUE(DecrementIndicesByShape(shape.begin(), shape.end(),
                                      idx.begin(), idx.end(),
                                      0));

  EXPECT_EQ(idx, std::vector<int>({1, 2, 0, 0}));
}
TEST(Util, Multi_Decrement_Right_to_front) {
  using namespace cpp_nn::util;

  std::vector<int> shape{2,3,1,2};
  std::vector<int> idx{1, 2, 0, 0}; // at 10th

  EXPECT_TRUE(DecrementIndicesByShape(shape.begin(), shape.end(),
                                      idx.begin(), idx.end(),
                                      10));

  EXPECT_EQ(idx, std::vector<int>({0, 0, 0, 0}));
}
TEST(Util, Multi_Decrement_Right_past_front) {
  using namespace cpp_nn::util;

  std::vector<int> shape{2,3,1,2};
  std::vector<int> idx{1, 2, 0, 0}; // at 10th

  EXPECT_FALSE(DecrementIndicesByShape(shape.begin(), shape.end(),
                                      idx.begin(), idx.end(),
                                      11));

  EXPECT_EQ(idx, std::vector<int>({1, 2, 0, 1}));
}
TEST(Util, Multi_Decrement_Underflow) {
  using namespace cpp_nn::util;

  std::vector<int> shape{2,3,1,2};
  std::vector<int> idx{1, 2, 0, 0}; // at 10th

  EXPECT_FALSE(DecrementIndicesByShape(shape.begin(), shape.end(),
                                      idx.begin(), idx.end(),
                                      20));

  EXPECT_EQ(idx, std::vector<int>({1, 2, 0, 1}));
}
TEST(Util, Multi_Decrement_Underflow2) {
  using namespace cpp_nn::util;

  std::vector<int> shape{2,3,1,2};
  std::vector<int> idx{1, 2, 0, 0}; // at 10th

  EXPECT_FALSE(DecrementIndicesByShape(shape.begin(), shape.end(),
                                      idx.begin(), idx.end(),
                                      200));

  EXPECT_EQ(idx, std::vector<int>({1, 2, 0, 1}));
}
TEST(Util, Multi_Increment_by_negative) {
  using namespace cpp_nn::util;

  std::vector<int> shape{2,3,1,2};
  std::vector<int> idx{1, 2, 0, 0}; // at 10th

  EXPECT_TRUE(IncrementIndicesByShape(shape.begin(), shape.end(),
                                      idx.begin(), idx.end(),
                                      -5));

  EXPECT_EQ(idx, std::vector<int>({0, 2, 0, 1}));
}
// End of  MULTI DECREMENT TESTING ------------------------------------------------------------

// BROADCAST TESTS ----------------------------------------------------------------------------
TEST(Util, Broadcast) {
  //                             |           |   
  std::vector<int> a =    {4, 3, 1, 1, 7, 8, 5, 3, 4, 1, 2};
  std::vector<int> b = {5, 4, 3, 1, 6, 1, 8, 5, 3};

  std::vector<int> expected = {1, 6, 7, 8};

  std::vector<int> res = cpp_nn::util::Broadcast(a.begin() + 2, a.end() - 5, 
                                                 b.begin() + 3, b.end() - 2);
  
  EXPECT_EQ(res, expected);
}

TEST(Util, Broadcast_Diff_Order) {
  //                       |                 |   
  std::vector<int> a =    {4, 3, 1, 1, 7, 8, 5, 3, 4, 1, 2};
  //                             |           |   
  std::vector<int> b = {5, 4, 3, 1, 6, 1, 8, 5, 3};

  std::vector<int> expected = {4, 3, 1, 6, 7, 8};

  std::vector<int> res = cpp_nn::util::Broadcast(a.begin()    , a.end() - 5, 
                                                 b.begin() + 3, b.end() - 2);
  
  EXPECT_EQ(res, expected);
}


TEST(Util, Broadcast_Diff_Incomp) {
  //                       |                 |   
  std::vector<int> a =    {4, 3, 1, 1, 7, 8, 5, 3, 4, 1, 2};
  //                             |           |   
  std::vector<int> b = {5, 4, 3, 1, 6, 3, 8, 5, 3};

  EXPECT_THROW({
    try
    {
      cpp_nn::util::Broadcast(a.begin()    , a.end() - 5, 
                                                 b.begin() + 3, b.end() - 2);
    }
    catch( const std::invalid_argument& e )
    {
      // and this tests that it has the correct message
      EXPECT_STREQ( "Broadcast- Incompatible shapes", e.what() );
      throw;
    }
  }, std::invalid_argument);
}


TEST(Util, Broadcast_One_is_empty) {
  std::vector<int> a = {};
  std::vector<int> b = {2, 1};

  std::vector<int> res = cpp_nn::util::Broadcast(a.begin(), a.end(), 
                                                 b.begin(), b.end());
  EXPECT_EQ(res, b);
}

TEST(Util, Broadcast_both_are_empty) {
  std::vector<int> a = {};
  std::vector<int> b = {};

  std::vector<int> res = cpp_nn::util::Broadcast(a.begin(), a.end(), 
                                                 b.begin(), b.end());
  EXPECT_EQ(res, b);
}

// CutToShape ------------------------------------------
TEST(Util, Broadcast_cut_idx_to_shape) {
  using namespace cpp_nn::util;

  std::vector<int> shape = {3, 2, 1, 1};
  std::vector<int> idx   = {2, 1, 2, 0};

  std::vector<int> cut_idx = CutToShape(idx, shape);

  EXPECT_EQ(cut_idx, std::vector<int>({2, 1, 0, 0}));
}

TEST(Util, Broadcast_cut_idx_to_shape_From_higher_order) {
  using namespace cpp_nn::util;

  std::vector<int> shape = {3, 2, 1, 1};
  std::vector<int> idx   = {2, 1, 2, 1, 2, 0};

  std::vector<int> cut_idx = CutToShape(idx, shape);

  EXPECT_EQ(cut_idx, std::vector<int>({2, 1, 0, 0}));
}

TEST(Util, Broadcast_cut_idx_to_shape_to_all_1_shape) {
  using namespace cpp_nn::util;

  std::vector<int> shape = {1, 1, 1, 1};
  std::vector<int> idx   = {2, 1, 2, 1, 2, 0};

  std::vector<int> cut_idx = CutToShape(idx, shape);

  EXPECT_EQ(cut_idx, std::vector<int>({0, 0, 0, 0}));
}

TEST(Util, Broadcast_cut_idx_to_shape_to_1_order_shape) {
  using namespace cpp_nn::util;

  std::vector<int> shape = {8};
  std::vector<int> idx   = {2, 1, 2, 1, 2, 2};

  std::vector<int> cut_idx = CutToShape(idx, shape);

  EXPECT_EQ(cut_idx, std::vector<int>({2}));
}
// End of CutToShape -----------------------------------
// END OF BROADCAST TESTS ---------------------------------------------------------------------
