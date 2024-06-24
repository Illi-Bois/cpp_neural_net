#include "gtest/gtest.h"

#include <list>

TEST(SanityCheck, SanityCheck) {
  EXPECT_TRUE(true);
}

TEST(SanityCheck, SanityCheckExceptions) {
  EXPECT_THROW({
    try
    {
      throw std::invalid_argument("SanityCheck");
    }
    catch( const std::invalid_argument& e )
    {
      // and this tests that it has the correct message
      EXPECT_STREQ( "SanityCheck", e.what() );
      throw;
    }
  }, std::invalid_argument);
}
