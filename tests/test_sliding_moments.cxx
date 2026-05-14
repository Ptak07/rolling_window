#include "SlidingMoments.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(SlidingMomentsTest, HandlesInitialization) {
  SlidingMoments sm(4);
  EXPECT_EQ(sm.current_size(), 0);
  EXPECT_TRUE(std::isnan(sm.get_skewness()));
  EXPECT_TRUE(std::isnan(sm.get_kurtosis()));
}

TEST(SlidingMomentsTest, SymmetricWindowSkewnessIsZero) {
  // [1, 2, 3] is symmetric → adjusted Fisher-Pearson skewness = 0
  SlidingMoments sm(3);
  sm.update(1.0);
  sm.update(2.0);
  sm.update(3.0);
  EXPECT_NEAR(sm.get_skewness(), 0.0, 1e-12);
  EXPECT_TRUE(std::isnan(sm.get_kurtosis()));  // n=3 < 4
}

TEST(SlidingMomentsTest, KnownKurtosisUniformWindow) {
  // [1, 2, 3, 4]: excess kurtosis = -1.2 (uniform distribution)
  SlidingMoments sm(4);
  sm.update(1.0);
  sm.update(2.0);
  sm.update(3.0);
  sm.update(4.0);
  EXPECT_NEAR(sm.get_kurtosis(), -1.2, 1e-10);
}

TEST(SlidingMomentsTest, NanAdvancesWindowAndReducesSize) {
  SlidingMoments sm(4);
  sm.update(1.0);
  sm.update(2.0);
  sm.update(3.0);
  sm.update(4.0);
  EXPECT_EQ(sm.current_size(), 4);
  EXPECT_FALSE(std::isnan(sm.get_kurtosis()));

  sm.update(NAN);  // window: [2, 3, 4, NaN] → 3 non-NaN
  EXPECT_EQ(sm.current_size(), 3);
  EXPECT_FALSE(std::isnan(sm.get_skewness()));
  EXPECT_TRUE(std::isnan(sm.get_kurtosis()));  // n=3 < 4

  sm.update(NAN);  // window: [3, 4, NaN, NaN] → 2 non-NaN
  EXPECT_EQ(sm.current_size(), 2);
  EXPECT_TRUE(std::isnan(sm.get_skewness()));  // n=2 < 3
}

TEST(SlidingMomentsTest, NanDoesNotCorruptMathematicalState) {
  SlidingMoments sm(3);
  sm.update(10.0);
  sm.update(20.0);
  sm.update(30.0);
  double skew_before = sm.get_skewness();

  sm.update(NAN);
  sm.update(NAN);
  sm.update(NAN);
  EXPECT_EQ(sm.current_size(), 0);

  sm.update(10.0);
  sm.update(20.0);
  sm.update(30.0);
  EXPECT_NEAR(sm.get_skewness(), skew_before, 1e-10);
}