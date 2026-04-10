#include "MonotonicMax.hpp"
#include "MultisetMedian.hpp"
#include "SlidingWelford.hpp"
#include "SlidingWelfordRing.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(MultisetMedianTest, HandlesInitialization) {
  MultisetMedian sm(3);
  EXPECT_EQ(sm.current_size(), 0);

  EXPECT_THROW(sm.get_median(), std::logic_error);
}

TEST(MultisetMedianTest, HandlesWarmUpPhase) {
  MultisetMedian sm(3);

  sm.update(10.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 10.0);

  sm.update(20.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 15.0);

  sm.update(5.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 10.0); //  5, 10, 20
}

TEST(MultisetMedianTest, HandlesWindowShifting) {
  MultisetMedian sm(3);
  sm.update(1.0);
  sm.update(2.0);
  sm.update(3.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 2.0);

  sm.update(4.0); // 2, 3, 4
  EXPECT_DOUBLE_EQ(sm.get_median(), 3.0);

  sm.update(0.0); // 3, 4, 0 -> 0, 3, 4
  EXPECT_DOUBLE_EQ(sm.get_median(), 3.0);
}

TEST(MultisetMedianTest, HandlesDuplicateValues) {
  MultisetMedian sm(4);
  sm.update(5.0);
  sm.update(5.0);
  sm.update(5.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 5.0);

  sm.update(5.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 5.0);
}

TEST(MonotonicMaxTest, BasicFunctionality) {
  MonotonicMax mm(3);

  mm.update(10.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 10.0);

  mm.update(20.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 20.0);

  mm.update(5.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 20.0); // [10, 20, 5]
}

TEST(MonotonicMaxTest, MaxExitsWindow) {
  MonotonicMax mm(3);
  mm.update(50.0);
  mm.update(10.0);
  mm.update(20.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 50.0);

  mm.update(5.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 20.0);
}

TEST(MonotonicMaxTest, StrictlyDecreasing) {
  MonotonicMax mm(3);
  mm.update(100.0);
  mm.update(90.0);
  mm.update(80.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 100.0);

  mm.update(70.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 90.0);
}

TEST(MonotonicMaxTest, Duplicates) {
  MonotonicMax mm(2);
  mm.update(10.0);
  mm.update(10.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 10.0);

  mm.update(5.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 10.0);
}

TEST(SlidingWelfordTest, ConstantValues) {
  SlidingWelford sw(5);
  for (int i = 0; i < 10; ++i) {
    sw.update(100.0);
  }
  EXPECT_DOUBLE_EQ(sw.get_mean(), 100.0);
  EXPECT_NEAR(sw.get_std_dev(), 0.0, 1e-9);
}

TEST(SlidingWelfordTest, KnownValues) {
  SlidingWelford sw(2); // window size 2

  sw.update(10.0);
  sw.update(20.0);

  // mean: 15
  EXPECT_DOUBLE_EQ(sw.get_mean(), 15.0);
  // Variance: ((10-15)^2 + (20-15)^2) / 1 = 50
  // std dev: sqrt(50) = 7.0710678...
  EXPECT_NEAR(sw.get_std_dev(), 7.071067811865475, 1e-9);
}

TEST(SlidingWelfordTest, WindowShift) {
  SlidingWelford sw(3);

  sw.update(10.0);
  sw.update(10.0);
  sw.update(10.0);
  EXPECT_DOUBLE_EQ(sw.get_std_dev(), 0.0);

  // [10, 10, 100]
  sw.update(100.0);

  // Mean: 40
  // M2: (10-40)^2 + (10-40)^2 + (100-40)^2 = 900 + 900 + 3600 = 5400
  // Var: 5400 / 2 = 2700
  EXPECT_DOUBLE_EQ(sw.get_mean(), 40.0);
  EXPECT_DOUBLE_EQ(sw.get_variance(), 2700.0);
}

TEST(SlidingWelfordRingTest, WarmUpPhase) {
  SlidingWelfordRing sw(3);

  sw.update(10.0);
  EXPECT_DOUBLE_EQ(sw.get_mean(), 10.0);
  EXPECT_TRUE(std::isnan(sw.get_variance()));

  sw.update(20.0);
  // Mean: 15, Variance: ((10-15)^2 + (20-15)^2) / 1 = 50
  EXPECT_DOUBLE_EQ(sw.get_mean(), 15.0);
  EXPECT_DOUBLE_EQ(sw.get_variance(), 50.0);
}

TEST(SlidingWelfordRingTest, ConstantValues) {
  SlidingWelfordRing sw(10);
  for (int i = 0; i < 20; ++i) {
    sw.update(5.0);
  }
  EXPECT_DOUBLE_EQ(sw.get_mean(), 5.0);
  EXPECT_NEAR(sw.get_std_dev(), 0.0, 1e-9);
}

TEST(SlidingWelfordRingTest, BufferWrapAround) {
  SlidingWelfordRing sw(2);

  sw.update(10.0);
  sw.update(20.0); // [10, 20]

  sw.update(30.0); // [20, 30]
  EXPECT_DOUBLE_EQ(sw.get_mean(), 25.0);
  EXPECT_DOUBLE_EQ(sw.get_variance(), 50.0);

  sw.update(40.0); // [30, 40]
  EXPECT_DOUBLE_EQ(sw.get_mean(), 35.0);
  EXPECT_DOUBLE_EQ(sw.get_variance(), 50.0);
}

TEST(SlidingWelfordRingTest, PrecisionTest) {
  SlidingWelfordRing sw(3);
  sw.update(1000000.1);
  sw.update(1000000.2);
  sw.update(1000000.3);

  EXPECT_NEAR(sw.get_mean(), 1000000.2, 1e-7);
  EXPECT_NEAR(sw.get_variance(), 0.01, 1e-9);
}