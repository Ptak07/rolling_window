#include "SlidingWelford.hpp"

#include <cmath>
#include <gtest/gtest.h>

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

TEST(SlidingWelfordTest, CrtpInterfaceMatchesVariance) {
  SlidingWelford sw(3);
  RollingMetric<SlidingWelford> &base = sw;

  base.update(10.0);
  EXPECT_TRUE(std::isnan(base.get_value()));

  base.update(20.0);
  EXPECT_EQ(base.current_size(), 2U);
  EXPECT_DOUBLE_EQ(base.get_value(), sw.get_variance());
}
