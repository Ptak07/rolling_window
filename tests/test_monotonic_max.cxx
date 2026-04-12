#include "MonotonicMax.hpp"

#include <cmath>
#include <gtest/gtest.h>

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

TEST(MonotonicMaxTest, CrtpInterfaceMatchesMax) {
  MonotonicMax mm(3);
  RollingMetric<MonotonicMax> &base = mm;

  EXPECT_TRUE(std::isnan(base.get_value()));

  base.update(1.0);
  base.update(9.0);
  base.update(2.0);

  EXPECT_EQ(base.current_size(), 3U);
  EXPECT_DOUBLE_EQ(base.get_value(), mm.get_max());
}
