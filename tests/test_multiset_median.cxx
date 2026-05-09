#include "MultisetMedian.hpp"

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

TEST(MultisetMedianTest, CrtpInterfaceMatchesMedian) {
  MultisetMedian sm(3);
  RollingMetric<MultisetMedian> &base = sm;

  EXPECT_THROW(base.get_value(), std::logic_error);

  base.update(7.0);
  base.update(1.0);
  base.update(4.0);

  EXPECT_EQ(base.current_size(), 3U);
  EXPECT_DOUBLE_EQ(base.get_value(), sm.get_median());
}

TEST(MultisetMedianTest, HandlesEvenWindowDescendingFill) {
  MultisetMedian sm(4);
  sm.update(4.0);
  sm.update(3.0);
  sm.update(2.0);
  sm.update(1.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 2.5);
}

TEST(MultisetMedianTest, HandlesWindowSize2Sliding) {
  MultisetMedian sm(2);
  sm.update(1.0);
  sm.update(2.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 1.5);
  sm.update(3.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 2.5);
}