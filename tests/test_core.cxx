#include <gtest/gtest.h> 
#include "SlidingMedian.hpp"

TEST(SlidingMedianTest, HandlesInitialization) {
    SlidingMedian sm(3);
    EXPECT_EQ(sm.current_size(), 0);
    
    EXPECT_THROW(sm.get_median(), std::logic_error);
}

TEST(SlidingMedianTest, HandlesWarmUpPhase) {
    SlidingMedian sm(3);
    
    sm.update(10.0);
    EXPECT_DOUBLE_EQ(sm.get_median(), 10.0);
    
    sm.update(20.0);
    EXPECT_DOUBLE_EQ(sm.get_median(), 15.0); 
    
    sm.update(5.0);
    EXPECT_DOUBLE_EQ(sm.get_median(), 10.0); //  5, 10, 20
}

TEST(SlidingMedianTest, HandlesWindowShifting) {
    SlidingMedian sm(3);
    sm.update(1.0);
    sm.update(2.0);
    sm.update(3.0);
    EXPECT_DOUBLE_EQ(sm.get_median(), 2.0);
    
    sm.update(4.0); // 2, 3, 4
    EXPECT_DOUBLE_EQ(sm.get_median(), 3.0);
    
    sm.update(0.0); // 3, 4, 0 -> 0, 3, 4
    EXPECT_DOUBLE_EQ(sm.get_median(), 3.0);
}

TEST(SlidingMedianTest, HandlesDuplicateValues) {
    SlidingMedian sm(4);
    sm.update(5.0);
    sm.update(5.0);
    sm.update(5.0);
    EXPECT_DOUBLE_EQ(sm.get_median(), 5.0);
    
    sm.update(5.0);
    EXPECT_DOUBLE_EQ(sm.get_median(), 5.0);
}