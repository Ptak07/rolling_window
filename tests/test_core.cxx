#include <gtest/gtest.h> 
#include "MultisetMedian.hpp"
#include "MonotonicMax.hpp"

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