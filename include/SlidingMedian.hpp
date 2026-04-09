#pragma once 
#include <vector> 

class SlidingMedian { 
public: 
    explicit SlidingMedian(int window_size); 

    void update(double new_value);

    double get_median() const; 
private: 
    int window_size; 
};