#pragma once 
#include <set> 
#include <queue> 
#include <stdexcept> 

class SlidingMedian { 
public: 
    explicit SlidingMedian(int window_size); 

    void update(double new_value);

    double get_median() const;
    
    int current_size() const; 
private: 
    int window_size; 
    std::multiset<double> window_data; 
    std::multiset<double>::iterator mid; 
    std::queue<double> history; 

    bool is_even() const; 
};