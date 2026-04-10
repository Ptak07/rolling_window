#pragma once 

#include <queue> 
#include <cstddef> 
#include <limits> 
#include <algorithm> 

class SlidingWelford { 
public: 
    explicit SlidingWelford(std::size_t window_size); 

    void update(double value); 
    double get_variance() const; 
    double get_mean() const; 
    double get_std_dev() const;
    std::size_t current_size() const; 

private: 
    std::size_t window_size_; 

    double mean_{0.0}; 
    double M2_{00}; 

    std::queue<double> window_;
};