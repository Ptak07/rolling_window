#pragma once 

#include <vector>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <stdexcept>

class SlidingWelfordRing { 
public: 
    explicit SlidingWelfordRing(std::size_t window_size);

    void update(double value);
    double get_mean() const;
    double get_std_dev() const;
    double get_variance() const;
    std::size_t current_size() const;

private:
    std::size_t window_size_;
    
    double mean_{0.0};
    double M2_{0.0};

    std::vector<double> buffer_; 
    std::size_t head_{0};        
    std::size_t count_{0};
};