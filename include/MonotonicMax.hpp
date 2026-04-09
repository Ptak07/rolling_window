#pragma once

#include <deque>
#include <cstddef>
#include <limits>

class MonotonicMax { 
public: 
    explicit MonotonicMax(std::size_t window_size); 

    void update(double value); 
    double get_max() const; 
    std::size_t current_size() const; 

private:
    struct Element { 
        double value;
        std::size_t tick_index;
    };

    std::size_t window_size_; 
    std::size_t current_tick_; 
    std::deque<Element> deque_;
};