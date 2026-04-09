#include "SlidingMedian.hpp"
#include <iterator>

SlidingMedian::SlidingMedian(int size) : window_size(size) { 
    if (size <= 0) { 
        throw std::invalid_argument("Window length must be greater than 0");
    }
}

bool SlidingMedian::is_even() const { 
    return window_data.size() % 2 == 0; 
}

int SlidingMedian::current_size() const { 
    return window_data.size();
}

double SlidingMedian::get_median() const { 
    if (window_data.empty()) { 
        throw std::logic_error("No data to calculate median.");
    }

    if (is_even()) { 
        auto prev_mid = std::prev(mid); 
        return (*mid + *prev_mid) / 2; 
    } else { 
        return *mid;
    }
}

void SlidingMedian::update(double new_value) { 
    if (history.size() == static_cast<size_t>(window_size)) { 
        double oldest_val = history.front(); 
        history.pop(); 

        auto it_to_remove = window_data.find(oldest_val); 

        if (it_to_remove == mid || *it_to_remove < *mid) { 
            mid++;
        }

        window_data.erase(it_to_remove);
    }

    history.push(new_value); 
    window_data.insert(new_value);

    if (window_data.size() == 1)
        mid = window_data.begin(); 
    else { 
        if (new_value < *mid) 
            mid--; 
        if (history.size() < static_cast<size_t>(window_size) && is_even()) { 
            mid++;
        }
    }
}