#include "MultisetMedian.hpp"
#include <iterator>

MultisetMedian::MultisetMedian(std::size_t size) : window_size_(size) {
  if (size <= 0) {
    throw std::invalid_argument("Window length must be greater than 0");
  }
}

bool MultisetMedian::is_even() const {
  return window_data_.size() % 2 == 0;
}

std::size_t MultisetMedian::current_size() const {
  return window_data_.size();
}

double MultisetMedian::get_median() const {
  if (window_data_.empty()) {
    throw std::logic_error("No data to calculate median.");
  }

  if (is_even()) {
    auto prev_mid_ = std::prev(mid_);
    return (*mid_ + *prev_mid_) / 2;
  } else {
    return *mid_;
  }
}

void MultisetMedian::update(double new_value) {
  if (history_.size() == window_size_) {
    double oldest_val = history_.front();
    history_.pop();

    auto it_to_remove = window_data_.find(oldest_val);

    if (it_to_remove == mid_ || *it_to_remove < *mid_) {
      mid_++;
    }

    window_data_.erase(it_to_remove);
  }

  history_.push(new_value);
  window_data_.insert(new_value);

  if (window_data_.size() == 1)
    mid_ = window_data_.begin();
  else {
    if (new_value < *mid_)
      mid_--;
    if (history_.size() < window_size_ && is_even()) {
      mid_++;
    }
  }
}