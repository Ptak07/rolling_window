#include "MonotonicMin.hpp"
#include <stdexcept>

MonotonicMin::MonotonicMin(std::size_t window_size)
    : window_size_(window_size), current_tick_(0) {
  if (window_size_ == 0) {
    throw std::invalid_argument("Window length must be greater than 0");
  }
}

double MonotonicMin::get_value_impl() const {
  if (deque_.empty())
    return std::numeric_limits<double>::quiet_NaN();
  return deque_.front().value;
}

std::size_t MonotonicMin::current_size_impl() const {
  return non_nan_ticks_.size();
}

double MonotonicMin::get_min() const {
  return get_value();
}

void MonotonicMin::skip_impl() {
  while (!deque_.empty() &&
         current_tick_ - deque_.front().tick_index >= window_size_) {
    deque_.pop_front();
  }
  while (!non_nan_ticks_.empty() &&
         current_tick_ - non_nan_ticks_.front() >= window_size_) {
    non_nan_ticks_.pop_front();
  }
  current_tick_++;
}

void MonotonicMin::update_impl(double value) {
  while (!deque_.empty() && deque_.back().value > value) {
    deque_.pop_back();
  }
  deque_.push_back({value, current_tick_});

  while (!deque_.empty() &&
         current_tick_ - deque_.front().tick_index >= window_size_) {
    deque_.pop_front();
  }
  while (!non_nan_ticks_.empty() &&
         current_tick_ - non_nan_ticks_.front() >= window_size_) {
    non_nan_ticks_.pop_front();
  }
  non_nan_ticks_.push_back(current_tick_);
  current_tick_++;
}
