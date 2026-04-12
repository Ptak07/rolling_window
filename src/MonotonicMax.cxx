#include "MonotonicMax.hpp"

MonotonicMax::MonotonicMax(std::size_t window_size)
    : window_size_(window_size), current_tick_(0) {
}

double MonotonicMax::get_value_impl() const {
  if (deque_.empty())
    return std::numeric_limits<double>::quiet_NaN();
  return deque_.front().value;
}

std::size_t MonotonicMax::current_size_impl() const {
  return std::min(current_tick_, window_size_);
}

double MonotonicMax::get_max() const {
  return get_value();
}

void MonotonicMax::update_impl(double value) {
  while (!deque_.empty() && deque_.back().value < value) {
    deque_.pop_back();
  }
  deque_.push_back({value, current_tick_});

  while (!deque_.empty() &&
         current_tick_ - deque_.front().tick_index >= window_size_) {
    deque_.pop_front();
  }
  current_tick_++;
}