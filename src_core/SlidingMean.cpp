#include "SlidingMean.hpp"
#include <stdexcept>

SlidingMean::SlidingMean(std::size_t window_size) : window_size_(window_size) {
  if (window_size_ == 0)
    throw std::invalid_argument("Window length must be greater than 0");
  buffer_.resize(window_size_, std::numeric_limits<double>::quiet_NaN());
}

void SlidingMean::update_impl(double value) {
  bool ring_full = (n_written_ >= window_size_);
  if (ring_full) {
    double x_out = buffer_[head_];
    if (!std::isnan(x_out)) {
      sum_ -= x_out;
      --count_;
    }
  }
  buffer_[head_] = value;
  head_ = (head_ + 1) % window_size_;
  if (!ring_full)
    ++n_written_;
  sum_ += value;
  ++count_;
}

void SlidingMean::skip_impl() {
  bool ring_full = (n_written_ >= window_size_);
  if (ring_full) {
    double x_out = buffer_[head_];
    if (!std::isnan(x_out)) {
      sum_ -= x_out;
      --count_;
    }
  }
  buffer_[head_] = std::numeric_limits<double>::quiet_NaN();
  head_ = (head_ + 1) % window_size_;
  if (!ring_full)
    ++n_written_;
}

double SlidingMean::get_value_impl() const { return get_mean(); }

std::size_t SlidingMean::current_size_impl() const { return count_; }

double SlidingMean::get_mean() const {
  if (count_ == 0)
    return std::numeric_limits<double>::quiet_NaN();
  return sum_ / static_cast<double>(count_);
}
