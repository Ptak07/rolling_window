#include "SlidingWelfordRing.hpp"

SlidingWelfordRing::SlidingWelfordRing(std::size_t window_size)
    : window_size_(window_size), mean_(0.0), M2_(0.0) {

  if (window_size_ == 0) {
    throw std::invalid_argument("Window length must be greater than 0");
  }

  buffer_.resize(window_size);
}

std::size_t SlidingWelfordRing::current_size_impl() const {
  return count_;
}

double SlidingWelfordRing::get_mean() const {
  return mean_;
}

double SlidingWelfordRing::get_value_impl() const {
  if (current_size() < 2) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::max(0.0, M2_) / (count_ - 1);
}

double SlidingWelfordRing::get_variance() const {
  return get_value();
}

double SlidingWelfordRing::get_std_dev() const {
  return std::sqrt(get_variance());
}

void SlidingWelfordRing::update_impl(double value) {
  if (count_ == window_size_) {
    if (window_size_ == 1) {
      mean_ = 0.0;
      M2_ = 0.0;
    } else {
      double x_out = buffer_[head_];
      double delta_out = x_out - mean_;
      mean_ -= delta_out / (window_size_ - 1);
      M2_ -= delta_out * (x_out - mean_);
    }
  } else {
    count_++;
  }
  buffer_[head_] = value;

  double delta = value - mean_;
  mean_ += delta / count_;
  M2_ += delta * (value - mean_);

  head_++;
  if (head_ == window_size_) {
    head_ = 0;
  }
}

