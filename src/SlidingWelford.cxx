#include "SlidingWelford.hpp"

SlidingWelford::SlidingWelford(std::size_t window_size)
    : window_size_(window_size), mean_(0.0), M2_(0.0) {

  if (window_size_ == 0) {
    throw std::invalid_argument("Window length must be greater than 0");
  }
}

std::size_t SlidingWelford::current_size() const {
  return window_.size();
}

double SlidingWelford::get_mean() const {
  return mean_;
}

double SlidingWelford::get_variance() const {
  if (current_size() < 2) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::max(0.0, M2_) / (current_size() - 1);
}

double SlidingWelford::get_std_dev() const {
  return std::sqrt(get_variance());
}

void SlidingWelford::update(double value) {
  if (current_size() == window_size_) {
    double x_out = window_.front();
    window_.pop();

    if (window_size_ == 1) {
      mean_ = 0.0;
      M2_ = 0.0;
    } else {
      double delta_out = x_out - mean_;
      mean_ = mean_ - delta_out / (window_size_ - 1);
      M2_ -= delta_out * (x_out - mean_);
    }
  }
  window_.push(value);
  double delta = value - mean_;
  mean_ = mean_ + delta / current_size();
  M2_ = M2_ + delta * (value - mean_);
}