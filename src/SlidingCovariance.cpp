#include "SlidingCovariance.hpp"
#include <stdexcept>

SlidingCovariance::SlidingCovariance(std::size_t window_size)
    : window_size_(window_size) {
  if (window_size_ == 0)
    throw std::invalid_argument("Window length must be greater than 0");

  buffer_x_.resize(window_size, std::numeric_limits<double>::quiet_NaN());
  buffer_y_.resize(window_size, std::numeric_limits<double>::quiet_NaN());
}

static void cov_add(double x, double y, std::size_t &count_, double &mean_x_,
                    double &mean_y_, double &C_, double &M2_x_, double &M2_y_) {
  count_++;
  double n = static_cast<double>(count_);
  double delta_x = x - mean_x_;
  double delta_y = y - mean_y_;

  mean_x_ += delta_x / n;
  mean_y_ += delta_y / n;

  double delta_x2 = x - mean_x_;
  double delta_y2 = y - mean_y_;

  C_ += delta_x * delta_y2;
  M2_x_ += delta_x * delta_x2;
  M2_y_ += delta_y * delta_y2;
}

static void cov_remove(double x_out, double y_out, std::size_t &count_,
                       double &mean_x_, double &mean_y_, double &C_,
                       double &M2_x_, double &M2_y_) {
  if (count_ == 1) {
    mean_x_ = 0.0;
    mean_y_ = 0.0;
    C_ = 0.0;
    M2_x_ = 0.0;
    M2_y_ = 0.0;
    count_ = 0;
    return;
  }

  double n = static_cast<double>(count_);
  double mean_x_new = (n * mean_x_ - x_out) / (n - 1);
  double mean_y_new = (n * mean_y_ - y_out) / (n - 1);

  double dx = x_out - mean_x_new;
  double dy = y_out - mean_y_new;

  C_ -= dx * (y_out - mean_y_);
  M2_x_ -= dx * (x_out - mean_x_);
  M2_y_ -= dy * (y_out - mean_y_);
  M2_x_ = std::max(0.0, M2_x_);
  M2_y_ = std::max(0.0, M2_y_);

  mean_x_ = mean_x_new;
  mean_y_ = mean_y_new;
  count_--;
}

std::size_t SlidingCovariance::current_size() const { return count_; }

double SlidingCovariance::get_mean_x() const {
  if (count_ == 0)
    return std::numeric_limits<double>::quiet_NaN();
  return mean_x_;
}

double SlidingCovariance::get_mean_y() const {
  if (count_ == 0)
    return std::numeric_limits<double>::quiet_NaN();
  return mean_y_;
}

double SlidingCovariance::get_covariance() const {
  if (count_ < 2)
    return std::numeric_limits<double>::quiet_NaN();
  return C_ / static_cast<double>(count_ - 1);
}

double SlidingCovariance::get_correlation() const {
  if (count_ < 2 || M2_x_ <= 0.0 || M2_y_ <= 0.0)
    return std::numeric_limits<double>::quiet_NaN();
  return C_ / std::sqrt(M2_x_ * M2_y_);
}

void SlidingCovariance::update(double x, double y) {
  bool ring_full = (n_written_ >= window_size_);

  if (ring_full) {
    double x_out = buffer_x_[head_];
    double y_out = buffer_y_[head_];

    if (!std::isnan(x_out) && !std::isnan(y_out))
      cov_remove(x_out, y_out, count_, mean_x_, mean_y_, C_, M2_x_, M2_y_);
  }

  buffer_x_[head_] = x;
  buffer_y_[head_] = y;
  head_++;
  if (head_ == window_size_)
    head_ = 0;
  if (!ring_full)
    n_written_++;

  if (!std::isnan(x) && !std::isnan(y))
    cov_add(x, y, count_, mean_x_, mean_y_, C_, M2_x_, M2_y_);
}