#include "SlidingMoments.hpp"

SlidingMoments::SlidingMoments(std::size_t window_size)
    : window_size_(window_size) {
  if (window_size_ == 0)
    throw std::invalid_argument("Window length must be greater than 0");

  buffer_.resize(window_size, std::numeric_limits<double>::quiet_NaN());
}

static void moments_add(double x, std::size_t &count_, double &mean_,
                        double &M2_, double &M3_, double &M4_) {
  count_++;
  double n = static_cast<double>(count_);
  double delta = x - mean_;
  double delta_n = delta / n;
  double term = delta * delta_n * (n - 1);

  M4_ += term * delta_n * delta_n * (n * n - 3 * n + 3) +
         6 * delta_n * delta_n * M2_ - 4 * delta_n * M3_;

  M3_ += term * delta_n * (n - 2) - 3 * delta_n * M2_;
  M2_ += term;
  mean_ += delta_n;
}

static void moments_remove(double x_out, std::size_t &count_, double &mean_,
                           double &M2_, double &M3_, double &M4_) {
  if (count_ == 1) {
    mean_ = 0.0;
    M2_ = 0.0;
    M3_ = 0.0;
    M4_ = 0.0;
    count_ = 0;
    return;
  }

  double n = static_cast<double>(count_);
  double mean_new = (n * mean_ - x_out) / (n - 1);
  double delta = x_out - mean_new;
  double delta_n = delta / n;
  double term = delta * delta_n * (n - 1);
  M4_ -= term * delta_n * delta_n * (n * n - 3 * n + 3) +
         6 * delta_n * delta_n * M2_ - 4 * delta_n * M3_;
  M3_ -= term * delta_n * (n - 2) - 3 * delta_n * M2_;
  M2_ -= term;
  M2_ = std::max(0.0, M2_);
  mean_ = mean_new;
  count_--;
}

size_t SlidingMoments::current_size() const {
  return count_;
}

void SlidingMoments::reset() {
  buffer_.assign(window_size_, std::numeric_limits<double>::quiet_NaN());

  head_ = {0};
  n_written_ = {0};
  count_ = {0};
  mean_ = {0.0};
  M2_ = {0.0};
  M3_ = {0.0};
  M4_ = {0.0};
}

double SlidingMoments::get_skewness() const {
  if (count_ < 3)
    return std::numeric_limits<double>::quiet_NaN();
  if (M2_ <= 0.0)
    return std::numeric_limits<double>::quiet_NaN();
  double n = static_cast<double>(count_);
  return M3_ * n * std::sqrt(n - 1.0) / (std::pow(M2_, 1.5) * (n - 2.0));
}

double SlidingMoments::get_kurtosis() const {
  if (count_ < 4)
    return std::numeric_limits<double>::quiet_NaN();
  if (M2_ <= 0.0)
    return std::numeric_limits<double>::quiet_NaN();
  double n = static_cast<double>(count_);
  double g2 = M4_ * n / (M2_ * M2_) - 3.0;
  return (n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * g2 + 6.0);
}

void SlidingMoments::update(double value) {
  bool ring_full = (n_written_ >= window_size_);

  if (ring_full) {
    double x_out = buffer_[head_];
    if (!std::isnan(x_out)) {
      moments_remove(x_out, count_, mean_, M2_, M3_, M4_);
    }
  }

  buffer_[head_] = value;
  head_++;
  if (head_ == window_size_)
    head_ = 0;

  if (!ring_full)
    n_written_++;

  if (!std::isnan(value))
    moments_add(value, count_, mean_, M2_, M3_, M4_);
}
