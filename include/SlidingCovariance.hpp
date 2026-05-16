#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

class SlidingCovariance {
public:
  explicit SlidingCovariance(std::size_t window_size);
  void update(double x, double y);
  std::size_t current_size() const;
  double get_covariance() const;
  double get_correlation() const;
  double get_mean_x() const;
  double get_mean_y() const;

private:
  std::size_t window_size_;
  std::vector<double> buffer_x_;
  std::vector<double> buffer_y_;
  std::size_t head_{0};
  std::size_t n_written_{0};
  std::size_t count_{0};
  double mean_x_{0.0};
  double mean_y_{0.0};
  double M2_x_{0.0};
  double M2_y_{0.0};
  double C_{0.0};
};