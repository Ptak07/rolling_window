#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

class SlidingMoments {
public:
  explicit SlidingMoments(std::size_t window_size);
  void update(double x);
  void reset();
  std::size_t current_size() const;
  double get_skewness() const; // adjusted Fisher-Pearson
  double get_kurtosis() const; // excess kurtosis (Fisher)
  double get_mean() const;

private:
  std::size_t window_size_;
  std::vector<double> buffer_;
  std::size_t head_{0};
  std::size_t n_written_{0};
  std::size_t count_{0}; // non-NaN int new window
  double mean_{0.0};
  double M2_{0.0};
  double M3_{0.0};
  double M4_{0.0};
};
