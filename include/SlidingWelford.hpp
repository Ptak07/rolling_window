#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <queue>

class SlidingWelford {
public:
  explicit SlidingWelford(std::size_t window_size);

  void update(double value);
  double get_mean() const;
  double get_std_dev() const;
  double get_variance() const;
  std::size_t current_size() const;

private:
  std::size_t window_size_;

  double mean_{0.0};
  double M2_{00};

  std::queue<double> window_;
};