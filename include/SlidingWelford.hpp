#pragma once

#include "RollingMetric.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <queue>

class SlidingWelford : public RollingMetric<SlidingWelford> {
  friend class RollingMetric<SlidingWelford>;

public:
  explicit SlidingWelford(std::size_t window_size);

  double get_mean() const;
  double get_std_dev() const;
  double get_variance() const;

private:
  void update_impl(double value);
  double get_value_impl() const;
  std::size_t current_size_impl() const;

  std::size_t window_size_;

  double mean_{0.0};
  double M2_{00};

  std::queue<double> window_;
};