#pragma once

#include "RollingMetric.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

class SlidingWelfordRing : public RollingMetric<SlidingWelfordRing> {
  friend class RollingMetric<SlidingWelfordRing>;

public:
  explicit SlidingWelfordRing(std::size_t window_size);

  double get_mean() const;
  double get_std_dev() const;
  double get_variance() const;

private:
  void update_impl(double value);
  double get_value_impl() const;
  std::size_t current_size_impl() const;

  std::size_t window_size_;
  double mean_{0.0};
  double M2_{0.0};
  std::vector<double> buffer_;
  std::size_t head_{0};
  std::size_t count_{0};
};
