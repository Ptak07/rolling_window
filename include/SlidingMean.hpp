#pragma once
#include "RollingMetric.hpp"
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

class SlidingMean : public RollingMetric<SlidingMean> {
  friend class RollingMetric<SlidingMean>;

public:
  explicit SlidingMean(std::size_t window_size);
  double get_mean() const;

private:
  void update_impl(double value);
  void skip_impl();
  double get_value_impl() const;
  std::size_t current_size_impl() const;

  std::size_t window_size_;
  std::vector<double> buffer_;
  std::size_t head_{0};
  std::size_t n_written_{0};
  std::size_t count_{0};
  double sum_{0.0};
};
