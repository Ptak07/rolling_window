#pragma once

#include "RollingMetric.hpp"

#include <cstddef>
#include <deque>
#include <limits>

class MonotonicMax : public RollingMetric<MonotonicMax> {
public:
  explicit MonotonicMax(std::size_t window_size);

  void update_impl(double value);
  double get_value_impl() const;
  std::size_t current_size_impl() const;

  double get_max() const;

private:
  struct Element {
    double value;
    std::size_t tick_index;
  };

  std::size_t window_size_;
  std::size_t current_tick_;
  std::deque<Element> deque_;
};