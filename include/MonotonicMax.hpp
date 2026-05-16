#pragma once

#include "RollingMetric.hpp"

#include <cstddef>
#include <deque>
#include <limits>

class MonotonicMax : public RollingMetric<MonotonicMax> {
  friend class RollingMetric<MonotonicMax>;

public:
  explicit MonotonicMax(std::size_t window_size);

  double get_max() const;

private:
  void update_impl(double value);
  void skip_impl();
  double get_value_impl() const;
  std::size_t current_size_impl() const;

  struct Element {
    double value;
    std::size_t tick_index;
  };

  std::size_t window_size_;
  std::size_t current_tick_;
  std::deque<Element> deque_;
  std::deque<std::size_t> non_nan_ticks_;
};
