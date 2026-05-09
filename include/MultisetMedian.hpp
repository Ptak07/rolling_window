#pragma once
#include "RollingMetric.hpp"
#include <queue>
#include <set>
#include <stdexcept>

class MultisetMedian : public RollingMetric<MultisetMedian> {
  friend class RollingMetric<MultisetMedian>;

public:
  explicit MultisetMedian(std::size_t window_size);

  double get_median() const;

private:
  void update_impl(double new_value);
  void skip_impl();
  double get_value_impl() const;
  std::size_t current_size_impl() const;

  void recompute_mid();

  std::size_t window_size_;
  std::multiset<double> window_data_;
  std::multiset<double>::iterator mid_;
  std::queue<double> history_;

  bool is_even() const;
};
