#pragma once
#include "IMedianAlgorithm.hpp"
#include <queue>
#include <set>
#include <stdexcept>

class MultisetMedian : public IMedianAlgorithm {
public:
  explicit MultisetMedian(std::size_t window_size);

  void update(double new_value);
  double get_median() const;
  std::size_t current_size() const;

private:
  std::size_t window_size_;
  std::multiset<double> window_data_;
  std::multiset<double>::iterator mid_;
  std::queue<double> history_;

  bool is_even() const;
};