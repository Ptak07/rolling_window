#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

class SlidingMomentsPrefix {
public:
  explicit SlidingMomentsPrefix(std::size_t window_size);

  void variance_batch(const double *in, std::size_t n, double *out,
                      std::size_t min_periods) const;
  void skewness_batch(const double *in, std::size_t n, double *out,
                      std::size_t min_periods) const;
  void kurtosis_batch(const double *in, std::size_t n, double *out,
                      std::size_t min_periods) const;

private:
  std::size_t window_size_;
};