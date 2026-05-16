#include "SlidingMomentsPrefix.hpp"
#include <stdexcept>

SlidingMomentsPrefix::SlidingMomentsPrefix(std::size_t window_size)
    : window_size_(window_size) {
}

static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

void SlidingMomentsPrefix::variance_batch(const double *in, std::size_t n,
                                          double *out,
                                          std::size_t min_periods) const {
  std::vector<double> ps(n + 1, 0.0), ps2(n + 1, 0.0), pn(n + 1, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    if (std::isnan(in[i])) {
      ps[i + 1] = ps[i];
      ps2[i + 1] = ps2[i];
      pn[i + 1] = pn[i];
    } else {
      ps[i + 1] = ps[i] + in[i];
      ps2[i + 1] = ps2[i] + in[i] * in[i];
      pn[i + 1] = pn[i] + 1.0;
    }
  }

  auto compute = [&](std::size_t l, std::size_t r) -> double {
    double cnt = pn[r] - pn[l];
    if (cnt < 2.0 || cnt < static_cast<double>(min_periods))
      return kNaN;
    double s = ps[r] - ps[l];
    double s2 = ps2[r] - ps2[l];
    return std::max(0.0, (s2 - s * s / cnt) / (cnt - 1.0));
  };

  for (std::size_t i = 0; i < std::min(window_size_ - 1, n); ++i)
    out[i] = compute(0, i + 1);
  for (std::size_t i = window_size_ - 1; i < n; ++i)
    out[i] = compute(i - window_size_ + 1, i + 1);
}

void SlidingMomentsPrefix::skewness_batch(const double *in, std::size_t n,
                                          double *out,
                                          std::size_t min_periods) const {
  std::vector<double> ps(n + 1, 0), ps2(n + 1, 0), ps3(n + 1, 0), pn(n + 1, 0);
  for (std::size_t i = 0; i < n; ++i) {
    if (std::isnan(in[i])) {
      ps[i + 1] = ps[i];
      ps2[i + 1] = ps2[i];
      ps3[i + 1] = ps3[i];
      pn[i + 1] = pn[i];
    } else {
      double x = in[i];
      ps[i + 1] = ps[i] + x;
      ps2[i + 1] = ps2[i] + x * x;
      ps3[i + 1] = ps3[i] + x * x * x;
      pn[i + 1] = pn[i] + 1.0;
    }
  }

  auto compute = [&](std::size_t l, std::size_t r) -> double {
    double cnt = pn[r] - pn[l];
    if (cnt < 3.0 || cnt < static_cast<double>(min_periods))
      return kNaN;
    double s1 = ps[r] - ps[l];
    double s2 = ps2[r] - ps2[l];
    double s3 = ps3[r] - ps3[l];
    double mean = s1 / cnt;
    double m2 = s2 / cnt - mean * mean;
    if (m2 <= 0.0)
      return kNaN;
    double m3 = s3 / cnt - 3.0 * mean * s2 / cnt + 2.0 * mean * mean * mean;
    double g1 = m3 / std::pow(m2, 1.5);
    return g1 * std::sqrt(cnt * (cnt - 1.0)) / (cnt - 2.0);
  };

  for (std::size_t i = 0; i < std::min(window_size_ - 1, n); ++i)
    out[i] = compute(0, i + 1);
  for (std::size_t i = window_size_ - 1; i < n; ++i)
    out[i] = compute(i - window_size_ + 1, i + 1);
}

void SlidingMomentsPrefix::kurtosis_batch(const double *in, std::size_t n,
                                          double *out,
                                          std::size_t min_periods) const {
  std::vector<double> ps(n + 1, 0), ps2(n + 1, 0), ps3(n + 1, 0), ps4(n + 1, 0),
      pn(n + 1, 0);
  for (std::size_t i = 0; i < n; ++i) {
    if (std::isnan(in[i])) {
      ps[i + 1] = ps[i];
      ps2[i + 1] = ps2[i];
      ps3[i + 1] = ps3[i];
      ps4[i + 1] = ps4[i];
      pn[i + 1] = pn[i];
    } else {
      double x = in[i], x2 = x * x;
      ps[i + 1] = ps[i] + x;
      ps2[i + 1] = ps2[i] + x2;
      ps3[i + 1] = ps3[i] + x2 * x;
      ps4[i + 1] = ps4[i] + x2 * x2;
      pn[i + 1] = pn[i] + 1.0;
    }
  }

  auto compute = [&](std::size_t l, std::size_t r) -> double {
    double cnt = pn[r] - pn[l];
    if (cnt < 4.0 || cnt < static_cast<double>(min_periods))
      return kNaN;
    double s1 = ps[r] - ps[l];
    double s2 = ps2[r] - ps2[l];
    double s3 = ps3[r] - ps3[l];
    double s4 = ps4[r] - ps4[l];
    double mean = s1 / cnt;
    double m2 = s2 / cnt - mean * mean;
    if (m2 <= 0.0)
      return kNaN;
    double m4 = s4 / cnt - 4.0 * mean * s3 / cnt +
                6.0 * mean * mean * s2 / cnt - 3.0 * mean * mean * mean * mean;
    double g2 = m4 / (m2 * m2) - 3.0;
    return (cnt - 1.0) / ((cnt - 2.0) * (cnt - 3.0)) * ((cnt + 1.0) * g2 + 6.0);
  };

  for (std::size_t i = 0; i < std::min(window_size_ - 1, n); ++i)
    out[i] = compute(0, i + 1);
  for (std::size_t i = window_size_ - 1; i < n; ++i)
    out[i] = compute(i - window_size_ + 1, i + 1);
}