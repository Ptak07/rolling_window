#include "SlidingMean.hpp"
#include <stdexcept>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

SlidingMean::SlidingMean(std::size_t window_size) : window_size_(window_size) {
  if (window_size_ == 0)
    throw std::invalid_argument("Window length must be greater than 0");
  buffer_.resize(window_size_, std::numeric_limits<double>::quiet_NaN());
}

void SlidingMean::fast_mean_batch(const double *in, std::size_t n, double *out,
                                  std::size_t min_periods) const {
  std::vector<double> prefix_sum(n + 1, 0.0);
  std::vector<double> prefix_count(n + 1, 0.0);

  for (std::size_t i = 0; i < n; ++i) {
    if (std::isnan(in[i])) {
      prefix_sum[i + 1] = prefix_sum[i];
      prefix_count[i + 1] = prefix_count[i];
    } else {
      prefix_sum[i + 1] = prefix_sum[i] + in[i];
      prefix_count[i + 1] = prefix_count[i] + 1.0;
    }
  }

  for (std::size_t i = 0; i < std::min(window_size_ - 1, n); ++i) {
    double cnt = prefix_count[i + 1];
    out[i] = cnt >= static_cast<double>(min_periods)
                 ? prefix_sum[i + 1] / cnt
                 : std::numeric_limits<double>::quiet_NaN();
  }

#if defined(__ARM_NEON)
  {
    float64x2_t mp_vec = vdupq_n_f64(static_cast<double>(min_periods));
    float64x2_t nan_vec = vdupq_n_f64(std::numeric_limits<double>::quiet_NaN());
    std::size_t i = window_size_ - 1;
    for (; i + 1 < n; i += 2) {
      float64x2_t ws = vsubq_f64(vld1q_f64(&prefix_sum[i + 1]),
                                 vld1q_f64(&prefix_sum[i - window_size_ + 1]));
      float64x2_t wc =
          vsubq_f64(vld1q_f64(&prefix_count[i + 1]),
                    vld1q_f64(&prefix_count[i - window_size_ + 1]));
      uint64x2_t mask = vcgeq_f64(wc, mp_vec);
      vst1q_f64(&out[i], vbslq_f64(mask, vdivq_f64(ws, wc), nan_vec));
    }
    for (; i < n; ++i) {
      double cnt = prefix_count[i + 1] - prefix_count[i - window_size_ + 1];
      double sm = prefix_sum[i + 1] - prefix_sum[i - window_size_ + 1];
      out[i] = cnt >= static_cast<double>(min_periods)
                   ? sm / cnt
                   : std::numeric_limits<double>::quiet_NaN();
    }
  }
#else
  for (std::size_t i = window_size_ - 1; i < n; ++i) {
    double cnt = prefix_count[i + 1] - prefix_count[i - window_size_ + 1];
    double sm = prefix_sum[i + 1] - prefix_sum[i - window_size_ + 1];
    out[i] = cnt >= static_cast<double>(min_periods)
                 ? sm / cnt
                 : std::numeric_limits<double>::quiet_NaN();
  }
#endif
}

void SlidingMean::update_impl(double value) {
  bool ring_full = (n_written_ >= window_size_);
  if (ring_full) {
    double x_out = buffer_[head_];
    if (!std::isnan(x_out)) {
      sum_ -= x_out;
      --count_;
    }
  }
  buffer_[head_] = value;
  head_ = (head_ + 1) % window_size_;
  if (!ring_full)
    ++n_written_;
  sum_ += value;
  ++count_;
}

void SlidingMean::skip_impl() {
  bool ring_full = (n_written_ >= window_size_);
  if (ring_full) {
    double x_out = buffer_[head_];
    if (!std::isnan(x_out)) {
      sum_ -= x_out;
      --count_;
    }
  }
  buffer_[head_] = std::numeric_limits<double>::quiet_NaN();
  head_ = (head_ + 1) % window_size_;
  if (!ring_full)
    ++n_written_;
}

double SlidingMean::get_value_impl() const {
  return get_mean();
}

std::size_t SlidingMean::current_size_impl() const {
  return count_;
}

double SlidingMean::get_mean() const {
  if (count_ == 0)
    return std::numeric_limits<double>::quiet_NaN();
  return sum_ / static_cast<double>(count_);
}
