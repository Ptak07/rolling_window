#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

template <typename Derived> class RollingMetric {
public:
  void update(double value) {
    static_cast<Derived *>(this)->update_impl(value);
  }

  double get_value() const {
    return static_cast<const Derived *>(this)->get_value_impl();
  }

  std::size_t current_size() const {
    return static_cast<const Derived *>(this)->current_size_impl();
  }

  void process_batch(const double *input_data, std::size_t length,
                     double *output_data) {
    for (std::size_t i = 0; i < length; ++i) {
      double current_tick = input_data[i];

      if (std::isnan(current_tick)) {
        output_data[i] = std::numeric_limits<double>::quiet_NaN();
      } else {
        this->update(current_tick);
        output_data[i] = this->get_value();
      }
    }
  }
};

