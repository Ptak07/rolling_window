#pragma once

#include <cstddef>

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
};
