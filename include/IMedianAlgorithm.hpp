#pragma once
#include <cstddef>

class IMedianAlgorithm {
public:
  virtual ~IMedianAlgorithm() = default;

  virtual void update(double new_value) = 0;
  virtual double get_median() const = 0;
  virtual std::size_t current_size() const = 0;
};