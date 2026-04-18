#include "../include/MonotonicMax.hpp"
#include "../include/SlidingWelfordRing.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template <typename Metric, typename Getter>
py::array_t<double> process_batch_generic(
    Metric &metric,
    py::array_t<double, py::array::c_style | py::array::forcecast> input_array,
    Getter getter) {
  py::buffer_info in_info = input_array.request();
  if (in_info.ndim != 1)
    throw std::runtime_error("Input must be 1D array");

  auto result_array = py::array_t<double>(
      py::array::ShapeContainer{in_info.shape[0]},
      py::array::StridesContainer{static_cast<py::ssize_t>(sizeof(double))});
  py::buffer_info out_info = result_array.request();

  const auto *in_ptr = static_cast<const double *>(in_info.ptr);
  auto *out_ptr = static_cast<double *>(out_info.ptr);

  for (py::ssize_t i = 0; i < in_info.shape[0]; ++i) {
    const double value = in_ptr[i];
    if (std::isnan(value)) {
      out_ptr[i] = std::numeric_limits<double>::quiet_NaN();
      continue;
    }
    metric.update(value);
    out_ptr[i] = getter(metric);
  }

  return result_array;
}

PYBIND11_MODULE(robust_rolling_core, m) {
  m.doc() = "Rolling Metrics Engine (Python Bindings)";

  py::class_<SlidingWelfordRing>(m, "SlidingWelford")
      .def(py::init<std::size_t>())
      .def("update", &SlidingWelfordRing::update)
      .def("get_variance", &SlidingWelfordRing::get_value)
      .def("process_batch",
           [](SlidingWelfordRing &self,
              py::array_t<double, py::array::c_style | py::array::forcecast>
                  input) {
             return process_batch_generic(
                 self, input,
                 [](SlidingWelfordRing &m) { return m.get_variance(); });
           });

  py::class_<MonotonicMax>(m, "MonotonicMax")
      .def(py::init<std::size_t>())
      .def("update", &MonotonicMax::update)
      .def("get_max", &MonotonicMax::get_max)
      .def("process_batch",
           [](MonotonicMax &self,
              py::array_t<double, py::array::c_style | py::array::forcecast>
                  input) {
             return process_batch_generic(
                 self, input, [](MonotonicMax &m) { return m.get_max(); });
           });
}