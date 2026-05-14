#include "../include/MonotonicMax.hpp"
#include "../include/MonotonicMin.hpp"
#include "../include/MultisetMedian.hpp"
#include "../include/SlidingMoments.hpp"
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
      metric.skip();
      out_ptr[i] = getter(metric);
    } else {
      metric.update(value);
      out_ptr[i] = getter(metric);
    }
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

  py::class_<MonotonicMin>(m, "MonotonicMin")
      .def(py::init<std::size_t>())
      .def("update", &MonotonicMin::update)
      .def("get_min", &MonotonicMin::get_min)
      .def("process_batch",
           [](MonotonicMin &self,
              py::array_t<double, py::array::c_style | py::array::forcecast>
                  input) {
             return process_batch_generic(
                 self, input, [](MonotonicMin &m) { return m.get_min(); });
           });

  py::class_<MultisetMedian>(m, "MultisetMedian")
      .def(py::init<std::size_t>())
      .def("update", &MultisetMedian::update)
      .def("get_median", &MultisetMedian::get_median)
      .def("process_batch",
           [](MultisetMedian &self,
              py::array_t<double, py::array::c_style | py::array::forcecast>
                  input) {
             return process_batch_generic(
                 self, input,
                 [](MultisetMedian &m) { return m.get_median(); });
           });

  py::class_<SlidingMoments>(m, "SlidingMoments")
      .def(py::init<std::size_t>())
      .def("update", &SlidingMoments::update)
      .def("reset", &SlidingMoments::reset)
      .def("current_size", &SlidingMoments::current_size)
      .def("get_mean", &SlidingMoments::get_mean)
      .def("get_skewness", &SlidingMoments::get_skewness)
      .def("get_kurtosis", &SlidingMoments::get_kurtosis)
      .def("process_skewness_batch",
           [](SlidingMoments &self,
              py::array_t<double, py::array::c_style | py::array::forcecast>
                  input) {
             py::buffer_info info = input.request();
             if (info.ndim != 1)
               throw std::runtime_error("Input must be 1D array");
             auto result = py::array_t<double>(
                 py::array::ShapeContainer{info.shape[0]},
                 py::array::StridesContainer{
                     static_cast<py::ssize_t>(sizeof(double))});
             const auto *in = static_cast<const double *>(info.ptr);
             auto *out = static_cast<double *>(result.request().ptr);
             for (py::ssize_t i = 0; i < info.shape[0]; ++i) {
               self.update(in[i]);
               out[i] = self.get_skewness();
             }
             return result;
           })
      .def("process_kurtosis_batch",
           [](SlidingMoments &self,
              py::array_t<double, py::array::c_style | py::array::forcecast>
                  input) {
             py::buffer_info info = input.request();
             if (info.ndim != 1)
               throw std::runtime_error("Input must be 1D array");
             auto result = py::array_t<double>(
                 py::array::ShapeContainer{info.shape[0]},
                 py::array::StridesContainer{
                     static_cast<py::ssize_t>(sizeof(double))});
             const auto *in = static_cast<const double *>(info.ptr);
             auto *out = static_cast<double *>(result.request().ptr);
             for (py::ssize_t i = 0; i < info.shape[0]; ++i) {
               self.update(in[i]);
               out[i] = self.get_kurtosis();
             }
             return result;
           });
}
