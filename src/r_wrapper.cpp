#include "MonotonicMax.hpp"
#include "MonotonicMin.hpp"
#include "MultisetMedian.hpp"
#include "SlidingWelfordRing.hpp"

#include <R_ext/Arith.h>
#include <R_ext/Rdynload.h>
#include <Rinternals.h>

namespace {

std::size_t read_window_size(SEXP r_window_size) {
  double value = Rf_asReal(r_window_size);
  if (!R_FINITE(value) || value <= 0.0) {
    Rf_error("Window size must be a positive finite number.");
  }
  return static_cast<std::size_t>(value);
}

SEXP rolling_variance_c(SEXP r_data, SEXP r_window_size) {
  if (!Rf_isReal(r_data)) {
    Rf_error("Input data must be a numeric vector.");
  }

  double *input_ptr = REAL(r_data);
  R_xlen_t n = XLENGTH(r_data);
  std::size_t k = read_window_size(r_window_size);

  SlidingWelfordRing welford(k);

  SEXP r_result;
  PROTECT(r_result = Rf_allocVector(REALSXP, n));
  double *output_ptr = REAL(r_result);

  welford.process_batch(input_ptr, static_cast<std::size_t>(n), output_ptr);

  UNPROTECT(1);
  return r_result;
}

SEXP rolling_max_c(SEXP r_data, SEXP r_window_size) {
  if (!Rf_isReal(r_data)) {
    Rf_error("Input data must be a numeric vector.");
  }

  double *input_ptr = REAL(r_data);
  R_xlen_t n = XLENGTH(r_data);
  std::size_t k = read_window_size(r_window_size);

  MonotonicMax monotonic_max(k);

  SEXP r_result;
  PROTECT(r_result = Rf_allocVector(REALSXP, n));
  double *output_ptr = REAL(r_result);

  monotonic_max.process_batch(input_ptr, static_cast<std::size_t>(n),
                              output_ptr);

  UNPROTECT(1);
  return r_result;
}

SEXP rolling_min_c(SEXP r_data, SEXP r_window_size) {
  if (!Rf_isReal(r_data)) {
    Rf_error("Input data must be a numeric vector.");
  }

  double *input_ptr = REAL(r_data);
  R_xlen_t n = XLENGTH(r_data);
  std::size_t k = read_window_size(r_window_size);

  MonotonicMin monotonic_min(k);

  SEXP r_result;
  PROTECT(r_result = Rf_allocVector(REALSXP, n));
  double *output_ptr = REAL(r_result);

  monotonic_min.process_batch(input_ptr, static_cast<std::size_t>(n),
                              output_ptr);

  UNPROTECT(1);
  return r_result;
}

SEXP rolling_median_c(SEXP r_data, SEXP r_window_size) {
  if (!Rf_isReal(r_data)) {
    Rf_error("Input data must be a numeric vector.");
  }

  double *input_ptr = REAL(r_data);
  R_xlen_t n = XLENGTH(r_data);
  std::size_t k = read_window_size(r_window_size);

  MultisetMedian multiset_median(k);

  SEXP r_result;
  PROTECT(r_result = Rf_allocVector(REALSXP, n));
  double *output_ptr = REAL(r_result);

  multiset_median.process_batch(input_ptr, static_cast<std::size_t>(n),
                                output_ptr);

  UNPROTECT(1);
  return r_result;
}

static const R_CallMethodDef CallEntries[] = {
    {"rolling_variance_c", reinterpret_cast<DL_FUNC>(&rolling_variance_c), 2},
    {"rolling_max_c", reinterpret_cast<DL_FUNC>(&rolling_max_c), 2},
    {"rolling_min_c", reinterpret_cast<DL_FUNC>(&rolling_min_c), 2},
    {"rolling_median_c", reinterpret_cast<DL_FUNC>(&rolling_median_c), 2},
    {nullptr, nullptr, 0}};

} // namespace

extern "C" {

void R_init_robustrolling(DllInfo *dll) {
  R_registerRoutines(dll, nullptr, CallEntries, nullptr, nullptr);
  R_useDynamicSymbols(dll, FALSE);
}

} // extern "C"
