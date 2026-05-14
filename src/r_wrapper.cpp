#include "MonotonicMax.hpp"
#include "MonotonicMin.hpp"
#include "MultisetMedian.hpp"
#include "SlidingCovariance.hpp"
#include "SlidingMoments.hpp"
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

SEXP rolling_mean_c(SEXP r_data, SEXP r_window_size) {
  if (!Rf_isReal(r_data))
    Rf_error("Input data must be a numeric vector.");

  double *input_ptr = REAL(r_data);
  R_xlen_t n = XLENGTH(r_data);
  std::size_t k = read_window_size(r_window_size);

  SlidingMoments sm(k);

  SEXP r_result;
  PROTECT(r_result = Rf_allocVector(REALSXP, n));
  double *output_ptr = REAL(r_result);

  for (R_xlen_t i = 0; i < n; ++i) {
    sm.update(input_ptr[i]);
    output_ptr[i] = sm.get_mean();
  }

  UNPROTECT(1);
  return r_result;
}

SEXP rolling_skewness_c(SEXP r_data, SEXP r_window_size) {
  if (!Rf_isReal(r_data))
    Rf_error("Input data must be a numeric vector.");

  double *input_ptr = REAL(r_data);
  R_xlen_t n = XLENGTH(r_data);
  std::size_t k = read_window_size(r_window_size);

  SlidingMoments sm(k);

  SEXP r_result;
  PROTECT(r_result = Rf_allocVector(REALSXP, n));
  double *output_ptr = REAL(r_result);

  for (R_xlen_t i = 0; i < n; ++i) {
    sm.update(input_ptr[i]);
    output_ptr[i] = sm.get_skewness();
  }

  UNPROTECT(1);
  return r_result;
}

SEXP rolling_kurtosis_c(SEXP r_data, SEXP r_window_size) {
  if (!Rf_isReal(r_data))
    Rf_error("Input data must be a numeric vector.");

  double *input_ptr = REAL(r_data);
  R_xlen_t n = XLENGTH(r_data);
  std::size_t k = read_window_size(r_window_size);

  SlidingMoments sm(k);

  SEXP r_result;
  PROTECT(r_result = Rf_allocVector(REALSXP, n));
  double *output_ptr = REAL(r_result);

  for (R_xlen_t i = 0; i < n; ++i) {
    sm.update(input_ptr[i]);
    output_ptr[i] = sm.get_kurtosis();
  }

  UNPROTECT(1);
  return r_result;
}

SEXP rolling_cov_c(SEXP r_x, SEXP r_y, SEXP r_window_size) {
  if (!Rf_isReal(r_x) || !Rf_isReal(r_y))
    Rf_error("Input data must be numeric vectors.");
  R_xlen_t n = XLENGTH(r_x);
  if (XLENGTH(r_y) != n)
    Rf_error("x and y must have the same length.");

  double *x_ptr = REAL(r_x);
  double *y_ptr = REAL(r_y);
  std::size_t k = read_window_size(r_window_size);

  SlidingCovariance sc(k);

  SEXP r_result;
  PROTECT(r_result = Rf_allocVector(REALSXP, n));
  double *output_ptr = REAL(r_result);

  for (R_xlen_t i = 0; i < n; ++i) {
    sc.update(x_ptr[i], y_ptr[i]);
    output_ptr[i] = sc.get_covariance();
  }

  UNPROTECT(1);
  return r_result;
}

SEXP rolling_cor_c(SEXP r_x, SEXP r_y, SEXP r_window_size) {
  if (!Rf_isReal(r_x) || !Rf_isReal(r_y))
    Rf_error("Input data must be numeric vectors.");
  R_xlen_t n = XLENGTH(r_x);
  if (XLENGTH(r_y) != n)
    Rf_error("x and y must have the same length.");

  double *x_ptr = REAL(r_x);
  double *y_ptr = REAL(r_y);
  std::size_t k = read_window_size(r_window_size);

  SlidingCovariance sc(k);

  SEXP r_result;
  PROTECT(r_result = Rf_allocVector(REALSXP, n));
  double *output_ptr = REAL(r_result);

  for (R_xlen_t i = 0; i < n; ++i) {
    sc.update(x_ptr[i], y_ptr[i]);
    output_ptr[i] = sc.get_correlation();
  }

  UNPROTECT(1);
  return r_result;
}

static const R_CallMethodDef CallEntries[] = {
    {"rolling_variance_c", reinterpret_cast<DL_FUNC>(&rolling_variance_c), 2},
    {"rolling_max_c", reinterpret_cast<DL_FUNC>(&rolling_max_c), 2},
    {"rolling_min_c", reinterpret_cast<DL_FUNC>(&rolling_min_c), 2},
    {"rolling_median_c", reinterpret_cast<DL_FUNC>(&rolling_median_c), 2},
    {"rolling_mean_c", reinterpret_cast<DL_FUNC>(&rolling_mean_c), 2},
    {"rolling_skewness_c", reinterpret_cast<DL_FUNC>(&rolling_skewness_c), 2},
    {"rolling_kurtosis_c", reinterpret_cast<DL_FUNC>(&rolling_kurtosis_c), 2},
    {"rolling_cov_c", reinterpret_cast<DL_FUNC>(&rolling_cov_c), 3},
    {"rolling_cor_c", reinterpret_cast<DL_FUNC>(&rolling_cor_c), 3},
    {nullptr, nullptr, 0}};

} // namespace

extern "C" {

void R_init_robustrolling(DllInfo *dll) {
  R_registerRoutines(dll, nullptr, CallEntries, nullptr, nullptr);
  R_useDynamicSymbols(dll, FALSE);
}

} // extern "C"
