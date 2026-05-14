from __future__ import annotations

import numpy as np

from robust_rolling_core import (
    MonotonicMax,
    MonotonicMin,
    MultisetMedian,
    SlidingCovariance,
    SlidingMoments,
    SlidingWelford,
)

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

__all__ = [
    "MonotonicMax",
    "MonotonicMin",
    "MultisetMedian",
    "SlidingCovariance",
    "SlidingMoments",
    "SlidingWelford",
    "rolling_max",
    "rolling_min",
    "rolling_variance",
    "rolling_median",
    "rolling_mean",
    "rolling_skewness",
    "rolling_kurtosis",
    "rolling_cov",
    "rolling_cor",
]


def _to_float64(x) -> np.ndarray:
    if _HAS_PANDAS and isinstance(x, pd.Series):
        return x.to_numpy(dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def _wrap(result: np.ndarray, original):
    if _HAS_PANDAS and isinstance(original, pd.Series):
        return pd.Series(result, index=original.index, name=original.name)
    return result


def _count_non_nan_in_window(arr: np.ndarray, window_size: int) -> np.ndarray:
    not_nan = (~np.isnan(arr)).astype(np.float64)
    cum = np.cumsum(not_nan)
    lagged = np.empty_like(cum)
    lagged[:window_size] = 0.0
    if len(arr) > window_size:
        lagged[window_size:] = cum[:-window_size]
    return cum - lagged


def _apply_min_periods(result: np.ndarray, arr: np.ndarray,
                       window_size: int, min_periods: int) -> np.ndarray:
    if min_periods == 0 or len(arr) == 0:
        return result
    non_na_count = _count_non_nan_in_window(arr, window_size)
    result = result.copy()
    result[non_na_count < min_periods] = np.nan
    return result


def _resolve_min_periods(min_periods: int | None, window_size: int) -> int:
    mp = window_size if min_periods is None else int(min_periods)
    if mp < 0 or mp > window_size:
        raise ValueError(
            f"min_periods must be in [0, window_size], got {mp}"
        )
    return mp


def rolling_max(x, window_size: int, min_periods: int | None = None):
    arr = _to_float64(x)
    mp = _resolve_min_periods(min_periods, window_size)
    result = MonotonicMax(window_size).process_batch(arr)
    result = _apply_min_periods(result, arr, window_size, mp)
    return _wrap(result, x)


def rolling_min(x, window_size: int, min_periods: int | None = None):
    arr = _to_float64(x)
    mp = _resolve_min_periods(min_periods, window_size)
    result = MonotonicMin(window_size).process_batch(arr)
    result = _apply_min_periods(result, arr, window_size, mp)
    return _wrap(result, x)


def rolling_variance(x, window_size: int, min_periods: int | None = None):
    arr = _to_float64(x)
    mp = _resolve_min_periods(min_periods, window_size)
    result = SlidingWelford(window_size).process_batch(arr)
    result = _apply_min_periods(result, arr, window_size, mp)
    return _wrap(result, x)


def rolling_median(x, window_size: int, min_periods: int | None = None):
    arr = _to_float64(x)
    mp = _resolve_min_periods(min_periods, window_size)
    result = MultisetMedian(window_size).process_batch(arr)
    result = _apply_min_periods(result, arr, window_size, mp)
    return _wrap(result, x)


def rolling_mean(x, window_size: int, min_periods: int | None = None):
    arr = _to_float64(x)
    mp = _resolve_min_periods(min_periods, window_size)
    result = SlidingMoments(window_size).process_mean_batch(arr)
    result = _apply_min_periods(result, arr, window_size, mp)
    return _wrap(result, x)


def rolling_skewness(x, window_size: int, min_periods: int | None = None):
    arr = _to_float64(x)
    mp = _resolve_min_periods(min_periods, window_size)
    result = SlidingMoments(window_size).process_skewness_batch(arr)
    result = _apply_min_periods(result, arr, window_size, mp)
    return _wrap(result, x)


def rolling_kurtosis(x, window_size: int, min_periods: int | None = None):
    arr = _to_float64(x)
    mp = _resolve_min_periods(min_periods, window_size)
    result = SlidingMoments(window_size).process_kurtosis_batch(arr)
    result = _apply_min_periods(result, arr, window_size, mp)
    return _wrap(result, x)


def _count_valid_pairs_in_window(x: np.ndarray, y: np.ndarray,
                                  window_size: int) -> np.ndarray:
    both_valid = (~np.isnan(x) & ~np.isnan(y)).astype(np.float64)
    cum = np.cumsum(both_valid)
    lagged = np.empty_like(cum)
    lagged[:window_size] = 0.0
    if len(x) > window_size:
        lagged[window_size:] = cum[:-window_size]
    return cum - lagged


def _apply_min_periods_pair(result: np.ndarray, x: np.ndarray, y: np.ndarray,
                             window_size: int, min_periods: int) -> np.ndarray:
    if min_periods == 0 or len(x) == 0:
        return result
    valid_count = _count_valid_pairs_in_window(x, y, window_size)
    result = result.copy()
    result[valid_count < min_periods] = np.nan
    return result


def rolling_cov(x, y, window_size: int, min_periods: int | None = None):
    ax = _to_float64(x)
    ay = _to_float64(y)
    mp = _resolve_min_periods(min_periods, window_size)
    result = SlidingCovariance(window_size).process_covariance_batch(ax, ay)
    result = _apply_min_periods_pair(result, ax, ay, window_size, mp)
    return _wrap(result, x)


def rolling_cor(x, y, window_size: int, min_periods: int | None = None):
    ax = _to_float64(x)
    ay = _to_float64(y)
    mp = _resolve_min_periods(min_periods, window_size)
    result = SlidingCovariance(window_size).process_correlation_batch(ax, ay)
    result = _apply_min_periods_pair(result, ax, ay, window_size, mp)
    return _wrap(result, x)
