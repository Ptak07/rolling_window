from __future__ import annotations

import numpy as np

from robust_rolling_core import (
    MonotonicMax,
    MultisetMedian,
    SlidingWelford,
)

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

__all__ = [
    "MonotonicMax",
    "MultisetMedian",
    "SlidingWelford",
    "rolling_max",
    "rolling_variance",
    "rolling_median",
]


def _to_float64(x) -> np.ndarray:
    if _HAS_PANDAS and isinstance(x, pd.Series):
        return x.to_numpy(dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def _wrap(result: np.ndarray, original):
    if _HAS_PANDAS and isinstance(original, pd.Series):
        return pd.Series(result, index=original.index, name=original.name)
    return result


def rolling_max(x, window_size: int):
    return _wrap(MonotonicMax(window_size).process_batch(_to_float64(x)), x)


def rolling_variance(x, window_size: int):
    return _wrap(SlidingWelford(window_size).process_batch(_to_float64(x)), x)


def rolling_median(x, window_size: int):
    return _wrap(MultisetMedian(window_size).process_batch(_to_float64(x)), x)