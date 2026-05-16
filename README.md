# robustrolling

[![R package check](https://github.com/IgorPtak/rolling_window/actions/workflows/r_check.yml/badge.svg)](https://github.com/IgorPtak/rolling_window/actions/workflows/r_check.yml)
[![C++ tests](https://github.com/IgorPtak/rolling_window/actions/workflows/cpp_test.yml/badge.svg)](https://github.com/IgorPtak/rolling_window/actions/workflows/cpp_test.yml)
[![Python package](https://github.com/IgorPtak/rolling_window/actions/workflows/python.yml/badge.svg)](https://github.com/IgorPtak/rolling_window/actions/workflows/python.yml)

High-performance rolling window metrics for R and Python, implemented in C++17.

---

## Overview

`robustrolling` provides numerically stable, memory-efficient rolling window
algorithms built in C++17 and exposed to both R and Python. All algorithms:

- run in **O(1) time per element** (O(log n) for median),
- handle **NaN / NA transparently**,
- support a **`min_periods`** parameter (pandas-compatible semantics),
- share a common **CRTP base** (`RollingMetric<Derived>`) — zero virtual
  dispatch, flat ring-buffer memory layout.

---

## Features

Six production-grade algorithms covering the most common rolling statistics:

| C++ class            | Algorithm                                   | Time           | R API                                                      | Python class        |
| -------------------- | ------------------------------------------- | -------------- | ---------------------------------------------------------- | ------------------- |
| `SlidingWelfordRing` | Welford online + ring buffer                | O(1)           | `rolling_variance()`                                       | `SlidingWelford`    |
| `MonotonicMax`       | Monotonic deque                             | O(1) amortised | `rolling_max()`                                            | `MonotonicMax`      |
| `MonotonicMin`       | Monotonic deque                             | O(1) amortised | `rolling_min()`                                            | `MonotonicMin`      |
| `MultisetMedian`     | `std::multiset` dual-iterator               | O(log n)       | `rolling_median()`                                         | `MultisetMedian`    |
| `SlidingMoments`     | Terriberry's algorithm (4th-moment Welford) | O(1)           | `rolling_mean()` `rolling_skewness()` `rolling_kurtosis()` | `SlidingMoments`    |
| `SlidingCovariance`  | Welford 2D online                           | O(1)           | `rolling_cov()` `rolling_cor()`                            | `SlidingCovariance` |

---

## Installation

### R

```r
remotes::install_github("IgorPtak/rolling_window")
```

Or build from source:

```bash
git clone https://github.com/IgorPtak/rolling_window.git
cd rolling_window
make r-build
```

Requires: R ≥ 4.0, a C++17 compiler.

### Python

```bash
git clone https://github.com/IgorPtak/rolling_window.git
cd rolling_window
pip install py_package/
```

With pandas support:

```bash
pip install "py_package/[pandas]"
```

Requires: Python ≥ 3.10, a C++17 compiler, pybind11.

---

## Usage

### R

```r
library(robustrolling)

x <- as.double(c(1, 3, 2, 5, 4))
```

**Max / Min**

```r
rolling_max(x, 3L)
#> [1] NA NA  3  5  5

rolling_min(x, 3L)
#> [1] NA NA  1  2  2
```

**Median**

```r
rolling_median(x, 3L)
#> [1] NA NA  2  3  4
```

**Variance and mean**

```r
y <- as.double(c(1, 2, 3, 4, 5))

rolling_variance(y, 3L)
#> [1] NA NA  1  1  1

rolling_mean(y, 3L)
#> [1] NA NA  2  3  4
```

**Higher moments**

```r
rolling_skewness(y, 3L)
#> [1] NA NA  0  0  0

rolling_kurtosis(y, 4L)
#> [1] NA NA NA -1.2 -1.2
```

**Covariance and Pearson correlation**

```r
a <- as.double(c(1, 2, 3, 4, 5))
b <- as.double(c(2, 4, 6, 8, 10))

rolling_cov(a, b, 3L)
#> [1] NA NA  2  2  2

rolling_cor(a, b, 3L)
#> [1] NA NA  1  1  1
```

**`min_periods` — require fewer observations**

```r
rolling_max(x, 3L, min_periods = 1L)
#> [1]  1  3  3  5  5
```

---

### Python — high-level API

All functions accept `np.ndarray` and `pd.Series`:

```python
import numpy as np
import robustrolling as rr

x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])

rr.rolling_max(x, 3)
# array([nan, nan,  3.,  5.,  5.])

rr.rolling_min(x, 3)
# array([nan, nan,  1.,  2.,  2.])

rr.rolling_median(x, 3)
# array([nan, nan,  2.,  3.,  4.])
```

```python
y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

rr.rolling_variance(y, 3)
# array([nan, nan,  1.,  1.,  1.])

rr.rolling_mean(y, 3)
# array([nan, nan,  2.,  3.,  4.])

rr.rolling_skewness(y, 3)
# array([nan, nan,  0.,  0.,  0.])

rr.rolling_kurtosis(y, 4)
# array([nan, nan,  nan, -1.2, -1.2])
```

```python
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

rr.rolling_cov(a, b, 3)
# array([nan, nan,  2.,  2.,  2.])

rr.rolling_cor(a, b, 3)
# array([nan, nan,  1.,  1.,  1.])
```

**pandas Series — index and name are preserved:**

```python
import pandas as pd

prices = pd.Series(
    [100.0, 102.0, 98.0, 105.0, 103.0],
    index=pd.date_range("2024-01-01", periods=5),
    name="close",
)

rr.rolling_max(prices, 3)
# 2024-01-01      NaN
# 2024-01-02      NaN
# 2024-01-03    102.0
# 2024-01-04    105.0
# 2024-01-05    105.0
# Freq: D, Name: close, dtype: float64
```

### Python — low-level C++ classes

Direct access to the engine objects for incremental (streaming) use:

```python
from robustrolling import MonotonicMax, SlidingMoments, SlidingCovariance
import numpy as np

# Streaming — one value at a time
engine = MonotonicMax(3)
for v in [1.0, 3.0, 2.0, 5.0]:
    engine.update(v)
    print(engine.get_value())
# 1.0 → 3.0 → 3.0 → 5.0

# Batch — zero-copy NumPy buffer
engine2 = SlidingMoments(3)
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(engine2.process_mean_batch(x))
# [nan, nan, 2., 3., 4.]
print(engine2.process_skewness_batch(x))
# [nan, nan, 0., 0., 0.]

# Covariance engine
cov_engine = SlidingCovariance(3)
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([2.0, 4.0, 6.0, 8.0])
print(cov_engine.process_covariance_batch(a, b))
# [nan, nan, 2., 2.]
print(cov_engine.process_correlation_batch(a, b))
# [nan, nan, 1., 1.]
```

---

## Architecture

The C++ core uses **CRTP (Curiously Recurring Template Pattern)** to share a
common interface across all algorithm classes without virtual dispatch:

```
RollingMetric<Derived>
├── MonotonicMax       — monotonic deque (max)
├── MonotonicMin       — monotonic deque (min)
├── MultisetMedian     — std::multiset + dual-iterator median tracking
├── SlidingWelfordRing — Welford variance + ring buffer eviction
├── SlidingMoments     — Terriberry's 4th-moment recurrence (mean, skewness, kurtosis)
└── SlidingCovariance  — 2D Welford for covariance and Pearson correlation
```

**Bindings:**

| Language | Technology                       | Notes                      |
| -------- | -------------------------------- | -------------------------- |
| R        | Pure R/C API (`.Call`)           | No Rcpp dependency         |
| Python   | pybind11 + NumPy buffer protocol | Zero-copy batch processing |

---

## Development

### Requirements

| Tool         | Version                                  |
| ------------ | ---------------------------------------- |
| C++ compiler | C++17 (GCC ≥ 9, Clang ≥ 10, MSVC ≥ 2019) |
| CMake        | ≥ 3.14                                   |
| R            | ≥ 4.0                                    |
| Python       | ≥ 3.10                                   |

### Build and test

```bash
# C++ unit tests (gtest)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_core --parallel
ctest --test-dir build --output-on-failure

# R package
make r-build   # sync headers + roxygen + R CMD INSTALL
make r-test    # tinytest

# Python package
make py-build  # editable install
make py-test   # pytest
```

---

## License

MIT © Igor Ptak
