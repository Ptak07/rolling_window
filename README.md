# robustrolling

High-performance rolling window metrics for R and Python, implemented in C++17.

---

## Overview

`robustrolling` provides numerically stable, O(1) rolling window algorithms built in C++ and exposed to both R and Python. All core algorithms use the CRTP pattern to share a common interface without virtual dispatch overhead.

### Algorithms

| Class | Algorithm | Time complexity | Memory |
|---|---|---|---|
| `MonotonicMax` | Monotonic deque | O(1) amortised | O(n) |
| `MultisetMedian` | `std::multiset` + tracked iterator | O(log n) | O(n) |
| `SlidingWelford` | Welford online algorithm | O(1) | O(n) |
| `SlidingWelfordRing` | Welford + ring buffer | O(1) | O(n) |

### R functions

`rolling_max()`, `rolling_median()`, `rolling_variance()`

### Python functions

`rolling_max()`, `rolling_median()`, `rolling_variance()` — accept both `pd.Series` and `np.ndarray`.

---

## Installation

### R

```r
# Install from source
install.packages("remotes")
remotes::install_github("Ptak07/rolling_window")
```

Or clone and build locally:

```bash
git clone https://github.com/Ptak07/rolling_window.git
cd rolling_window
make r-build
```

### Python

```bash
git clone https://github.com/Ptak07/rolling_window.git
cd rolling_window
python -m venv .venv && source .venv/bin/activate
pip install py_package/
```

With pandas support:

```bash
pip install "py_package/[pandas]"
```

---

## Usage

### R

```r
library(robustrolling)

x <- as.double(c(1, 3, 2, 5, 4, 6, 3))

rolling_max(x, 3L)
#> [1] 1 3 3 5 5 6 6

rolling_median(x, 3L)
#> [1] 1 2 2 3 4 5 4

rolling_variance(x, 3L)
#> [1]        NA 2.000000 1.000000 4.333333 2.333333 2.333333 4.333333
```

### Python — numpy

```python
import numpy as np
import robustrolling as rr

x = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 3.0])

rr.rolling_max(x, 3)
# array([1., 3., 3., 5., 5., 6., 6.])

rr.rolling_median(x, 3)
# array([1., 2., 2., 3., 4., 5., 4.])
```

### Python — pandas

```python
import pandas as pd
import robustrolling as rr

prices = pd.Series(
    [100.0, 102.0, 98.0, 105.0, 103.0],
    index=pd.date_range("2024-01-01", periods=5),
    name="close",
)

rr.rolling_max(prices, 3)
# 2024-01-01    100.0
# 2024-01-02    102.0
# 2024-01-03    102.0
# 2024-01-04    105.0
# 2024-01-05    105.0
# Freq: D, Name: close, dtype: float64
```

### Python — low-level C++ classes

```python
import robust_rolling_core as rrc
import numpy as np

engine = rrc.MonotonicMax(3)
engine.update(1.0)
engine.update(3.0)
engine.update(2.0)
print(engine.get_value())  # 3.0

# batch processing
out = engine.process_batch(np.array([1.0, 3.0, 2.0, 5.0, 4.0]))
```

---

## Development

### Requirements

- C++17 compiler
- CMake ≥ 3.15
- R ≥ 4.0 with `devtools`, `tinytest`, `roxygen2`
- Python ≥ 3.10 with `pybind11`, `scikit-build-core`, `pytest`

### Build and test

```bash
# C++ tests
cmake -B build && cmake --build build
ctest --test-dir build --output-on-failure

# R package
make r-all

# Python package
make py-all
```

---

## License

MIT © Igor Ptak
