# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-05-09

### Added

#### C++
- `MonotonicMax` — rolling maximum via monotonic deque, O(1) amortised
- `MultisetMedian` — rolling median via `std::multiset` with tracked iterator, O(log n)
- `SlidingWelford` — rolling variance via Welford online algorithm, O(1)
- `SlidingWelfordRing` — Welford variant using a ring buffer, O(1)
- CRTP base class `RollingMetric<Derived>` providing shared `update()`, `process_batch()`, `get_value()`, `current_size()` interface

#### R package
- `rolling_max(x, window_size)` — rolling maximum
- `rolling_variance(x, window_size)` — rolling sample variance
- `rolling_median(x, window_size)` — rolling median

#### Python package
- Low-level C++ bindings via pybind11 (`robust_rolling_core`): `MonotonicMax`, `MultisetMedian`, `SlidingWelford`
- High-level Python API (`robustrolling`): `rolling_max()`, `rolling_variance()`, `rolling_median()`
- Pandas integration: functions accept `pd.Series` and return `pd.Series` with index and name preserved

### Fixed
- Iterator bug in `MultisetMedian::update_impl()`: even-sized windows filled in descending order returned wrong median
- Segfault in `MultisetMedian` for `window_size=2` during sliding phase
