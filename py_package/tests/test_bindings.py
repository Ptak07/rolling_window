import math

import numpy as np
import pandas as pd
import pytest

import robust_rolling_core as rrc
import robustrolling as rr


# ── helpers ────────────────────────────────────────────────────────────────────

def _nan_allclose(result, expected, rtol=1e-12, atol=1e-12):
    """Assert arrays are equal, treating NaN positions as matching."""
    result = np.asarray(result, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    nan_exp = np.isnan(expected)
    assert np.array_equal(np.isnan(result), nan_exp), (
        f"NaN mask mismatch:\n  result  = {result}\n  expected= {expected}"
    )
    if np.any(~nan_exp):
        np.testing.assert_allclose(result[~nan_exp], expected[~nan_exp], rtol=rtol, atol=atol)


def _var_ref(x, k):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - k + 1):i + 1]
        if len(w) >= 2:
            out[i] = np.var(w, ddof=1)
    return out


def _median_ref(x, k):
    return np.array([np.median(x[max(0, i - k + 1):i + 1]) for i in range(len(x))])


# ── C++ engine — SlidingWelford ────────────────────────────────────────────────

class TestSlidingWelford:

    def test_known_values(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = rrc.SlidingWelford(3).process_batch(x)
        _nan_allclose(out, [np.nan, 0.5, 1.0, 1.0, 1.0])

    def test_window_size_1_all_nan(self):
        out = rrc.SlidingWelford(1).process_batch(np.array([10.0, 11.0, 12.0]))
        assert np.all(np.isnan(out))

    def test_window_larger_than_array(self):
        out = rrc.SlidingWelford(10).process_batch(np.array([2.0, 4.0, 6.0]))
        assert np.isnan(out[0])
        assert np.isclose(out[1], 2.0, rtol=1e-12)   # var([2,4])
        assert np.isclose(out[2], 4.0, rtol=1e-12)   # var([2,4,6])

    def test_constant_zero_variance(self):
        out = rrc.SlidingWelford(4).process_batch(np.full(8, 5.0))
        assert np.isnan(out[0])
        np.testing.assert_allclose(out[1:], 0.0, atol=1e-12)

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_against_naive_reference(self, k):
        x = np.array([-3.0, -1.0, 0.0, 2.0, 10.0, 7.0, 7.0, 8.0])
        out = rrc.SlidingWelford(k).process_batch(x)
        np.testing.assert_allclose(out, _var_ref(x, k), rtol=1e-11, atol=1e-11,
                                   equal_nan=True)

    def test_nan_does_not_contribute_to_welford(self):
        # NaN at pos 2: window still holds [1,2], so var=0.5
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        out = rrc.SlidingWelford(3).process_batch(x)
        assert np.isclose(out[2], 0.5, atol=1e-12)

    def test_empty_input(self):
        out = rrc.SlidingWelford(3).process_batch(np.array([]))
        assert len(out) == 0 and out.dtype == np.float64

    def test_rejects_zero_window(self):
        with pytest.raises(ValueError, match="Window length must be greater than 0"):
            rrc.SlidingWelford(0)

    @pytest.mark.parametrize("k", [-1, -5])
    def test_rejects_negative_window(self, k):
        with pytest.raises(TypeError):
            rrc.SlidingWelford(k)

    def test_rejects_none_window(self):
        with pytest.raises(TypeError):
            rrc.SlidingWelford(None)

    def test_rejects_inf_window(self):
        with pytest.raises((ValueError, OverflowError, TypeError)):
            rrc.SlidingWelford(float("inf"))

    def test_rejects_2d_input(self):
        with pytest.raises(RuntimeError, match="Input must be 1D array"):
            rrc.SlidingWelford(2).process_batch(np.ones((2, 3)))

    def test_integer_input_converted_to_float64(self):
        out = rrc.SlidingWelford(2).process_batch(np.array([1, 2, 3, 4], dtype=np.int32))
        assert out.dtype == np.float64


# ── C++ engine — MonotonicMax ──────────────────────────────────────────────────

class TestMonotonicMax:

    def test_known_values(self):
        x = np.array([1.0, 2.0, 3.0, 2.0, 5.0])
        out = rrc.MonotonicMax(3).process_batch(x)
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 3.0, 5.0], atol=1e-12)

    def test_window_size_1_identity(self):
        x = np.array([-1.0, 0.0, 10.0, 2.0])
        np.testing.assert_allclose(rrc.MonotonicMax(1).process_batch(x), x)

    def test_window_larger_than_array_cumulative(self):
        x = np.array([2.0, -3.0, 7.0, 1.0])
        np.testing.assert_allclose(rrc.MonotonicMax(20).process_batch(x),
                                   np.maximum.accumulate(x))

    def test_decreasing_sequence(self):
        x = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
        np.testing.assert_allclose(rrc.MonotonicMax(3).process_batch(x),
                                   [9.0, 9.0, 9.0, 7.0, 5.0])

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_against_naive_reference(self, k):
        x = np.array([-2.0, 6.0, 1.0, 8.0, 0.0, 8.0, -1.0])
        out = rrc.MonotonicMax(k).process_batch(x)
        expected = np.array([np.max(x[max(0, i - k + 1):i + 1]) for i in range(len(x))])
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_nan_advances_window_returns_current_max(self):
        # window=2: [1,2,NaN,1]. At NaN: window=[2,NaN], max=2 (not NaN passthrough)
        x = np.array([1.0, 2.0, np.nan, 1.0])
        out = rrc.MonotonicMax(2).process_batch(x)
        assert out[2] == 2.0

    def test_nan_at_start_empty_window_returns_nan(self):
        out = rrc.MonotonicMax(2).process_batch(np.array([np.nan, 2.0, 3.0]))
        assert np.isnan(out[0])

    def test_empty_input(self):
        out = rrc.MonotonicMax(3).process_batch(np.array([]))
        assert len(out) == 0 and out.dtype == np.float64

    def test_rejects_zero_window(self):
        with pytest.raises(ValueError, match="Window length must be greater than 0"):
            rrc.MonotonicMax(0)

    @pytest.mark.parametrize("k", [-1, -5])
    def test_rejects_negative_window(self, k):
        with pytest.raises(TypeError):
            rrc.MonotonicMax(k)

    def test_rejects_none_window(self):
        with pytest.raises(TypeError):
            rrc.MonotonicMax(None)

    def test_rejects_inf_window(self):
        with pytest.raises((ValueError, OverflowError, TypeError)):
            rrc.MonotonicMax(float("inf"))

    def test_rejects_2d_input(self):
        with pytest.raises(RuntimeError, match="Input must be 1D array"):
            rrc.MonotonicMax(2).process_batch(np.ones((2, 3)))

    def test_integer_input_converted_to_float64(self):
        out = rrc.MonotonicMax(2).process_batch(np.array([1, 2, 3], dtype=np.int32))
        assert out.dtype == np.float64


# ── C++ engine — MonotonicMin ──────────────────────────────────────────────────

class TestMonotonicMin:

    def test_known_values(self):
        x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        out = rrc.MonotonicMin(3).process_batch(x)
        np.testing.assert_allclose(out, [1.0, 1.0, 1.0, 2.0, 2.0], atol=1e-12)

    def test_window_size_1_identity(self):
        x = np.array([-1.0, 0.0, 10.0, 2.0])
        np.testing.assert_allclose(rrc.MonotonicMin(1).process_batch(x), x)

    def test_window_larger_than_array_cumulative(self):
        x = np.array([2.0, -3.0, 7.0, 1.0])
        np.testing.assert_allclose(rrc.MonotonicMin(20).process_batch(x),
                                   np.minimum.accumulate(x))

    def test_increasing_sequence(self):
        x = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        np.testing.assert_allclose(rrc.MonotonicMin(3).process_batch(x),
                                   [1.0, 1.0, 1.0, 3.0, 5.0])

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_against_naive_reference(self, k):
        x = np.array([-2.0, 6.0, 1.0, 8.0, 0.0, 8.0, -1.0])
        out = rrc.MonotonicMin(k).process_batch(x)
        expected = np.array([np.min(x[max(0, i - k + 1):i + 1]) for i in range(len(x))])
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_nan_advances_window_returns_current_min(self):
        # window=2: [5,1,NaN,9]. At NaN: window=[1,NaN], min=1
        x = np.array([5.0, 1.0, np.nan, 9.0])
        out = rrc.MonotonicMin(2).process_batch(x)
        assert out[2] == 1.0

    def test_nan_at_start_empty_window_returns_nan(self):
        out = rrc.MonotonicMin(2).process_batch(np.array([np.nan, 2.0, 3.0]))
        assert np.isnan(out[0])

    def test_empty_input(self):
        out = rrc.MonotonicMin(3).process_batch(np.array([]))
        assert len(out) == 0

    def test_rejects_zero_window(self):
        with pytest.raises(ValueError, match="Window length must be greater than 0"):
            rrc.MonotonicMin(0)

    def test_rejects_2d_input(self):
        with pytest.raises(RuntimeError, match="Input must be 1D array"):
            rrc.MonotonicMin(2).process_batch(np.ones((2, 3)))


# ── C++ engine — MultisetMedian ───────────────────────────────────────────────

class TestMultisetMedian:

    def test_known_values_odd_window(self):
        x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        np.testing.assert_allclose(rrc.MultisetMedian(3).process_batch(x),
                                   _median_ref(x, 3), rtol=1e-12)

    def test_known_values_even_window(self):
        x = np.array([1.0, 3.0, 2.0, 4.0])
        np.testing.assert_allclose(rrc.MultisetMedian(4).process_batch(x),
                                   _median_ref(x, 4), rtol=1e-12)

    def test_window_size_1_identity(self):
        x = np.array([-1.0, 0.0, 10.0, 2.0])
        np.testing.assert_allclose(rrc.MultisetMedian(1).process_batch(x), x)

    def test_window_size_2_regression(self):
        # Regression: window_size=2 caused segfault before fix
        x = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
        np.testing.assert_allclose(rrc.MultisetMedian(2).process_batch(x),
                                   _median_ref(x, 2), rtol=1e-12)

    def test_even_window_descending_fill_regression(self):
        # Regression: even window filled descending caused wrong mid_ position
        x = np.array([4.0, 3.0, 2.0, 1.0])
        np.testing.assert_allclose(rrc.MultisetMedian(4).process_batch(x),
                                   _median_ref(x, 4), rtol=1e-12)

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_against_naive_reference(self, k):
        x = np.array([-2.0, 6.0, 1.0, -8.0, 0.0, 8.0, -1.0])
        np.testing.assert_allclose(rrc.MultisetMedian(k).process_batch(x),
                                   _median_ref(x, k), rtol=1e-12)

    def test_large_array_against_reference(self):
        np.random.seed(42)
        x = np.random.randn(1000)
        k = 15
        np.testing.assert_allclose(rrc.MultisetMedian(k).process_batch(x),
                                   _median_ref(x, k), rtol=1e-10)

    def test_element_entering_equals_leaving(self):
        x = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(rrc.MultisetMedian(3).process_batch(x),
                                   _median_ref(x, 3), rtol=1e-12)

    def test_window_larger_than_array(self):
        x = np.array([3.0, 1.0, 4.0, 1.0])
        np.testing.assert_allclose(rrc.MultisetMedian(20).process_batch(x),
                                   _median_ref(x, 20), rtol=1e-12)

    def test_empty_input(self):
        out = rrc.MultisetMedian(3).process_batch(np.array([]))
        assert len(out) == 0 and out.dtype == np.float64

    def test_rejects_zero_window(self):
        with pytest.raises(ValueError, match="Window length must be greater than 0"):
            rrc.MultisetMedian(0)

    def test_rejects_none_window(self):
        with pytest.raises(TypeError):
            rrc.MultisetMedian(None)

    def test_rejects_2d_input(self):
        with pytest.raises(RuntimeError, match="Input must be 1D array"):
            rrc.MultisetMedian(2).process_batch(np.ones((2, 3)))

    def test_integer_input_converted_to_float64(self):
        out = rrc.MultisetMedian(2).process_batch(np.array([1, 2, 3], dtype=np.int32))
        assert out.dtype == np.float64

# ── C++ engine — SlidingMean ─────────────────────────────────────────────────────

def _mean_ref(x, k):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - k + 1):i + 1]
        valid = w[~np.isnan(w)]
        if len(valid) >= 1:
            out[i] = valid.mean()
    return out


class TestSlidingMean:

    def test_known_values(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = rrc.SlidingMean(3).process_batch(x)
        _nan_allclose(out, [1.0, 1.5, 2.0, 3.0, 4.0])

    def test_window_size_1_identity(self):
        x = np.array([3.0, 7.0, -1.0])
        out = rrc.SlidingMean(1).process_batch(x)
        _nan_allclose(out, x)

    def test_window_larger_than_array(self):
        out = rrc.SlidingMean(10).process_batch(np.array([2.0, 4.0, 6.0]))
        _nan_allclose(out, [2.0, 3.0, 4.0])

    def test_constant_sequence(self):
        out = rrc.SlidingMean(3).process_batch(np.full(6, 5.0))
        np.testing.assert_allclose(out, 5.0, atol=1e-12)

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_against_naive_reference(self, k):
        x = np.array([-3.0, -1.0, 0.0, 2.0, 10.0, 7.0, 7.0, 8.0])
        out = rrc.SlidingMean(k).process_batch(x)
        np.testing.assert_allclose(out, _mean_ref(x, k), rtol=1e-12, atol=1e-12,
                                   equal_nan=True)

    def test_nan_does_not_contribute(self):
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        out = rrc.SlidingMean(3).process_batch(x)
        # window [1, nan, 3] → mean of [1, 3] = 2.0
        assert np.isclose(out[2], 2.0, atol=1e-12)

    def test_nan_advances_window(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        out = rrc.SlidingMean(3).process_batch(x)
        # window [2, nan, 4] → mean of [2, 4] = 3.0
        assert np.isclose(out[3], 3.0, atol=1e-12)

    def test_all_nan_returns_nan(self):
        x = np.array([np.nan, np.nan, np.nan])
        out = rrc.SlidingMean(2).process_batch(x)
        assert np.all(np.isnan(out))

    def test_empty_input(self):
        out = rrc.SlidingMean(3).process_batch(np.array([]))
        assert len(out) == 0 and out.dtype == np.float64

    def test_rejects_zero_window(self):
        with pytest.raises(ValueError, match="Window length must be greater than 0"):
            rrc.SlidingMean(0)

    def test_rejects_2d_input(self):
        with pytest.raises(RuntimeError, match="Input must be 1D array"):
            rrc.SlidingMean(2).process_batch(np.ones((2, 3)))

    def test_integer_input_converted_to_float64(self):
        out = rrc.SlidingMean(2).process_batch(np.array([1, 2, 3, 4], dtype=np.int32))
        assert out.dtype == np.float64

    def test_min_periods_masks_warmup(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = rrc.SlidingMean(3).process_batch(x, min_periods=3)
        assert np.isnan(out[0]) and np.isnan(out[1])
        assert np.isclose(out[2], 2.0, atol=1e-12)

    def test_min_periods_with_nan_input(self):
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        out = rrc.SlidingMean(3).process_batch(x, min_periods=2)
        # pos 1: window [1, nan] → 1 non-NaN < 2 → NaN
        assert np.isnan(out[1])
        # pos 2: window [1, nan, 3] → 2 non-NaN >= 2 → 2.0
        assert np.isclose(out[2], 2.0, atol=1e-12)


# ── C++ engine — SlidingMoments ──────────────────────────────────────────────────

class TestSlidingMoments:

    def test_initial_state_is_nan(self):
        sm = rrc.SlidingMoments(4)
        assert sm.current_size() == 0
        assert math.isnan(sm.get_skewness())
        assert math.isnan(sm.get_kurtosis())

    def test_symmetric_window_skewness_is_zero(self):
        # [1, 2, 3] is symmetric -> skewness = 0
        sm = rrc.SlidingMoments(3)
        for v in [1.0, 2.0, 3.0]:
            sm.update(v)
        assert pytest.approx(sm.get_skewness(), abs=1e-12) == 0.0
        assert math.isnan(sm.get_kurtosis())  # n=3 < 4

    def test_known_kurtosis_uniform_window(self):
        # [1, 2, 3, 4]: excess kurtosis = -1.2 (uniform distribution)
        sm = rrc.SlidingMoments(4)
        for v in [1.0, 2.0, 3.0, 4.0]:
            sm.update(v)
        assert pytest.approx(sm.get_kurtosis(), abs=1e-10) == -1.2

    def test_nan_advances_window_and_reduces_size(self):
        sm = rrc.SlidingMoments(4)
        for v in [1.0, 2.0, 3.0, 4.0]:
            sm.update(v)
        assert sm.current_size() == 4
        assert not math.isnan(sm.get_kurtosis())

        sm.update(float('nan'))
        assert sm.current_size() == 3
        assert not math.isnan(sm.get_skewness())
        assert math.isnan(sm.get_kurtosis())  # n=3 < 4

    def test_nan_does_not_corrupt_state(self):
        sm = rrc.SlidingMoments(3)
        for v in [10.0, 20.0, 30.0]:
            sm.update(v)
        skew_before = sm.get_skewness()

        for _ in range(3):
            sm.update(float('nan'))
        assert sm.current_size() == 0

        for v in [10.0, 20.0, 30.0]:
            sm.update(v)
        assert pytest.approx(sm.get_skewness(), abs=1e-10) == skew_before

    def test_nan_behavior_matches_pandas(self):
        data = [1.0, 2.0, 5.0, 7.0, np.nan, np.nan, 8.0, 1.0, 3.0]
        window_size = 4

        s = pd.Series(data)
        expected_skew = s.rolling(window_size, min_periods=3).skew()
        expected_kurt = s.rolling(window_size, min_periods=4).kurt()
        expected_count = s.rolling(window_size, min_periods=0).count()

        sm = rrc.SlidingMoments(window_size)

        for i, val in enumerate(data):
            sm.update(val)
            assert sm.current_size() == expected_count[i], f"count mismatch at {i}"

            cpp_skew, pd_skew = sm.get_skewness(), expected_skew[i]
            if pd.isna(pd_skew):
                assert math.isnan(cpp_skew), f"skew should be NaN at {i}"
            else:
                assert pytest.approx(cpp_skew, abs=1e-5) == pd_skew, f"skew mismatch at {i}"

            cpp_kurt, pd_kurt = sm.get_kurtosis(), expected_kurt[i]
            if pd.isna(pd_kurt):
                assert math.isnan(cpp_kurt), f"kurtosis should be NaN at {i}"
            else:
                assert pytest.approx(cpp_kurt, abs=1e-5) == pd_kurt, f"kurtosis mismatch at {i}"

    def test_randomized_fuzzing_against_pandas(self):
        np.random.seed(42)
        window_size = int(np.random.randint(4, 20))
        data = np.random.randn(1000) * 50.0
        data[np.random.rand(1000) < 0.15] = np.nan

        s = pd.Series(data)
        expected_skew = s.rolling(window=window_size, min_periods=3).skew()
        expected_kurt = s.rolling(window=window_size, min_periods=4).kurt()
        expected_count = s.rolling(window=window_size, min_periods=0).count()

        sm = rrc.SlidingMoments(window_size)

        for i, val in enumerate(data):
            sm.update(val)
            assert sm.current_size() == expected_count[i], f"count mismatch at {i}"

            cpp_skew, pd_skew = sm.get_skewness(), expected_skew[i]
            if pd.isna(pd_skew):
                assert math.isnan(cpp_skew), f"skewness should be NaN at {i}"
            else:
                assert pytest.approx(cpp_skew, rel=1e-4, abs=1e-6) == pd_skew, \
                    f"skewness mismatch at {i}: expected {pd_skew}, got {cpp_skew}"

            cpp_kurt, pd_kurt = sm.get_kurtosis(), expected_kurt[i]
            if pd.isna(pd_kurt):
                assert math.isnan(cpp_kurt), f"kurtosis should be NaN at {i}"
            else:
                assert pytest.approx(cpp_kurt, rel=1e-4, abs=1e-6) == pd_kurt, \
                    f"kurtosis mismatch at {i}: expected {pd_kurt}, got {cpp_kurt}"

# ── High-level API — output type preservation ─────────────────────────────────

_ALL_FNS = [rr.rolling_max, rr.rolling_min, rr.rolling_median, rr.rolling_variance]


class TestOutputType:

    @pytest.mark.parametrize("fn", _ALL_FNS)
    def test_series_input_returns_series(self, fn):
        s = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0], name="x")
        out = fn(s, 3)
        assert isinstance(out, pd.Series)
        assert out.name == "x"

    @pytest.mark.parametrize("fn", _ALL_FNS)
    def test_series_preserves_datetime_index(self, fn):
        idx = pd.date_range("2024-01-01", periods=5)
        s = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0], index=idx)
        out = fn(s, 3)
        assert isinstance(out, pd.Series)
        assert out.index.equals(idx)

    @pytest.mark.parametrize("fn", _ALL_FNS)
    def test_series_preserves_range_index(self, fn):
        idx = pd.RangeIndex(start=10, stop=15)
        s = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0], index=idx)
        assert fn(s, 3).index.equals(idx)

    @pytest.mark.parametrize("fn", _ALL_FNS)
    def test_ndarray_input_returns_ndarray(self, fn):
        arr = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        assert isinstance(fn(arr, 3), np.ndarray)

    @pytest.mark.parametrize("fn", _ALL_FNS)
    def test_output_length_equals_input_length(self, fn):
        arr = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        assert len(fn(arr, 3)) == len(arr)


# ── High-level API — pandas value comparison ──────────────────────────────────
#
# Each case is (data, k, min_periods).  None -> default (= window_size).
# Tested against the equivalent pandas rolling call.

_PANDAS_CASES = [
    # label,                          data,                                    k   mp
    ("clean_default",   [1.0, 3.0, 2.0, 5.0, 4.0],                          3,  None),
    ("clean_mp1",       [1.0, 3.0, 2.0, 5.0, 4.0],                          3,  1),
    ("clean_mp2",       [1.0, 3.0, 2.0, 5.0, 4.0],                          3,  2),
    ("longer_default",  [-2.0, 6.0, 1.0, 8.0, 0.0, 8.0, -1.0],             4,  None),
    ("longer_mp2",      [-2.0, 6.0, 1.0, 8.0, 0.0, 8.0, -1.0],             4,  2),
    ("nan_mp1",         [1.0, np.nan, 3.0, 4.0, 5.0],                        3,  1),
    ("nan_mp2",         [1.0, np.nan, 3.0, 4.0, 5.0],                        3,  2),
    ("nan_default",     [1.0, np.nan, 3.0, 4.0, 5.0],                        3,  None),
    ("leading_nans",    [np.nan, np.nan, 3.0, 4.0],                          2,  1),
    ("k_gt_n",          [1.0, 2.0, 3.0],                                     10, 1),
    ("constant",        [5.0] * 6,                                            3,  None),
    ("negatives_mp1",   [-5.0, -1.0, -3.0, -2.0, -4.0],                     3,  1),
    ("mixed_nan_mp1",   [np.nan, 1.0, np.nan, 3.0, np.nan],                  2,  1),
    ("window2_nan_gap", [5.0, 1.0, np.nan, 0.0],                             2,  1),
]

_CASE_IDS = [c[0] for c in _PANDAS_CASES]
_CASE_PARAMS = [(c[1], c[2], c[3]) for c in _PANDAS_CASES]  # (data, k, mp)


def _pd_mp(mp, k):
    """Resolve None -> k for pandas min_periods argument."""
    return k if mp is None else mp


class TestPandasComparison:
    """rolling_max/min/median/variance must produce values identical to pandas."""

    @pytest.mark.parametrize("data,k,mp", _CASE_PARAMS, ids=_CASE_IDS)
    def test_rolling_max_matches_pandas(self, data, k, mp):
        arr = np.array(data, dtype=np.float64)
        expected = pd.Series(arr).rolling(k, min_periods=_pd_mp(mp, k)).max().to_numpy()
        kw = {} if mp is None else {"min_periods": mp}
        _nan_allclose(rr.rolling_max(arr, k, **kw), expected)

    @pytest.mark.parametrize("data,k,mp", _CASE_PARAMS, ids=_CASE_IDS)
    def test_rolling_min_matches_pandas(self, data, k, mp):
        arr = np.array(data, dtype=np.float64)
        expected = pd.Series(arr).rolling(k, min_periods=_pd_mp(mp, k)).min().to_numpy()
        kw = {} if mp is None else {"min_periods": mp}
        _nan_allclose(rr.rolling_min(arr, k, **kw), expected)

    @pytest.mark.parametrize("data,k,mp", _CASE_PARAMS, ids=_CASE_IDS)
    def test_rolling_median_matches_pandas(self, data, k, mp):
        arr = np.array(data, dtype=np.float64)
        expected = pd.Series(arr).rolling(k, min_periods=_pd_mp(mp, k)).median().to_numpy()
        kw = {} if mp is None else {"min_periods": mp}
        _nan_allclose(rr.rolling_median(arr, k, **kw), expected)

    @pytest.mark.parametrize("data,k,mp", _CASE_PARAMS, ids=_CASE_IDS)
    def test_rolling_variance_matches_pandas(self, data, k, mp):
        arr = np.array(data, dtype=np.float64)
        expected = pd.Series(arr).rolling(k, min_periods=_pd_mp(mp, k)).var().to_numpy()
        kw = {} if mp is None else {"min_periods": mp}
        _nan_allclose(rr.rolling_variance(arr, k, **kw), expected)

    @pytest.mark.parametrize("data,k,mp", _CASE_PARAMS, ids=_CASE_IDS)
    def test_rolling_mean_matches_pandas(self, data, k, mp):
        arr = np.array(data, dtype=np.float64)
        expected = pd.Series(arr).rolling(k, min_periods=_pd_mp(mp, k)).mean().to_numpy()
        kw = {} if mp is None else {"min_periods": mp}
        _nan_allclose(rr.rolling_mean(arr, k, **kw), expected)


# ── High-level API — min_periods edge cases ───────────────────────────────────

class TestMinPeriods:

    def test_default_equals_window_size_max_min_median(self):
        """First k-1 positions are NaN with default min_periods."""
        x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        for fn in (rr.rolling_max, rr.rolling_min, rr.rolling_median):
            out = fn(x, 3)
            assert np.isnan(out[0]) and np.isnan(out[1]) and not np.isnan(out[2])

    def test_default_equals_window_size_variance(self):
        """Variance: first k-1 positions NaN; pos k-1 has valid value."""
        out = rr.rolling_variance(np.array([1.0, 2.0, 3.0, 4.0]), 3)
        assert np.isnan(out[0]) and np.isnan(out[1])
        assert pytest.approx(out[2], abs=1e-10) == 1.0

    def test_min_periods_0_no_masking_on_clean_input(self):
        """min_periods=0 never adds extra NaN for clean (non-NaN) input."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        for fn in (rr.rolling_max, rr.rolling_min, rr.rolling_median):
            assert not np.any(np.isnan(fn(x, 3, min_periods=0)))

    def test_nan_series_default_all_masked(self):
        """[1,nan,3,4] window=3: non_na_count=[1,1,2,2], all < 3 -> all NaN."""
        x = np.array([1.0, np.nan, 3.0, 4.0])
        for fn in (rr.rolling_max, rr.rolling_min, rr.rolling_median):
            assert np.all(np.isnan(fn(x, 3)))

    def test_window_expiry_max_nan_gap(self):
        """Value before NaN gap must expire: [5,1,NaN,0] window=2."""
        x = np.array([5.0, 1.0, np.nan, 0.0])
        out = rr.rolling_max(x, 2, min_periods=1)
        assert pytest.approx(out[2], abs=1e-12) == 1.0  # window=[1,nan], max=1
        assert pytest.approx(out[3], abs=1e-12) == 0.0  # window=[nan,0], max=0

    def test_window_expiry_min_nan_gap(self):
        """Value before NaN gap must expire: [1,5,NaN,9] window=2."""
        x = np.array([1.0, 5.0, np.nan, 9.0])
        out = rr.rolling_min(x, 2, min_periods=1)
        assert pytest.approx(out[2], abs=1e-12) == 5.0  # window=[5,nan], min=5
        assert pytest.approx(out[3], abs=1e-12) == 9.0  # window=[nan,9], min=9

    @pytest.mark.parametrize("fn", _ALL_FNS)
    def test_validation_negative_min_periods(self, fn):
        with pytest.raises(Exception):
            fn(np.array([1.0, 2.0, 3.0]), 3, min_periods=-1)

    @pytest.mark.parametrize("fn", _ALL_FNS)
    def test_validation_min_periods_exceeds_window(self, fn):
        with pytest.raises(Exception):
            fn(np.array([1.0, 2.0, 3.0]), 3, min_periods=4)

    @pytest.mark.parametrize("fn", _ALL_FNS)
    def test_empty_input_returns_empty(self, fn):
        out = fn(np.array([]), 3)
        assert len(out) == 0

    def test_variance_min_periods_1_single_value_is_nan(self):
        """With min_periods=1, pos 0 is still NaN: var needs ≥2 values (ddof=1).
        This NaN comes from the algorithm, not min_periods masking."""
        out = rr.rolling_variance(np.array([1.0, 5.0]), 2, min_periods=1)
        assert np.isnan(out[0])
        assert pytest.approx(out[1], abs=1e-10) == 8.0  # var([1,5]) = 8


# ── Memory layout and dtype robustness ────────────────────────────────────────

_ENGINE_CLASSES = [rrc.MonotonicMax, rrc.MonotonicMin, rrc.MultisetMedian, rrc.SlidingWelford]
_ENGINE_IDS = ["MonotonicMax", "MonotonicMin", "MultisetMedian", "SlidingWelford"]


class TestMemoryLayoutAndDtypes:
    """Verify pybind11 c_style|forcecast handles non-standard memory layouts and dtypes."""

    @pytest.mark.parametrize("cls", _ENGINE_CLASSES, ids=_ENGINE_IDS)
    def test_strided_every_other_element(self, cls):
        """Non-contiguous array (stride=2) must give same result as contiguous copy."""
        x = np.arange(10, dtype=np.float64)
        strided = x[::2]                                # [0,2,4,6,8] — NOT C_CONTIGUOUS
        assert not strided.flags["C_CONTIGUOUS"]
        out_strided = cls(2).process_batch(strided)
        out_contig  = cls(2).process_batch(np.ascontiguousarray(strided))
        np.testing.assert_array_equal(out_strided, out_contig)

    @pytest.mark.parametrize("cls", _ENGINE_CLASSES, ids=_ENGINE_IDS)
    def test_strided_column_slice(self, cls):
        """Column of a 2D array is non-contiguous — must still work correctly."""
        m = np.arange(16.0).reshape(8, 2)
        col = m[:, 0]                                   # [0,2,4,6,8,10,12,14]
        assert not col.flags["C_CONTIGUOUS"]
        out_col    = cls(3).process_batch(col)
        out_contig = cls(3).process_batch(np.ascontiguousarray(col))
        np.testing.assert_array_equal(out_col, out_contig)

    @pytest.mark.parametrize("cls", _ENGINE_CLASSES, ids=_ENGINE_IDS)
    def test_strided_reversed(self, cls):
        """Reverse-strided array (negative step) must work correctly."""
        x = np.arange(8, dtype=np.float64)
        rev = x[::-1]                                   # [7,6,5,4,3,2,1,0]
        assert not rev.flags["C_CONTIGUOUS"]
        out_rev    = cls(2).process_batch(rev)
        out_contig = cls(2).process_batch(np.ascontiguousarray(rev))
        np.testing.assert_array_equal(out_rev, out_contig)

    @pytest.mark.parametrize("dtype", [np.float32, np.int32, np.int64],
                             ids=["float32", "int32", "int64"])
    def test_numeric_dtype_cast_in_engine(self, dtype):
        """C++ engine accepts numeric dtypes via forcecast -> output is always float64."""
        x = np.array([1, 3, 2, 5, 4], dtype=dtype)
        out = rrc.MonotonicMax(3).process_batch(x)
        assert out.dtype == np.float64
        np.testing.assert_allclose(out, rrc.MonotonicMax(3).process_batch(x.astype(np.float64)))

    @pytest.mark.parametrize("fn", _ALL_FNS, ids=["max", "min", "median", "variance"])
    @pytest.mark.parametrize("dtype", [np.float32, np.int64, np.bool_],
                             ids=["float32", "int64", "bool"])
    def test_numeric_dtype_in_high_level_api(self, fn, dtype):
        """High-level API must handle any numeric dtype, returning float64."""
        x = np.array([1, 3, 2, 5, 4], dtype=dtype)
        out = fn(x, 3)
        assert out.dtype == np.float64
        _nan_allclose(out, fn(x.astype(np.float64), 3))

    @pytest.mark.parametrize("fn", _ALL_FNS, ids=["max", "min", "median", "variance"])
    def test_strided_high_level_api(self, fn):
        """High-level API must give same result for strided and contiguous input."""
        base = np.array([1.0, 3.0, 2.0, 7.0, 4.0, 6.0, 5.0, 8.0])
        strided = base[::2]                             # [1,2,4,5] — non-contiguous
        assert not strided.flags["C_CONTIGUOUS"]
        out_s = fn(strided, 2)
        out_c = fn(np.ascontiguousarray(strided), 2)
        _nan_allclose(np.asarray(out_s), np.asarray(out_c))


# ── Fuzz / stress tests ────────────────────────────────────────────────────────

class TestFuzz:
    """Large random-NaN arrays compared against pandas — catches edge cases
    in MultisetMedian mid_ tracking and SlidingWelfordRing under churn."""

    @pytest.mark.parametrize("fn,pd_fn", [
        (rr.rolling_max,      lambda s, k, mp: s.rolling(k, min_periods=mp).max().to_numpy()),
        (rr.rolling_min,      lambda s, k, mp: s.rolling(k, min_periods=mp).min().to_numpy()),
        (rr.rolling_median,   lambda s, k, mp: s.rolling(k, min_periods=mp).median().to_numpy()),
        (rr.rolling_variance, lambda s, k, mp: s.rolling(k, min_periods=mp).var().to_numpy()),
    ], ids=["max", "min", "median", "variance"])
    def test_random_nan_20pct_matches_pandas(self, fn, pd_fn):
        np.random.seed(42)
        x = np.random.randn(5000)
        x[np.random.rand(5000) < 0.2] = np.nan
        k, mp = 15, 5
        _nan_allclose(fn(x, k, min_periods=mp),
                      pd_fn(pd.Series(x), k, mp),
                      rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("fn,pd_fn", [
        (rr.rolling_max,      lambda s, k, mp: s.rolling(k, min_periods=mp).max().to_numpy()),
        (rr.rolling_min,      lambda s, k, mp: s.rolling(k, min_periods=mp).min().to_numpy()),
        (rr.rolling_median,   lambda s, k, mp: s.rolling(k, min_periods=mp).median().to_numpy()),
        (rr.rolling_variance, lambda s, k, mp: s.rolling(k, min_periods=mp).var().to_numpy()),
    ], ids=["max", "min", "median", "variance"])
    def test_consecutive_nan_blocks_matches_pandas(self, fn, pd_fn):
        """Long runs of NaN stress-test window expiry logic."""
        np.random.seed(7)
        x = np.random.randn(2000)
        # Insert 3 blocks of 20 consecutive NaN
        for start in [100, 500, 1400]:
            x[start:start + 20] = np.nan
        k, mp = 10, 3
        _nan_allclose(fn(x, k, min_periods=mp),
                      pd_fn(pd.Series(x), k, mp),
                      rtol=1e-10, atol=1e-10)

    def test_stress_many_instances_no_crash(self):
        """Create and discard 2000 instances — verifies no resource leak crashes."""
        x = np.random.randn(100).astype(np.float64)
        for _ in range(2000):
            rrc.MultisetMedian(10).process_batch(x)


# ── C++ engine — SlidingCovariance ────────────────────────────────────────────

def _cov_ref(x, y, k):
    """Sample covariance (ddof=1) over valid pairs in each rolling window."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        xi = x[max(0, i - k + 1):i + 1]
        yi = y[max(0, i - k + 1):i + 1]
        mask = ~np.isnan(xi) & ~np.isnan(yi)
        xi, yi = xi[mask], yi[mask]
        if len(xi) >= 2:
            out[i] = np.cov(xi, yi, ddof=1)[0, 1]
    return out


def _cor_ref(x, y, k):
    """Pearson correlation over valid pairs in each rolling window."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        xi = x[max(0, i - k + 1):i + 1]
        yi = y[max(0, i - k + 1):i + 1]
        mask = ~np.isnan(xi) & ~np.isnan(yi)
        xi, yi = xi[mask], yi[mask]
        if len(xi) >= 2 and np.std(xi) > 0 and np.std(yi) > 0:
            out[i] = np.corrcoef(xi, yi)[0, 1]
    return out


class TestSlidingCovariance:

    def test_initial_state_is_nan(self):
        sc = rrc.SlidingCovariance(3)
        assert math.isnan(sc.get_covariance())
        assert math.isnan(sc.get_correlation())
        assert math.isnan(sc.get_mean_x())
        assert math.isnan(sc.get_mean_y())

    def test_perfect_positive_correlation(self):
        sc = rrc.SlidingCovariance(3)
        for xi, yi in [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]:
            sc.update(xi, yi)
        assert pytest.approx(sc.get_correlation(), abs=1e-12) == 1.0
        assert pytest.approx(sc.get_covariance(), abs=1e-12) == 2.0  # cov([1,2,3],[2,4,6])

    def test_perfect_negative_correlation(self):
        sc = rrc.SlidingCovariance(3)
        for xi, yi in [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]:
            sc.update(xi, yi)
        assert pytest.approx(sc.get_correlation(), abs=1e-12) == -1.0

    def test_uncorrelated_constant_x(self):
        sc = rrc.SlidingCovariance(4)
        for xi, yi in [(5.0, 1.0), (5.0, 2.0), (5.0, 3.0), (5.0, 4.0)]:
            sc.update(xi, yi)
        # M2_x == 0 -> correlation is NaN
        assert math.isnan(sc.get_correlation())
        # covariance should be ~0
        assert pytest.approx(sc.get_covariance(), abs=1e-12) == 0.0

    def test_window_expiry_removes_old_pairs(self):
        sc = rrc.SlidingCovariance(2)
        sc.update(1.0, 1.0)
        sc.update(2.0, 2.0)
        sc.update(10.0, 10.0)  # window: [(2,2),(10,10)]
        assert pytest.approx(sc.get_mean_x(), abs=1e-12) == 6.0
        assert pytest.approx(sc.get_correlation(), abs=1e-12) == 1.0

    def test_nan_pair_skipped_not_added(self):
        # Window=3: add (1,2),(2,4),(3,6) then NaN pair.
        # NaN removes oldest (1,2) but adds nothing -> valid: [(2,4),(3,6)]
        sc = rrc.SlidingCovariance(3)
        for xi, yi in [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]:
            sc.update(xi, yi)
        sc.update(math.nan, 5.0)
        assert pytest.approx(sc.get_covariance(), abs=1e-12) == 1.0  # cov([2,3],[4,6])
        assert pytest.approx(sc.get_correlation(), abs=1e-12) == 1.0

    def test_single_pair_covariance_is_nan(self):
        sc = rrc.SlidingCovariance(3)
        sc.update(1.0, 2.0)
        assert math.isnan(sc.get_covariance())  # needs >=2 pairs

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_covariance_against_numpy_reference(self, k):
        x = np.array([-3.0, -1.0, 0.0, 2.0, 10.0, 7.0, 7.0, 8.0])
        y = np.array([1.0, 3.0, -1.0, 4.0, 2.0, 6.0, 5.0, 0.0])
        sc = rrc.SlidingCovariance(k)
        out = sc.process_covariance_batch(x, y)
        _nan_allclose(out, _cov_ref(x, y, k), rtol=1e-11, atol=1e-11)

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_correlation_against_numpy_reference(self, k):
        x = np.array([-3.0, -1.0, 0.0, 2.0, 10.0, 7.0, 7.0, 8.0])
        y = np.array([1.0, 3.0, -1.0, 4.0, 2.0, 6.0, 5.0, 0.0])
        sc = rrc.SlidingCovariance(k)
        out = sc.process_correlation_batch(x, y)
        _nan_allclose(out, _cor_ref(x, y, k), rtol=1e-11, atol=1e-11)

    def test_random_nan_fuzzing_against_reference(self):
        np.random.seed(123)
        x = np.random.randn(500)
        y = np.random.randn(500)
        x[np.random.rand(500) < 0.15] = np.nan
        y[np.random.rand(500) < 0.15] = np.nan
        k = 10
        sc = rrc.SlidingCovariance(k)
        cov_out = sc.process_covariance_batch(x, y)
        sc2 = rrc.SlidingCovariance(k)
        cor_out = sc2.process_correlation_batch(x, y)
        _nan_allclose(cov_out, _cov_ref(x, y, k), rtol=1e-9, atol=1e-9)
        _nan_allclose(cor_out, _cor_ref(x, y, k), rtol=1e-9, atol=1e-9)

    def test_process_batch_rejects_length_mismatch(self):
        sc = rrc.SlidingCovariance(3)
        with pytest.raises(RuntimeError, match="same length"):
            sc.process_covariance_batch(np.array([1.0, 2.0]), np.array([1.0]))

    def test_process_batch_rejects_2d_input(self):
        sc = rrc.SlidingCovariance(3)
        with pytest.raises(RuntimeError):
            sc.process_covariance_batch(np.ones((2, 3)), np.ones(3))

    def test_empty_input(self):
        sc = rrc.SlidingCovariance(3)
        out = sc.process_covariance_batch(np.array([]), np.array([]))
        assert len(out) == 0 and out.dtype == np.float64


# ── High-level API — rolling_cov / rolling_cor ────────────────────────────────

class TestRollingCovCor:

    def test_rolling_cov_known_values(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        # cov(x*2, x) = 2 * var(x) = 2 * 1.0 = 2.0
        out = rr.rolling_cov(x, y, 3)
        expected = np.array([np.nan, np.nan, 2.0, 2.0, 2.0])
        _nan_allclose(out, expected)

    def test_rolling_cov_default_min_periods_masks_warmup(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([4.0, 3.0, 2.0, 1.0])
        out = rr.rolling_cov(x, y, 3)
        assert np.isnan(out[0]) and np.isnan(out[1])
        assert not np.isnan(out[2])

    def test_rolling_cor_known_perfect_positive(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = x * 3.0
        out = rr.rolling_cor(x, y, 3)
        np.testing.assert_allclose(out[2:], 1.0, atol=1e-12)

    def test_rolling_cor_known_perfect_negative(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = -x + 6.0
        out = rr.rolling_cor(x, y, 3)
        np.testing.assert_allclose(out[2:], -1.0, atol=1e-12)

    def test_rolling_cov_matches_pandas(self):
        np.random.seed(7)
        x = np.random.randn(200)
        y = np.random.randn(200)
        k = 8
        expected = pd.Series(x).rolling(k, min_periods=k).cov(pd.Series(y)).to_numpy()
        _nan_allclose(rr.rolling_cov(x, y, k), expected, rtol=1e-10, atol=1e-10)

    def test_rolling_cor_matches_pandas(self):
        np.random.seed(7)
        x = np.random.randn(200)
        y = np.random.randn(200)
        k = 8
        expected = pd.Series(x).rolling(k, min_periods=k).corr(pd.Series(y)).to_numpy()
        _nan_allclose(rr.rolling_cor(x, y, k), expected, rtol=1e-10, atol=1e-10)

    def test_rolling_cov_with_nan_matches_pandas(self):
        np.random.seed(42)
        x = np.random.randn(300)
        y = np.random.randn(300)
        x[np.random.rand(300) < 0.15] = np.nan
        y[np.random.rand(300) < 0.15] = np.nan
        k, mp = 7, 3
        expected = pd.Series(x).rolling(k, min_periods=mp).cov(pd.Series(y)).to_numpy()
        _nan_allclose(rr.rolling_cov(x, y, k, min_periods=mp), expected, rtol=1e-9, atol=1e-9)

    def test_rolling_cor_with_nan_matches_pandas(self):
        np.random.seed(42)
        x = np.random.randn(300)
        y = np.random.randn(300)
        x[np.random.rand(300) < 0.15] = np.nan
        y[np.random.rand(300) < 0.15] = np.nan
        k, mp = 7, 3
        expected = pd.Series(x).rolling(k, min_periods=mp).corr(pd.Series(y)).to_numpy()
        _nan_allclose(rr.rolling_cor(x, y, k, min_periods=mp), expected, rtol=1e-9, atol=1e-9)

    def test_series_input_returns_series(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="a")
        t = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], name="b")
        out = rr.rolling_cov(s, t, 3)
        assert isinstance(out, pd.Series)
        assert out.name == "a"

    def test_empty_input(self):
        out = rr.rolling_cov(np.array([]), np.array([]), 3)
        assert len(out) == 0


# ── method="fast" (SlidingMomentsPrefix) ─────────────────────────────────────

class TestFastMethod:

    @pytest.mark.parametrize("fn,pd_fn", [
        (lambda x, k: rr.rolling_variance(x, k, method="fast"),
         lambda s, k: s.rolling(k).var().to_numpy()),
        (lambda x, k: rr.rolling_skewness(x, k, method="fast"),
         lambda s, k: s.rolling(k).skew().to_numpy()),
        (lambda x, k: rr.rolling_kurtosis(x, k, method="fast"),
         lambda s, k: s.rolling(k).kurt().to_numpy()),
    ], ids=["variance", "skewness", "kurtosis"])
    def test_fast_matches_pandas_no_nan(self, fn, pd_fn):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(300)
        k = 10
        _nan_allclose(fn(x, k), pd_fn(pd.Series(x), k), rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("fn_stable,fn_fast", [
        (lambda x, k, mp: rr.rolling_variance(x, k, min_periods=mp),
         lambda x, k, mp: rr.rolling_variance(x, k, min_periods=mp, method="fast")),
        (lambda x, k, mp: rr.rolling_skewness(x, k, min_periods=mp),
         lambda x, k, mp: rr.rolling_skewness(x, k, min_periods=mp, method="fast")),
        (lambda x, k, mp: rr.rolling_kurtosis(x, k, min_periods=mp),
         lambda x, k, mp: rr.rolling_kurtosis(x, k, min_periods=mp, method="fast")),
    ], ids=["variance", "skewness", "kurtosis"])
    def test_fast_agrees_with_stable_with_nan(self, fn_stable, fn_fast):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(300)
        x[rng.random(300) < 0.1] = np.nan
        k, mp = 10, 5
        _nan_allclose(fn_fast(x, k, mp), fn_stable(x, k, mp), rtol=1e-9, atol=1e-9)

    def test_fast_variance_warmup_nan(self):
        x = np.arange(1.0, 11.0)
        out = rr.rolling_variance(x, 5, method="fast")
        assert all(np.isnan(out[:4]))
        assert not np.isnan(out[4])

    def test_fast_skewness_needs_3_obs(self):
        x = np.arange(1.0, 11.0)
        out = rr.rolling_skewness(x, 5, method="fast")
        assert all(np.isnan(out[:4]))

    def test_fast_kurtosis_needs_4_obs(self):
        x = np.arange(1.0, 11.0)
        out = rr.rolling_kurtosis(x, 5, method="fast")
        assert all(np.isnan(out[:4]))

    def test_fast_min_periods(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = rr.rolling_variance(x, 3, min_periods=2, method="fast")
        assert np.isnan(out[0])
        assert not np.isnan(out[1])

    def test_fast_all_nan_window_returns_nan(self):
        x = np.array([np.nan, np.nan, np.nan, 1.0, 2.0, 3.0])
        out = rr.rolling_variance(x, 3, min_periods=1, method="fast")
        assert np.isnan(out[2])
        assert not np.isnan(out[5])

    def test_fast_empty_input(self):
        for fn in [rr.rolling_variance, rr.rolling_skewness, rr.rolling_kurtosis]:
            assert len(fn(np.array([]), 3, method="fast")) == 0