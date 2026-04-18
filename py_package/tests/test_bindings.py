import numpy as np
import pytest

import robust_rolling_core as rrc


# SLIDING WELFORD (Rolling Variance) Tests

class TestSlidingWelfordBaseline:
    """Baseline correctness scenarios for SlidingWelford."""

    def test_known_values(self):
        """Test against known variance values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        engine = rrc.SlidingWelford(3)
        out = engine.process_batch(data)

        expected = np.array([np.nan, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        assert np.isnan(out[0])
        np.testing.assert_allclose(out[1:], expected[1:], rtol=1e-12, atol=1e-12)

    def test_return_type_is_float64(self):
        """Output should be float64."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        engine = rrc.SlidingWelford(2)
        out = engine.process_batch(data)

        assert out.dtype == np.float64
        assert len(out) == len(data)

    def test_output_length_matches_input(self):
        """Output array length must equal input length."""
        for n in [1, 5, 10, 100]:
            data = np.random.randn(n).astype(np.float64)
            engine = rrc.SlidingWelford(2)
            out = engine.process_batch(data)
            assert len(out) == n


class TestSlidingWelfordEdgeCases:
    """Edge cases for window sizes and array properties."""

    def test_window_size_one_always_nan(self):
        """Window size 1 should return all NaN (no variance with single point)."""
        data = np.array([10.0, 11.0, 12.0], dtype=np.float64)
        engine = rrc.SlidingWelford(1)
        out = engine.process_batch(data)

        assert np.all(np.isnan(out))

    def test_window_larger_than_array(self):
        """When window > array length, compute cumulative variance."""
        data = np.array([2.0, 4.0, 6.0], dtype=np.float64)
        engine = rrc.SlidingWelford(10)
        out = engine.process_batch(data)

        assert np.isnan(out[0])  # First point: no variance
        assert np.isclose(out[1], 2.0, rtol=1e-12)  # var([2, 4])
        assert np.isclose(out[2], 4.0, rtol=1e-12)  # var([2, 4, 6])

    def test_stable_sequence_zero_variance(self):
        """Constant values should have zero variance once window is full."""
        flat = np.array([5.0] * 8, dtype=np.float64)
        engine = rrc.SlidingWelford(4)
        out = engine.process_batch(flat)

        assert np.isnan(out[0])  # First point
        np.testing.assert_allclose(out[1:], 0.0, atol=1e-12)

    def test_empty_input(self):
        """Empty array should return empty output."""
        data = np.array([], dtype=np.float64)
        engine = rrc.SlidingWelford(3)
        out = engine.process_batch(data)

        assert len(out) == 0
        assert out.dtype == np.float64


class TestSlidingWelfordSequencePatterns:
    """Test various data patterns."""

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_against_reference_implementation(self, k):
        """Verify against reference variance calculation."""
        x = np.array([-3.0, -1.0, 0.0, 2.0, 10.0, 7.0, 7.0, 8.0], dtype=np.float64)
        engine = rrc.SlidingWelford(k)
        out = engine.process_batch(x)

        # Reference implementation: compute variance naively
        expected = np.full_like(x, np.nan)
        for i in range(len(x)):
            left = max(0, i - k + 1)
            window = x[left:i + 1]
            if len(window) >= 2:
                expected[i] = np.var(window, ddof=1)  # Sample variance

        np.testing.assert_allclose(out, expected, rtol=1e-11, atol=1e-11)

    def test_increasing_sequence(self):
        """Test with strictly increasing values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        engine = rrc.SlidingWelford(2)
        out = engine.process_batch(data)

        assert np.isnan(out[0])
        # Variance of [1,2] is 0.5, [2,3] is 0.5, etc.
        np.testing.assert_allclose(out[1:], 0.5, rtol=1e-12)

    def test_duplicate_values(self):
        """Test with repeated identical values."""
        data = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float64)
        engine = rrc.SlidingWelford(2)
        out = engine.process_batch(data)

        assert np.isnan(out[0])
        np.testing.assert_allclose(out[1:], 0.0, atol=1e-12)


class TestSlidingWelfordNaNHandling:
    """Test NaN input handling."""

    def test_nan_passthrough(self):
        """NaN in input should result in NaN in output at that position."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float64)
        engine = rrc.SlidingWelford(3)
        out = engine.process_batch(data)

        assert np.isnan(out[2])

    def test_nan_at_beginning(self):
        """NaN at start should propagate correctly."""
        data = np.array([np.nan, 2.0, 3.0, 4.0], dtype=np.float64)
        engine = rrc.SlidingWelford(2)
        out = engine.process_batch(data)

        assert np.isnan(out[0])
        assert out.shape == data.shape

    def test_multiple_nans(self):
        """Multiple NaN values should each produce NaN output."""
        data = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float64)
        engine = rrc.SlidingWelford(3)
        out = engine.process_batch(data)

        assert np.isnan(out[1])
        assert np.isnan(out[3])


class TestSlidingWelfordInputValidation:
    """Test input validation and error handling."""

    def test_accepts_integer_input_with_conversion(self):
        """Integer arrays are automatically converted to float64."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        engine = rrc.SlidingWelford(2)
        out = engine.process_batch(data)

        assert out.dtype == np.float64
        assert len(out) == len(data)

    def test_rejects_2d_array(self):
        """2D arrays should raise error."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        engine = rrc.SlidingWelford(2)

        with pytest.raises(RuntimeError, match="Input must be 1D array"):
            engine.process_batch(data)

    def test_rejects_3d_array(self):
        """3D arrays should raise error."""
        data = np.arange(8.0).reshape((2, 2, 2))
        engine = rrc.SlidingWelford(2)

        with pytest.raises(RuntimeError):
            engine.process_batch(data)

    @pytest.mark.parametrize("invalid_k", [-1, -5])
    def test_rejects_negative_window_size(self, invalid_k):
        """Negative window size should raise TypeError."""
        with pytest.raises(TypeError):
            rrc.SlidingWelford(invalid_k)

    def test_rejects_zero_window_size(self):
        """Window size 0 should raise ValueError."""
        with pytest.raises(ValueError, match="Window length must be greater than 0"):
            rrc.SlidingWelford(0)

    def test_constructor_rejects_na_window(self):
        """NA/None window values should be rejected."""
        with pytest.raises(TypeError):
            rrc.SlidingWelford(None)

    def test_constructor_rejects_infinite_window(self):
        """Infinite window values should be rejected or handled."""
        # inf cannot be converted to size_t - should raise
        with pytest.raises((ValueError, OverflowError, TypeError)):
            rrc.SlidingWelford(float('inf'))


class TestSlidingWelfordLargeData:
    """Test with larger datasets."""

    def test_large_array(self):
        """Test with large random array."""
        n = 10000
        data = np.random.randn(n).astype(np.float64)
        engine = rrc.SlidingWelford(10)
        out = engine.process_batch(data)

        assert len(out) == n
        assert not np.all(np.isnan(out))  # Most should have valid variance

    def test_very_large_window(self):
        """Test with window size close to array length."""
        data = np.arange(1000.0, dtype=np.float64)
        engine = rrc.SlidingWelford(999)
        out = engine.process_batch(data)

        assert len(out) == 1000
        assert np.isnan(out[0])
        # Last element should have variance of all elements
        assert not np.isnan(out[-1])


# MONOTONIC MAX (Rolling Maximum) Tests

class TestMonotonicMaxBaseline:
    """Baseline correctness scenarios for MonotonicMax."""

    def test_known_values(self):
        """Test against known maximum values."""
        data = np.array([1.0, 2.0, 3.0, 2.0, 5.0], dtype=np.float64)
        engine = rrc.MonotonicMax(3)
        out = engine.process_batch(data)

        expected = np.array([1.0, 2.0, 3.0, 3.0, 5.0], dtype=np.float64)
        np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)

    def test_return_type_is_float64(self):
        """Output should be float64."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        engine = rrc.MonotonicMax(2)
        out = engine.process_batch(data)

        assert out.dtype == np.float64
        assert len(out) == len(data)

    def test_output_length_matches_input(self):
        """Output array length must equal input length."""
        for n in [1, 5, 10, 100]:
            data = np.random.randn(n).astype(np.float64)
            engine = rrc.MonotonicMax(2)
            out = engine.process_batch(data)
            assert len(out) == n


class TestMonotonicMaxEdgeCases:
    """Edge cases for window sizes and array properties."""

    def test_window_size_one_returns_identity(self):
        """Window size 1 should return original series."""
        data = np.array([-1.0, 0.0, 10.0, 2.0], dtype=np.float64)
        engine = rrc.MonotonicMax(1)
        out = engine.process_batch(data)

        np.testing.assert_allclose(out, data, rtol=1e-12)

    def test_window_larger_than_array_cumulative_max(self):
        """Window larger than array should give cumulative max."""
        data = np.array([2.0, -3.0, 7.0, 1.0], dtype=np.float64)
        engine = rrc.MonotonicMax(20)
        out = engine.process_batch(data)

        expected = np.maximum.accumulate(data)
        np.testing.assert_allclose(out, expected, rtol=1e-12)

    def test_empty_input(self):
        """Empty array should return empty output."""
        data = np.array([], dtype=np.float64)
        engine = rrc.MonotonicMax(3)
        out = engine.process_batch(data)

        assert len(out) == 0
        assert out.dtype == np.float64


class TestMonotonicMaxSequencePatterns:
    """Test various data patterns."""

    def test_decreasing_sequence(self):
        """Test with strictly decreasing values."""
        data = np.array([9.0, 7.0, 5.0, 3.0, 1.0], dtype=np.float64)
        engine = rrc.MonotonicMax(3)
        out = engine.process_batch(data)

        expected = np.array([9.0, 9.0, 9.0, 7.0, 5.0], dtype=np.float64)
        np.testing.assert_allclose(out, expected, rtol=1e-12)

    def test_duplicate_values(self):
        """Test with repeated identical values."""
        data = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float64)
        engine = rrc.MonotonicMax(2)
        out = engine.process_batch(data)

        np.testing.assert_allclose(out, data, rtol=1e-12)

    def test_increasing_sequence(self):
        """Test with strictly increasing values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        engine = rrc.MonotonicMax(2)
        out = engine.process_batch(data)

        np.testing.assert_allclose(out, data, rtol=1e-12)

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_against_reference_implementation(self, k):
        """Verify against reference maximum calculation."""
        x = np.array([-2.0, 6.0, 1.0, 8.0, 0.0, 8.0, -1.0], dtype=np.float64)
        engine = rrc.MonotonicMax(k)
        out = engine.process_batch(x)

        # Reference implementation: compute max naively
        expected = np.zeros_like(x)
        for i in range(len(x)):
            left = max(0, i - k + 1)
            expected[i] = np.max(x[left:i + 1])

        np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)

    def test_peak_in_middle(self):
        """Test with a peak in the middle of the sequence."""
        data = np.array([1.0, 2.0, 5.0, 3.0, 2.0], dtype=np.float64)
        engine = rrc.MonotonicMax(3)
        out = engine.process_batch(data)

        expected = np.array([1.0, 2.0, 5.0, 5.0, 5.0], dtype=np.float64)
        np.testing.assert_allclose(out, expected, rtol=1e-12)


class TestMonotonicMaxNaNHandling:
    """Test NaN input handling."""

    def test_nan_passthrough(self):
        """NaN in input should result in NaN in output at that position."""
        data = np.array([1.0, 2.0, np.nan, 1.0], dtype=np.float64)
        engine = rrc.MonotonicMax(2)
        out = engine.process_batch(data)

        assert np.isnan(out[2])

    def test_nan_at_beginning(self):
        """NaN at start should propagate correctly."""
        data = np.array([np.nan, 2.0, 3.0], dtype=np.float64)
        engine = rrc.MonotonicMax(2)
        out = engine.process_batch(data)

        assert np.isnan(out[0])
        assert out.shape == data.shape

    def test_multiple_nans(self):
        """Multiple NaN values should each produce NaN output."""
        data = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float64)
        engine = rrc.MonotonicMax(3)
        out = engine.process_batch(data)

        assert np.isnan(out[1])
        assert np.isnan(out[3])


class TestMonotonicMaxInputValidation:
    """Test input validation and error handling."""

    def test_accepts_integer_input_with_conversion(self):
        """Integer arrays are automatically converted to float64."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        engine = rrc.MonotonicMax(2)
        out = engine.process_batch(data)

        assert out.dtype == np.float64
        assert len(out) == len(data)

    def test_rejects_2d_array(self):
        """2D arrays should raise error."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        engine = rrc.MonotonicMax(2)

        with pytest.raises(RuntimeError, match="Input must be 1D array"):
            engine.process_batch(data)

    def test_rejects_3d_array(self):
        """3D arrays should raise error."""
        data = np.arange(8.0).reshape((2, 2, 2))
        engine = rrc.MonotonicMax(2)

        with pytest.raises(RuntimeError):
            engine.process_batch(data)

    def test_rejects_zero_window_size(self):
        """Window size 0 must raise ValueError."""
        with pytest.raises(ValueError, match="Window length must be greater than 0"):
            rrc.MonotonicMax(0)

    @pytest.mark.parametrize("invalid_k", [-1, -5])
    def test_rejects_negative_window_size(self, invalid_k):
        """Negative window sizes cannot be mapped to std::size_t - raises TypeError."""
        with pytest.raises(TypeError):
            rrc.MonotonicMax(invalid_k)

    def test_constructor_rejects_na_window(self):
        """NA/None window values should be rejected."""
        with pytest.raises(TypeError):
            rrc.MonotonicMax(None)

    def test_constructor_rejects_infinite_window(self):
        """Infinite window values should be rejected or handled."""
        # inf cannot be converted to size_t - should raise
        with pytest.raises((ValueError, OverflowError, TypeError)):
            rrc.MonotonicMax(float('inf'))


class TestMonotonicMaxLargeData:
    """Test with larger datasets."""

    def test_large_array(self):
        """Test with large random array."""
        n = 10000
        data = np.random.randn(n).astype(np.float64)
        engine = rrc.MonotonicMax(10)
        out = engine.process_batch(data)

        assert len(out) == n

    def test_very_large_window(self):
        """Test with window size close to array length."""
        data = np.arange(1000.0, dtype=np.float64)
        engine = rrc.MonotonicMax(999)
        out = engine.process_batch(data)

        assert len(out) == 1000
        # Last element should be cumulative max
        assert out[-1] == np.max(data)