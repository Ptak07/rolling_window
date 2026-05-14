rolling_variance_ref <- function(x, k) {
	n <- length(x)
	out <- rep(NA_real_, n)
	if (n == 0) return(out)
	for (i in seq_len(n)) {
		left <- max(1L, i - k + 1L)
		window <- x[left:i]
		non_na <- window[!is.na(window)]
		if (length(non_na) >= 2L) out[i] <- stats::var(non_na)
	}
	out
}

# ---- baseline: default min_periods = window_size ----
x <- as.double(c(1, 2, 3, 4))
res <- robustrolling::rolling_variance(x, 3L)
expect_true(is.double(res))
expect_equal(length(res), length(x))
expect_true(all(is.na(res[1:2])))   # first 2 positions: < 3 non-NA
expect_equal(res[3], 1.0, tolerance = 1e-12)
expect_equal(res[4], 1.0, tolerance = 1e-12)

# Window size 1 always returns NaN (variance undefined for a single point)
res_k1 <- robustrolling::rolling_variance(as.double(c(10, 11, 12)), 1L)
expect_true(all(is.nan(res_k1)))

# Window larger than series with min_periods = 1: cumulative variance
x_small <- as.double(c(2, 4, 6))
expect_equal(
	robustrolling::rolling_variance(x_small, 10L, min_periods = 1L),
	rolling_variance_ref(x_small, 10L),
	tolerance = 1e-12
)

# Stable sequence: zero variance once window is full
flat <- as.double(rep(5, 8))
flat_res <- robustrolling::rolling_variance(flat, 4L)
expect_true(all(is.na(flat_res[1:3])))    # first 3 positions: < 4 non-NA
expect_equal(flat_res[4:8], rep(0, 5), tolerance = 1e-12)

# Cross-check against reference (min_periods = 1 to include warm-up)
x_ref <- as.double(c(-3, -1, 0, 2, 10, 7, 7, 8))
for (k in c(2L, 3L, 5L)) {
	expect_equal(
		robustrolling::rolling_variance(x_ref, k, min_periods = 1L),
		rolling_variance_ref(x_ref, k),
		tolerance = 1e-12
	)
}

# NA input: position with NA gets NA when min_periods is not met
x_na <- as.double(c(1, 2, NA_real_, 4, 5))
res_na <- robustrolling::rolling_variance(x_na, 3L)
expect_true(is.na(res_na[3]))

# Empty input
expect_equal(robustrolling::rolling_variance(numeric(0), 3L), numeric(0))

# Input validation
expect_true(is.double(robustrolling::rolling_variance(1:5, 3L)))
expect_error(robustrolling::rolling_variance(as.double(1:5), 0L))
expect_error(robustrolling::rolling_variance(as.double(1:5), -2L))
expect_error(robustrolling::rolling_variance(as.double(1:5), NA_integer_))
expect_error(robustrolling::rolling_variance(as.double(1:5), Inf))

# ---- min_periods tests ----

# min_periods = 2: outputs once 2 non-NA values are in window
x <- as.double(c(1, 2, 3, 4))
res_mp2 <- robustrolling::rolling_variance(x, 3L, min_periods = 2L)
expect_true(is.na(res_mp2[1]))          # only 1 non-NA in window
expect_equal(res_mp2[2], 0.5, tolerance = 1e-12)  # var([1,2])
expect_equal(res_mp2[3], 1.0, tolerance = 1e-12)
expect_equal(res_mp2[4], 1.0, tolerance = 1e-12)

# NaN in series, default min_periods (= window_size = 3)
# non_na_count for window=3: [1, 1, 2, 2] -> all < 3 -> all NA
x_na <- as.double(c(1, NA_real_, 3, 5))
expect_true(all(is.na(robustrolling::rolling_variance(x_na, 3L))))

# NaN in series, min_periods = 2
# non_na_count: [1, 1, 2, 2] -> positions 3,4 have 2 non-NA >= 2
res_na2 <- robustrolling::rolling_variance(x_na, 3L, min_periods = 2L)
expect_true(all(is.na(res_na2[1:2])))
expect_equal(res_na2[3], 2.0, tolerance = 1e-12)  # var(1, 3) = 2
expect_equal(res_na2[4], 2.0, tolerance = 1e-12)  # var(3, 5) = 2

# min_periods = 0: no mask applied; valid positions return a number
# (variance still NaN when < 2 non-NA values present in window by algorithm design)
x_full <- as.double(c(1, 2, 3, 4))
res_mp0 <- robustrolling::rolling_variance(x_full, 3L, min_periods = 0L)
expect_equal(res_mp0, robustrolling::rolling_variance(x_full, 3L, min_periods = 1L),
             tolerance = 1e-12)

# min_periods validation
expect_error(robustrolling::rolling_variance(as.double(1:5), 3L, min_periods = -1L))
expect_error(robustrolling::rolling_variance(as.double(1:5), 3L, min_periods = 4L))
