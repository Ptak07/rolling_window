rolling_variance_ref <- function(x, k) {
	n <- length(x)
	out <- rep(NaN, n)
	if (n == 0) {
		return(out)
	}

	for (i in seq_len(n)) {
		left <- max(1L, i - k + 1L)
		window <- x[left:i]
		if (length(window) >= 2L) {
			out[i] <- stats::var(window)
		}
	}

	out
}

# Baseline scenario
x <- as.double(c(1, 2, 3, 4))
res <- robustrolling::rolling_variance(x, 3L)
expect_true(is.double(res))
expect_equal(length(res), length(x))
expect_true(is.nan(res[1]))
expect_equal(res[2], 0.5, tolerance = 1e-12)
expect_equal(res[3], 1.0, tolerance = 1e-12)
expect_equal(res[4], 1.0, tolerance = 1e-12)

# Window size equal to one should always return NaN variance
res_k1 <- robustrolling::rolling_variance(as.double(c(10, 11, 12)), 1L)
expect_true(all(is.nan(res_k1)))

# Window larger than series length behaves as cumulative variance
x_small <- as.double(c(2, 4, 6))
res_large_k <- robustrolling::rolling_variance(x_small, 10L)
expect_equal(res_large_k, rolling_variance_ref(x_small, 10L), tolerance = 1e-12)

# Stable sequence should have zero variance once enough points are present
flat <- as.double(rep(5, 8))
flat_res <- robustrolling::rolling_variance(flat, 4L)
expect_true(all(is.nan(flat_res[1:1])))
expect_equal(flat_res[2:8], rep(0, 7), tolerance = 1e-12)

# Cross-check against a simple reference implementation
x_ref <- as.double(c(-3, -1, 0, 2, 10, 7, 7, 8))
for (k in c(2L, 3L, 5L)) {
	expect_equal(
		robustrolling::rolling_variance(x_ref, k),
		rolling_variance_ref(x_ref, k),
		tolerance = 1e-12
	)
}

# NA values: NA slot maps to NaN output (current tick skipped)
x_na <- as.double(c(1, 2, NA_real_, 4, 5))
res_na <- robustrolling::rolling_variance(x_na, 3L)
expect_true(is.nan(res_na[3]))

# Empty input should return empty output
expect_equal(
	robustrolling::rolling_variance(numeric(0), 3L),
	numeric(0)
)

# Input validation (R wrapper + C layer)
expect_error(robustrolling::rolling_variance(1:5, 3L))
expect_error(robustrolling::rolling_variance(as.double(1:5), 0L))
expect_error(robustrolling::rolling_variance(as.double(1:5), -2L))
expect_error(robustrolling::rolling_variance(as.double(1:5), NA_integer_))
expect_error(robustrolling::rolling_variance(as.double(1:5), Inf))
