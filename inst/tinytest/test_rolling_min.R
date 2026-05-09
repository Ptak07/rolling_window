rolling_min_ref <- function(x, k) {
	n <- length(x)
	out <- rep(NA_real_, n)
	if (n == 0) return(out)
	for (i in seq_len(n)) {
		left <- max(1L, i - k + 1L)
		window <- x[left:i]
		non_na <- window[!is.na(window)]
		if (length(non_na) > 0L) out[i] <- min(non_na)
	}
	out
}

# ---- baseline: default min_periods = window_size ----
x <- as.double(c(1, 3, 2, 5, 4))
res <- robustrolling::rolling_min(x, 3L)
expect_true(is.double(res))
expect_equal(length(res), length(x))
expect_equal(res, c(NA_real_, NA_real_, 1, 2, 2), tolerance = 1e-12)

# k = 1: min_periods = 1, no warm-up NA
x_k1 <- as.double(c(-1, 0, 10, 2))
expect_equal(robustrolling::rolling_min(x_k1, 1L), x_k1, tolerance = 1e-12)

# k larger than n with min_periods = 1 behaves like cumulative min
x_small <- as.double(c(2, -3, 7, 1))
expect_equal(
	robustrolling::rolling_min(x_small, 20L, min_periods = 1L),
	cummin(x_small),
	tolerance = 1e-12
)

# Increasing sequence
inc <- as.double(c(1, 3, 5, 7, 9))
expect_equal(
	robustrolling::rolling_min(inc, 3L),
	c(NA_real_, NA_real_, 1, 3, 5),
	tolerance = 1e-12
)

# Decreasing sequence: minimum is always the newest element
dec <- as.double(c(9, 7, 5, 3, 1))
expect_equal(
	robustrolling::rolling_min(dec, 3L),
	c(NA_real_, NA_real_, 5, 3, 1),
	tolerance = 1e-12
)

# Duplicate values
dup <- as.double(c(4, 4, 4, 4))
expect_equal(
	robustrolling::rolling_min(dup, 2L),
	c(NA_real_, 4, 4, 4),
	tolerance = 1e-12
)

# Cross-check with reference (min_periods = 1 to include warm-up)
x_ref <- as.double(c(-2, 6, 1, 8, 0, 8, -1))
for (k in c(2L, 3L, 5L)) {
	expect_equal(
		robustrolling::rolling_min(x_ref, k, min_periods = 1L),
		rolling_min_ref(x_ref, k),
		tolerance = 1e-12
	)
}

# NA input: position with NA gets NA when min_periods is not met
x_na <- as.double(c(1, 2, NA_real_, 1))
res_na <- robustrolling::rolling_min(x_na, 2L)
expect_true(is.na(res_na[3]))

# Empty input
expect_equal(robustrolling::rolling_min(numeric(0), 3L), numeric(0))

# Input validation
expect_error(robustrolling::rolling_min(1:5, 3L))
expect_error(robustrolling::rolling_min(as.double(1:5), 0L))
expect_error(robustrolling::rolling_min(as.double(1:5), -3L))
expect_error(robustrolling::rolling_min(as.double(1:5), NA_integer_))
expect_error(robustrolling::rolling_min(as.double(1:5), Inf))

# ---- min_periods tests ----

# min_periods = 1: outputs as soon as 1 non-NA in window
x <- as.double(c(1, 3, 2, 5, 4))
expect_equal(
	robustrolling::rolling_min(x, 3L, min_periods = 1L),
	c(1, 1, 1, 2, 2),
	tolerance = 1e-12
)

# NaN in series, default min_periods (= window_size = 3)
# non_na_count for window=3: [1, 1, 2, 2] → all < 3 → all NA
x_na <- as.double(c(5, NA_real_, 3, 1))
expect_equal(
	robustrolling::rolling_min(x_na, 3L),
	rep(NA_real_, 4L)
)

# NaN in series, min_periods = 2
# non_na_count: [1, 1, 2, 2] → positions 3,4 have 2 non-NA >= 2
res_mp2 <- robustrolling::rolling_min(x_na, 3L, min_periods = 2L)
expect_equal(res_mp2[1:2], c(NA_real_, NA_real_))
expect_equal(res_mp2[3], 3, tolerance = 1e-12)  # min([5,NA,3]) = 3
expect_equal(res_mp2[4], 1, tolerance = 1e-12)  # min([NA,3,1]) = 1

# min_periods = 0: never NA
res_mp0 <- robustrolling::rolling_min(x_na, 3L, min_periods = 0L)
expect_false(any(is.na(res_mp0)))

# min_periods validation
expect_error(robustrolling::rolling_min(as.double(1:5), 3L, min_periods = -1L))
expect_error(robustrolling::rolling_min(as.double(1:5), 3L, min_periods = 4L))
