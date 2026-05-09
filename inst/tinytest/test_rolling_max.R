rolling_max_ref <- function(x, k) {
	n <- length(x)
	out <- rep(NA_real_, n)
	if (n == 0) return(out)
	for (i in seq_len(n)) {
		left <- max(1L, i - k + 1L)
		window <- x[left:i]
		non_na <- window[!is.na(window)]
		if (length(non_na) > 0L) out[i] <- max(non_na)
	}
	out
}

# ---- baseline: default min_periods = window_size ----
x <- as.double(c(1, 3, 2, 5, 4))
res <- robustrolling::rolling_max(x, 3L)
expect_true(is.double(res))
expect_equal(length(res), length(x))
expect_equal(res, c(NA_real_, NA_real_, 3, 5, 5), tolerance = 1e-12)

# k = 1: min_periods = 1, no warm-up NA
x_k1 <- as.double(c(-1, 0, 10, 2))
expect_equal(robustrolling::rolling_max(x_k1, 1L), x_k1, tolerance = 1e-12)

# k larger than n with min_periods = 1 behaves like cumulative max
x_small <- as.double(c(2, -3, 7, 1))
expect_equal(
	robustrolling::rolling_max(x_small, 20L, min_periods = 1L),
	cummax(x_small),
	tolerance = 1e-12
)

# Decreasing sequence
dec <- as.double(c(9, 7, 5, 3, 1))
expect_equal(
	robustrolling::rolling_max(dec, 3L),
	c(NA_real_, NA_real_, 9, 7, 5),
	tolerance = 1e-12
)

# Duplicate values
dup <- as.double(c(4, 4, 4, 4))
expect_equal(
	robustrolling::rolling_max(dup, 2L),
	c(NA_real_, 4, 4, 4),
	tolerance = 1e-12
)

# Cross-check with reference (min_periods = 1 to include warm-up)
x_ref <- as.double(c(-2, 6, 1, 8, 0, 8, -1))
for (k in c(2L, 3L, 5L)) {
	expect_equal(
		robustrolling::rolling_max(x_ref, k, min_periods = 1L),
		rolling_max_ref(x_ref, k),
		tolerance = 1e-12
	)
}

# NA input: position with NA gets NA (not NaN) when min_periods is not met
x_na <- as.double(c(1, 2, NA_real_, 1))
res_na <- robustrolling::rolling_max(x_na, 2L)
expect_true(is.na(res_na[3]))

# Empty input
expect_equal(robustrolling::rolling_max(numeric(0), 3L), numeric(0))

# Input validation
expect_error(robustrolling::rolling_max(1:5, 3L))
expect_error(robustrolling::rolling_max(as.double(1:5), 0L))
expect_error(robustrolling::rolling_max(as.double(1:5), -3L))
expect_error(robustrolling::rolling_max(as.double(1:5), NA_integer_))
expect_error(robustrolling::rolling_max(as.double(1:5), Inf))

# ---- min_periods tests ----

# min_periods = 1: outputs as soon as 1 non-NA in window
x <- as.double(c(1, 3, 2, 5, 4))
expect_equal(
	robustrolling::rolling_max(x, 3L, min_periods = 1L),
	c(1, 3, 3, 5, 5),
	tolerance = 1e-12
)

# NaN in series, default min_periods (= window_size = 3)
# non_na_count for window=3: [1, 1, 2, 2] → all < 3 → all NA
x_na <- as.double(c(1, NA_real_, 3, 4))
expect_equal(
	robustrolling::rolling_max(x_na, 3L),
	rep(NA_real_, 4L)
)

# NaN in series, min_periods = 2 (< window_size = 3)
# non_na_count: [1, 1, 2, 2] → positions 3,4 have 2 non-NA >= 2
res_mp2 <- robustrolling::rolling_max(x_na, 3L, min_periods = 2L)
expect_equal(res_mp2[1:2], c(NA_real_, NA_real_))
expect_equal(res_mp2[3], 3, tolerance = 1e-12)  # max([1,NA,3]) = 3
expect_equal(res_mp2[4], 4, tolerance = 1e-12)  # max([NA,3,4]) = 4

# min_periods = 0: never NA
res_mp0 <- robustrolling::rolling_max(x_na, 3L, min_periods = 0L)
expect_false(any(is.na(res_mp0)))

# min_periods validation
expect_error(robustrolling::rolling_max(as.double(1:5), 3L, min_periods = -1L))
expect_error(robustrolling::rolling_max(as.double(1:5), 3L, min_periods = 4L))
