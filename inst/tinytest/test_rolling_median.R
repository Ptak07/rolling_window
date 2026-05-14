rolling_median_ref <- function(x, k) {
	n <- length(x)
	out <- rep(NA_real_, n)
	if (n == 0) return(out)
	for (i in seq_len(n)) {
		left <- max(1L, i - k + 1L)
		window <- x[left:i]
		non_na <- window[!is.na(window)]
		if (length(non_na) > 0L) out[i] <- median(non_na)
	}
	out
}

# Regression: used to segfault
x_simple <- as.double(c(3.0, 1.0, 2.0))
rolling_median(x_simple, 2L)

# ---- baseline: default min_periods = window_size ----
x <- as.double(c(1, 3, 2, 5, 4))
res <- robustrolling::rolling_median(x, 3L)
expect_true(is.double(res))
expect_equal(length(res), length(x))
expect_equal(res, c(NA_real_, NA_real_, 2, 3, 4), tolerance = 1e-12)

# k = 1: min_periods = 1, returns original series
x_k1 <- as.double(c(-1, 0, 10, 2))
expect_equal(robustrolling::rolling_median(x_k1, 1L), x_k1, tolerance = 1e-12)

# Even window size
x_even <- as.double(c(1, 3, 2, 4))
expect_equal(
	robustrolling::rolling_median(x_even, 4L, min_periods = 1L),
	rolling_median_ref(x_even, 4L),
	tolerance = 1e-12
)

# Window = 2
x_w2 <- as.double(c(1, 3, 2, 5, 4))
expect_equal(
	robustrolling::rolling_median(x_w2, 2L),
	c(NA_real_, 2, 2.5, 3.5, 4.5),
	tolerance = 1e-12
)

# Descending sequence
dec <- as.double(c(9, 7, 5, 3, 1))
expect_equal(
	robustrolling::rolling_median(dec, 3L),
	c(NA_real_, NA_real_, 7, 5, 3),
	tolerance = 1e-12
)

# Ascending sequence
asc <- as.double(c(1, 2, 3, 4, 5))
expect_equal(
	robustrolling::rolling_median(asc, 3L),
	c(NA_real_, NA_real_, 2, 3, 4),
	tolerance = 1e-12
)

# All duplicate values
dup <- as.double(c(5, 5, 5, 5, 5))
expect_equal(
	robustrolling::rolling_median(dup, 3L),
	c(NA_real_, NA_real_, 5, 5, 5),
	tolerance = 1e-12
)

# Element entering equals element leaving
x_eq <- as.double(c(1, 2, 3, 1, 2, 3))
expect_equal(
	robustrolling::rolling_median(x_eq, 3L, min_periods = 1L),
	rolling_median_ref(x_eq, 3L),
	tolerance = 1e-12
)

# k larger than n with min_periods = 1: cumulative median
x_small <- as.double(c(3, 1, 4, 1))
expect_equal(
	robustrolling::rolling_median(x_small, 20L, min_periods = 1L),
	rolling_median_ref(x_small, 20L),
	tolerance = 1e-12
)

# Negative values
x_neg <- as.double(c(-5, -1, -3, -2, -4))
expect_equal(
	robustrolling::rolling_median(x_neg, 3L, min_periods = 1L),
	rolling_median_ref(x_neg, 3L),
	tolerance = 1e-12
)

# Mixed positive and negative (cross-check with reference)
x_mix <- as.double(c(-2, 6, 1, -8, 0, 8, -1))
for (k in c(2L, 3L, 5L)) {
	expect_equal(
		robustrolling::rolling_median(x_mix, k, min_periods = 1L),
		rolling_median_ref(x_mix, k),
		tolerance = 1e-12
	)
}

# Single element
expect_equal(
	robustrolling::rolling_median(as.double(42), 1L),
	as.double(42),
	tolerance = 1e-12
)

# Empty input
expect_equal(robustrolling::rolling_median(numeric(0), 3L), numeric(0))

# Input validation
expect_error(robustrolling::rolling_median(1:5, 3L))
expect_error(robustrolling::rolling_median(as.double(1:5), 0L))
expect_error(robustrolling::rolling_median(as.double(1:5), -1L))

# ---- min_periods tests ----

# min_periods = 1: outputs from the first element
x <- as.double(c(1, 3, 2, 5, 4))
expect_equal(
	robustrolling::rolling_median(x, 3L, min_periods = 1L),
	c(1, 2, 2, 3, 4),
	tolerance = 1e-12
)

# NaN in series, default min_periods (= window_size = 3)
# non_na_count for window=3: [1, 1, 2, 2] -> all < 3 -> all NA
x_na <- as.double(c(1, NA_real_, 3, 4))
expect_true(all(is.na(robustrolling::rolling_median(x_na, 3L))))

# NaN in series, min_periods = 2
# non_na_count: [1, 1, 2, 2] -> positions 3,4 have 2 non-NA >= 2
res_mp2 <- robustrolling::rolling_median(x_na, 3L, min_periods = 2L)
expect_equal(res_mp2[1:2], c(NA_real_, NA_real_))
expect_equal(res_mp2[3], 2,   tolerance = 1e-12)    # median([1,3]) = 2
expect_equal(res_mp2[4], 3.5, tolerance = 1e-12)    # median([3,4]) = 3.5

# min_periods = 0: never NA
res_mp0 <- robustrolling::rolling_median(x_na, 3L, min_periods = 0L)
expect_false(any(is.na(res_mp0)))

# min_periods validation
expect_error(robustrolling::rolling_median(as.double(1:5), 3L, min_periods = -1L))
expect_error(robustrolling::rolling_median(as.double(1:5), 3L, min_periods = 4L))
