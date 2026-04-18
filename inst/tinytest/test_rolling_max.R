rolling_max_ref <- function(x, k) {
	n <- length(x)
	out <- numeric(n)
	if (n == 0) {
		return(out)
	}

	for (i in seq_len(n)) {
		left <- max(1L, i - k + 1L)
		out[i] <- max(x[left:i])
	}

	out
}

# Baseline scenario
x <- as.double(c(1, 3, 2, 5, 4))
res <- robustrolling::rolling_max(x, 3L)
expect_true(is.double(res))
expect_equal(length(res), length(x))
expect_equal(res, as.double(c(1, 3, 3, 5, 5)), tolerance = 1e-12)

# k = 1 should return original series
x_k1 <- as.double(c(-1, 0, 10, 2))
expect_equal(robustrolling::rolling_max(x_k1, 1L), x_k1, tolerance = 1e-12)

# k larger than n should behave as cumulative max
x_small <- as.double(c(2, -3, 7, 1))
expect_equal(
	robustrolling::rolling_max(x_small, 20L),
	cummax(x_small),
	tolerance = 1e-12
)

# Decreasing and duplicate sequences
dec <- as.double(c(9, 7, 5, 3, 1))
expect_equal(
	robustrolling::rolling_max(dec, 3L),
	as.double(c(9, 9, 9, 7, 5)),
	tolerance = 1e-12
)

dup <- as.double(c(4, 4, 4, 4))
expect_equal(
	robustrolling::rolling_max(dup, 2L),
	as.double(c(4, 4, 4, 4)),
	tolerance = 1e-12
)

# Cross-check with reference implementation
x_ref <- as.double(c(-2, 6, 1, 8, 0, 8, -1))
for (k in c(2L, 3L, 5L)) {
	expect_equal(
		robustrolling::rolling_max(x_ref, k),
		rolling_max_ref(x_ref, k),
		tolerance = 1e-12
	)
}

# NA values: NA slot maps to NaN output (current tick skipped)
x_na <- as.double(c(1, 2, NA_real_, 1))
res_na <- robustrolling::rolling_max(x_na, 2L)
expect_true(is.nan(res_na[3]))

# Empty input should return empty output
expect_equal(robustrolling::rolling_max(numeric(0), 3L), numeric(0))

# Input validation
expect_error(robustrolling::rolling_max(1:5, 3L))
expect_error(robustrolling::rolling_max(as.double(1:5), 0L))
expect_error(robustrolling::rolling_max(as.double(1:5), -3L))
expect_error(robustrolling::rolling_max(as.double(1:5), NA_integer_))
expect_error(robustrolling::rolling_max(as.double(1:5), Inf))
