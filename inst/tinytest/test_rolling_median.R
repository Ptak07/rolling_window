rolling_median_ref <- function(x, k) {
	n <- length(x)
	out <- numeric(n)
	if (n == 0) {
		return(out)
	}

	for (i in seq_len(n)) {
		left <- max(1L, i - k + 1L)
		out[i] <- median(x[left:i])
	}

	out
}

# Minimalny reproducer segfaultu
x_simple <- as.double(c(3.0, 1.0, 2.0))
rolling_median(x_simple, 2L)

# Baseline scenario — odd window
x <- as.double(c(1, 3, 2, 5, 4))
res <- robustrolling::rolling_median(x, 3L)
expect_true(is.double(res))
expect_equal(length(res), length(x))
expect_equal(res, rolling_median_ref(x, 3L), tolerance = 1e-12)

# k = 1 should return original series
x_k1 <- as.double(c(-1, 0, 10, 2))
expect_equal(robustrolling::rolling_median(x_k1, 1L), x_k1, tolerance = 1e-12)

# Even window size — median is average of two middle elements
x_even <- as.double(c(1, 3, 2, 4))
expect_equal(
	robustrolling::rolling_median(x_even, 4L),
	rolling_median_ref(x_even, 4L),
	tolerance = 1e-12
)

# Even window size, window = 2
x_w2 <- as.double(c(1, 3, 2, 5, 4))
expect_equal(
	robustrolling::rolling_median(x_w2, 2L),
	rolling_median_ref(x_w2, 2L),
	tolerance = 1e-12
)

# Descending sequence
dec <- as.double(c(9, 7, 5, 3, 1))
expect_equal(
	robustrolling::rolling_median(dec, 3L),
	rolling_median_ref(dec, 3L),
	tolerance = 1e-12
)

# Ascending sequence
asc <- as.double(c(1, 2, 3, 4, 5))
expect_equal(
	robustrolling::rolling_median(asc, 3L),
	rolling_median_ref(asc, 3L),
	tolerance = 1e-12
)

# All duplicate values
dup <- as.double(c(5, 5, 5, 5, 5))
expect_equal(
	robustrolling::rolling_median(dup, 3L),
	as.double(c(5, 5, 5, 5, 5)),
	tolerance = 1e-12
)

# Element entering equals element leaving
x_eq <- as.double(c(1, 2, 3, 1, 2, 3))
expect_equal(
	robustrolling::rolling_median(x_eq, 3L),
	rolling_median_ref(x_eq, 3L),
	tolerance = 1e-12
)

# k larger than n — behaves as cumulative median
x_small <- as.double(c(3, 1, 4, 1))
expect_equal(
	robustrolling::rolling_median(x_small, 20L),
	rolling_median_ref(x_small, 20L),
	tolerance = 1e-12
)

# Negative values
x_neg <- as.double(c(-5, -1, -3, -2, -4))
expect_equal(
	robustrolling::rolling_median(x_neg, 3L),
	rolling_median_ref(x_neg, 3L),
	tolerance = 1e-12
)

# Mixed positive and negative
x_mix <- as.double(c(-2, 6, 1, -8, 0, 8, -1))
for (k in c(2L, 3L, 5L)) {
	expect_equal(
		robustrolling::rolling_median(x_mix, k),
		rolling_median_ref(x_mix, k),
		tolerance = 1e-12
	)
}

# Single element vector
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