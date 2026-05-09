rolling_min_ref <- function(x, k) {
    n <- length(x)
    out <- numeric(n)
    if (n == 0) return(out)
    for (i in seq_len(n)) {
        left <- max(1L, i - k + 1L)
        out[i] <- min(x[left:i])
    }
    out
}

# Baseline scenario
x <- as.double(c(1, 3, 2, 5, 4))
res <- robustrolling::rolling_min(x, 3L)
expect_true(is.double(res))
expect_equal(length(res), length(x))
expect_equal(res, as.double(c(1, 1, 1, 2, 2)), tolerance = 1e-12)

# k = 1 should return original series
x_k1 <- as.double(c(-1, 0, 10, 2))
expect_equal(robustrolling::rolling_min(x_k1, 1L), x_k1, tolerance = 1e-12)

# k larger than n: cumulative minimum
x_small <- as.double(c(2, -3, 7, 1))
expect_equal(
    robustrolling::rolling_min(x_small, 20L),
    cummin(x_small),
    tolerance = 1e-12
)

# Increasing sequence: minimum slides out
inc <- as.double(c(1, 3, 5, 7, 9))
expect_equal(
    robustrolling::rolling_min(inc, 3L),
    as.double(c(1, 1, 1, 3, 5)),
    tolerance = 1e-12
)

# Decreasing sequence: minimum is always the newest element
dec <- as.double(c(9, 7, 5, 3, 1))
expect_equal(
    robustrolling::rolling_min(dec, 3L),
    as.double(c(9, 7, 5, 3, 1)),
    tolerance = 1e-12
)

# Duplicate values
dup <- as.double(c(4, 4, 4, 4))
expect_equal(
    robustrolling::rolling_min(dup, 2L),
    as.double(c(4, 4, 4, 4)),
    tolerance = 1e-12
)

# Cross-check with reference implementation
x_ref <- as.double(c(-2, 6, 1, 8, 0, 8, -1))
for (k in c(2L, 3L, 5L)) {
    expect_equal(
        robustrolling::rolling_min(x_ref, k),
        rolling_min_ref(x_ref, k),
        tolerance = 1e-12
    )
}

# NA input maps to NaN output at that position
x_na <- as.double(c(1, 2, NA_real_, 1))
res_na <- robustrolling::rolling_min(x_na, 2L)
expect_true(is.nan(res_na[3]))

# Empty input
expect_equal(robustrolling::rolling_min(numeric(0), 3L), numeric(0))

# Input validation
expect_error(robustrolling::rolling_min(1:5, 3L))
expect_error(robustrolling::rolling_min(as.double(1:5), 0L))
expect_error(robustrolling::rolling_min(as.double(1:5), -3L))
expect_error(robustrolling::rolling_min(as.double(1:5), NA_integer_))
expect_error(robustrolling::rolling_min(as.double(1:5), Inf))
