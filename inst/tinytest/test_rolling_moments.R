rolling_mean_ref <- function(x, k) {
  n <- length(x)
  out <- rep(NA_real_, n)
  for (i in seq_len(n)) {
    w <- x[max(1L, i - k + 1L):i]
    non_na <- w[!is.na(w)]
    if (length(non_na) >= 1L) out[i] <- mean(non_na)
  }
  out
}

skew_adj <- function(x) {
  x <- x[!is.na(x)]
  n <- length(x)
  if (n < 3L) return(NA_real_)
  m <- mean(x)
  g1 <- mean((x - m)^3) / mean((x - m)^2)^1.5
  g1 * sqrt(n * (n - 1L)) / (n - 2L)
}

kurt_excess <- function(x) {
  x <- x[!is.na(x)]
  n <- length(x)
  if (n < 4L) return(NA_real_)
  m <- mean(x)
  g2 <- mean((x - m)^4) / mean((x - m)^2)^2 - 3
  (n - 1L) / ((n - 2L) * (n - 3L)) * ((n + 1L) * g2 + 6)
}

rolling_skew_ref <- function(x, k) {
  n <- length(x)
  out <- rep(NA_real_, n)
  for (i in seq_len(n)) {
    w <- x[max(1L, i - k + 1L):i]
    out[i] <- skew_adj(w)
  }
  out
}

rolling_kurt_ref <- function(x, k) {
  n <- length(x)
  out <- rep(NA_real_, n)
  for (i in seq_len(n)) {
    w <- x[max(1L, i - k + 1L):i]
    out[i] <- kurt_excess(w)
  }
  out
}

# ---- rolling_mean ----

x <- as.double(c(1, 2, 3, 4))
res <- robustrolling::rolling_mean(x, 3L)
expect_true(all(is.na(res[1:2])))
expect_equal(res[3], 2.0, tolerance = 1e-12)   # mean(1,2,3)
expect_equal(res[4], 3.0, tolerance = 1e-12)   # mean(2,3,4)

# NA is skipped, not propagated
x_na <- as.double(c(1, NA_real_, 3, 4))
res_na <- robustrolling::rolling_mean(x_na, 3L, min_periods = 1L)
expect_equal(res_na[2], 1.0, tolerance = 1e-12)  # window=[1,NA]: mean(1)=1
expect_equal(res_na[3], 2.0, tolerance = 1e-12)  # window=[1,NA,3]: mean(1,3)=2

# Cross-check against reference
x_ref <- as.double(c(-3, -1, 0, 2, 10, 7, 7, 8))
for (k in c(2L, 3L, 5L)) {
  expect_equal(
    robustrolling::rolling_mean(x_ref, k, min_periods = 1L),
    rolling_mean_ref(x_ref, k),
    tolerance = 1e-12
  )
}

# Empty input
expect_equal(robustrolling::rolling_mean(numeric(0), 3L), numeric(0))

# Input validation
expect_true(is.double(robustrolling::rolling_mean(1:5, 3L)))
expect_error(robustrolling::rolling_mean(as.double(1:5), 0L))

# ---- rolling_skewness ----

# Symmetric window -> skewness = 0
x_sym <- as.double(c(1, 2, 3))
res_sym <- robustrolling::rolling_skewness(x_sym, 3L)
expect_equal(res_sym[3], 0.0, tolerance = 1e-12)

# Needs at least 3 non-NA values; n=2 -> NA
x2 <- as.double(c(1, 2, 3, 4))
res2 <- robustrolling::rolling_skewness(x2, 2L)
expect_true(all(is.na(res2)))   # window=2 always < 3 -> NA or NaN, never a number

# Cross-check against reference (min_periods = 1 reveals warm-up values)
x_ref <- as.double(c(1, 3, -1, 5, 2, 8, 0))
for (k in c(3L, 4L, 5L)) {
  expect_equal(
    robustrolling::rolling_skewness(x_ref, k, min_periods = 1L),
    rolling_skew_ref(x_ref, k),
    tolerance = 1e-10
  )
}

# NA handling: NA advances window without contributing
x_na <- as.double(c(1, 2, 3, NA_real_, 5))
res_na <- robustrolling::rolling_skewness(x_na, 4L, min_periods = 1L)
expect_equal(res_na[3], skew_adj(c(1, 2, 3)),   tolerance = 1e-10)
expect_equal(res_na[5], skew_adj(c(2, 3, 5)),   tolerance = 1e-10)

# Empty input
expect_equal(robustrolling::rolling_skewness(numeric(0), 3L), numeric(0))

# Input validation
expect_true(is.double(robustrolling::rolling_skewness(1:5, 3L)))
expect_error(robustrolling::rolling_skewness(as.double(1:5), 0L))

# ---- rolling_kurtosis ----

# Known value: [1,2,3,4] excess kurtosis = -1.2
x_known <- as.double(c(1, 2, 3, 4))
res_known <- robustrolling::rolling_kurtosis(x_known, 4L)
expect_true(all(is.na(res_known[1:3])))
expect_equal(res_known[4], -1.2, tolerance = 1e-10)

# Needs at least 4 non-NA values; n=3 -> NA
x3 <- as.double(c(1, 2, 3, 4))
res3 <- robustrolling::rolling_kurtosis(x3, 3L)
expect_true(all(is.na(res3)))   # window=3 always < 4 -> NA or NaN, never a number

# Cross-check against reference
x_ref <- as.double(c(1, 3, -1, 5, 2, 8, 0, 4))
for (k in c(4L, 5L, 6L)) {
  expect_equal(
    robustrolling::rolling_kurtosis(x_ref, k, min_periods = 1L),
    rolling_kurt_ref(x_ref, k),
    tolerance = 1e-10
  )
}

# NA handling
x_na <- as.double(c(1, 2, 3, 4, NA_real_, 6))
res_na <- robustrolling::rolling_kurtosis(x_na, 5L, min_periods = 1L)
expect_equal(res_na[4], kurt_excess(c(1, 2, 3, 4)), tolerance = 1e-10)
expect_equal(res_na[6], kurt_excess(c(2, 3, 4, 6)), tolerance = 1e-10)

# Empty input
expect_equal(robustrolling::rolling_kurtosis(numeric(0), 4L), numeric(0))

# Input validation
expect_true(is.double(robustrolling::rolling_kurtosis(1:5, 4L)))
expect_error(robustrolling::rolling_kurtosis(as.double(1:5), 0L))

# ---- method = "fast" (SlidingMomentsPrefix) ----------------------------------

x_fast <- as.double(c(1, 3, -1, 5, 2, 8, 0, 4))

# fast skewness matches stable on well-conditioned data
for (k in c(4L, 5L, 6L)) {
  expect_equal(
    robustrolling::rolling_skewness(x_fast, k, min_periods = 1L, method = "fast"),
    robustrolling::rolling_skewness(x_fast, k, min_periods = 1L),
    tolerance = 1e-9
  )
}

# fast kurtosis matches stable on well-conditioned data
for (k in c(4L, 5L, 6L)) {
  expect_equal(
    robustrolling::rolling_kurtosis(x_fast, k, min_periods = 1L, method = "fast"),
    robustrolling::rolling_kurtosis(x_fast, k, min_periods = 1L),
    tolerance = 1e-9
  )
}

# warmup NAs — skewness needs 3 obs, kurtosis needs 4
x_lin <- as.double(1:10)
expect_true(all(is.na(robustrolling::rolling_skewness(x_lin, 5L, method = "fast")[1:4])))
expect_true(all(is.na(robustrolling::rolling_kurtosis(x_lin, 5L, method = "fast")[1:4])))

# min_periods respected
x_mp <- as.double(c(1, 2, 3, 4, 5))
out_mp <- robustrolling::rolling_skewness(x_mp, 5L, min_periods = 3L, method = "fast")
expect_true(is.na(out_mp[1]))
expect_true(is.na(out_mp[2]))
expect_false(is.na(out_mp[3]))

# NA in input handled correctly
x_na2 <- as.double(c(1, 2, NA_real_, 4, 5, 6, 7, 8))
expect_equal(
  robustrolling::rolling_skewness(x_na2, 5L, min_periods = 1L, method = "fast"),
  robustrolling::rolling_skewness(x_na2, 5L, min_periods = 1L),
  tolerance = 1e-9
)

# empty input
expect_equal(robustrolling::rolling_skewness(numeric(0), 3L, method = "fast"), numeric(0))
expect_equal(robustrolling::rolling_kurtosis(numeric(0), 4L, method = "fast"), numeric(0))
