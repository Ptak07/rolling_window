rolling_cov_ref <- function(x, y, k) {
  n <- length(x)
  out <- rep(NA_real_, n)
  for (i in seq_len(n)) {
    left <- max(1L, i - k + 1L)
    xi <- x[left:i]; yi <- y[left:i]
    mask <- !is.na(xi) & !is.na(yi)
    xi <- xi[mask]; yi <- yi[mask]
    if (length(xi) >= 2L) out[i] <- stats::cov(xi, yi)
  }
  out
}

rolling_cor_ref <- function(x, y, k) {
  n <- length(x)
  out <- rep(NA_real_, n)
  for (i in seq_len(n)) {
    left <- max(1L, i - k + 1L)
    xi <- x[left:i]; yi <- y[left:i]
    mask <- !is.na(xi) & !is.na(yi)
    xi <- xi[mask]; yi <- yi[mask]
    if (length(xi) >= 2L && stats::sd(xi) > 0 && stats::sd(yi) > 0)
      out[i] <- stats::cor(xi, yi)
  }
  out
}

# ---- perfect positive correlation: y = 2x ----
x <- as.double(c(1, 2, 3, 4, 5))
y <- 2.0 * x
res_cov <- robustrolling::rolling_cov(x, y, 3L)
res_cor <- robustrolling::rolling_cor(x, y, 3L)
expect_true(all(is.na(res_cov[1:2])))
expect_true(all(is.na(res_cor[1:2])))
expect_equal(res_cov[3:5], rep(2.0, 3), tolerance = 1e-12)
expect_equal(res_cor[3:5], rep(1.0, 3), tolerance = 1e-12)

# ---- perfect negative correlation: y = -x + 6 ----
y_neg <- -x + 6.0
res_neg <- robustrolling::rolling_cor(x, y_neg, 3L)
expect_equal(res_neg[3:5], rep(-1.0, 3), tolerance = 1e-12)

# ---- cross-check against reference for multiple window sizes ----
x_ref <- as.double(c(-3, -1, 0, 2, 10, 7, 7, 8))
y_ref <- as.double(c(1, 3, -1, 4, 2, 6, 5, 0))
for (k in c(2L, 3L, 5L)) {
  expect_equal(
    robustrolling::rolling_cov(x_ref, y_ref, k, min_periods = 1L),
    rolling_cov_ref(x_ref, y_ref, k),
    tolerance = 1e-11
  )
  expect_equal(
    robustrolling::rolling_cor(x_ref, y_ref, k, min_periods = 1L),
    rolling_cor_ref(x_ref, y_ref, k),
    tolerance = 1e-11
  )
}

# ---- NA handling: pair skipped when either is NA ----
# x_na=(1,NA,NA,4,5), y_na=(2,4,6,8,10): window=3
# Position 3: x=(1,NA,NA), valid pairs=1 < 2 -> NA with min_periods=2
# Position 4: x=(NA,NA,4), valid pairs=1 < 2 -> NA with min_periods=2
# Position 5: x=(NA,4,5), valid pairs=2 >= 2 -> value
x_na <- as.double(c(1, NA_real_, NA_real_, 4, 5))
y_na <- as.double(c(2, 4, 6, 8, 10))
res_na_cov <- robustrolling::rolling_cov(x_na, y_na, 3L, min_periods = 2L)
expect_true(is.na(res_na_cov[3]))
expect_true(is.na(res_na_cov[4]))
expect_false(is.na(res_na_cov[5]))

# ---- default min_periods = window_size masks warm-up ----
res_default <- robustrolling::rolling_cov(x_ref, y_ref, 4L)
expect_true(all(is.na(res_default[1:3])))
expect_false(is.na(res_default[4]))

# ---- window size larger than input: cumulative ----
x_small <- as.double(c(1, 2, 3, 4))
y_small <- as.double(c(2, 4, 6, 8))
expect_equal(
  robustrolling::rolling_cov(x_small, y_small, 10L, min_periods = 1L),
  rolling_cov_ref(x_small, y_small, 10L),
  tolerance = 1e-12
)

# ---- empty input ----
expect_equal(robustrolling::rolling_cov(numeric(0), numeric(0), 3L), numeric(0))
expect_equal(robustrolling::rolling_cor(numeric(0), numeric(0), 3L), numeric(0))

# ---- input validation ----
expect_error(robustrolling::rolling_cov(1:5, as.double(1:5), 3L))       # x not double
expect_error(robustrolling::rolling_cov(as.double(1:5), 1:5, 3L))       # y not double
expect_error(robustrolling::rolling_cov(as.double(1:5), as.double(1:4), 3L))  # length mismatch
expect_error(robustrolling::rolling_cov(as.double(1:5), as.double(1:5), 0L))  # bad window
expect_error(robustrolling::rolling_cov(as.double(1:5), as.double(1:5), 3L, min_periods = -1L))
expect_error(robustrolling::rolling_cov(as.double(1:5), as.double(1:5), 3L, min_periods = 4L))
