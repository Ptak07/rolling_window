# Count non-NA values inside each rolling window using vectorised cumsum.
.count_non_na <- function(x, window_size) {
  cum <- cumsum(!is.na(x))
  lagged <- c(rep(0L, window_size), cum)[seq_along(x)]
  cum - lagged
}

.apply_min_periods <- function(result, x, window_size, min_periods) {
  if (min_periods == 0L || length(x) == 0L) return(result)
  result[.count_non_na(x, window_size) < min_periods] <- NA_real_
  result
}

.check_window <- function(window_size) {
  if (!is.finite(window_size) || window_size < 1L)
    stop("window_size must be a positive finite integer.", call. = FALSE)
}

.check_min_periods <- function(min_periods, window_size) {
  mp <- suppressWarnings(as.integer(min_periods))
  if (is.na(mp) || mp < 0L || mp > window_size)
    stop("min_periods must be an integer in [0, window_size].", call. = FALSE)
  mp
}

#' @title Rolling Sample Variance
#'
#' @description
#' Computes the rolling sample variance over a numeric vector.
#'
#' @param x A numeric vector of type double.
#' @param window_size Positive integer window length.
#' @param min_periods Minimum number of non-\code{NA} observations required in
#'   a window to return a result. Defaults to \code{window_size} (pandas
#'   semantics). Positions with fewer non-\code{NA} values yield \code{NA}.
#'
#' @return
#' A numeric vector with rolling sample variance values. Entries are
#' \code{NA} when fewer than \code{min_periods} non-\code{NA} observations are
#' present in the window, and \code{NaN} when variance is undefined (fewer
#' than two values).
#'
#' @export
#'
#' @examples
#' x <- as.double(c(1, 2, 3, 4))
#' rolling_variance(x, 3L)
rolling_variance <- function(x, window_size, min_periods = window_size) {
  if (!is.double(x)) stop("x must be a double vector.", call. = FALSE)
  .check_window(window_size)
  mp <- .check_min_periods(min_periods, window_size)
  result <- .Call("rolling_variance_c", x, as.integer(window_size),
                  PACKAGE = "robustrolling")
  .apply_min_periods(result, x, window_size, mp)
}

#' @title Rolling Maximum
#'
#' @description
#' Computes the rolling maximum over a numeric vector using a monotonic deque.
#'
#' @param x A numeric vector of type double.
#' @param window_size Positive integer window length.
#' @param min_periods Minimum number of non-\code{NA} observations required in
#'   a window to return a result. Defaults to \code{window_size}.
#'
#' @return
#' A numeric vector with rolling maximum values.
#'
#' @export
#'
#' @examples
#' x <- as.double(c(1, 3, 2, 5, 4))
#' rolling_max(x, 3L)
rolling_max <- function(x, window_size, min_periods = window_size) {
  if (!is.double(x)) stop("x must be a double vector.", call. = FALSE)
  .check_window(window_size)
  mp <- .check_min_periods(min_periods, window_size)
  result <- .Call("rolling_max_c", x, as.integer(window_size),
                  PACKAGE = "robustrolling")
  .apply_min_periods(result, x, window_size, mp)
}

#' @title Rolling Minimum
#'
#' @description
#' Computes the rolling minimum over a numeric vector using a monotonic deque.
#'
#' @param x A numeric vector of type double.
#' @param window_size Positive integer window length.
#' @param min_periods Minimum number of non-\code{NA} observations required in
#'   a window to return a result. Defaults to \code{window_size}.
#'
#' @return
#' A numeric vector with rolling minimum values.
#'
#' @export
#'
#' @examples
#' x <- as.double(c(1, 3, 2, 5, 4))
#' rolling_min(x, 3L)
rolling_min <- function(x, window_size, min_periods = window_size) {
  if (!is.double(x)) stop("x must be a double vector.", call. = FALSE)
  .check_window(window_size)
  mp <- .check_min_periods(min_periods, window_size)
  result <- .Call("rolling_min_c", x, as.integer(window_size),
                  PACKAGE = "robustrolling")
  .apply_min_periods(result, x, window_size, mp)
}

#' @title Rolling Median
#'
#' @description
#' Computes the rolling median over a numeric vector using an ordered multiset
#' with a tracked median iterator. Time complexity: O(log n) per element.
#'
#' @param x A numeric vector of type double.
#' @param window_size Positive integer window length.
#' @param min_periods Minimum number of non-\code{NA} observations required in
#'   a window to return a result. Defaults to \code{window_size}.
#'
#' @return
#' A numeric vector with rolling median values.
#'
#' @export
#'
#' @examples
#' x <- as.double(c(1, 3, 2, 5, 4))
#' rolling_median(x, 3L)
rolling_median <- function(x, window_size, min_periods = window_size) {
  if (!is.double(x)) stop("x must be a double vector.", call. = FALSE)
  .check_window(window_size)
  mp <- .check_min_periods(min_periods, window_size)
  result <- .Call("rolling_median_c", x, as.integer(window_size),
                  PACKAGE = "robustrolling")
  .apply_min_periods(result, x, window_size, mp)
}

#' @title Rolling Mean
#'
#' @description
#' Computes the rolling mean over a numeric vector.
#'
#' @param x A numeric vector of type double.
#' @param window_size Positive integer window length.
#' @param min_periods Minimum number of non-\code{NA} observations required in
#'   a window to return a result. Defaults to \code{window_size}.
#'
#' @return A numeric vector with rolling mean values.
#'
#' @export
#'
#' @examples
#' x <- as.double(c(1, 2, 3, 4))
#' rolling_mean(x, 3L)
rolling_mean <- function(x, window_size, min_periods = window_size) {
  if (!is.double(x)) stop("x must be a double vector.", call. = FALSE)
  .check_window(window_size)
  mp <- .check_min_periods(min_periods, window_size)
  result <- .Call("rolling_mean_c", x, as.integer(window_size),
                  PACKAGE = "robustrolling")
  .apply_min_periods(result, x, window_size, mp)
}

#' @title Rolling Skewness
#'
#' @description
#' Computes the rolling adjusted Fisher-Pearson skewness over a numeric vector.
#' Requires at least 3 non-\code{NA} observations per window.
#'
#' @param x A numeric vector of type double.
#' @param window_size Positive integer window length.
#' @param min_periods Minimum number of non-\code{NA} observations required in
#'   a window to return a result. Defaults to \code{window_size}.
#'
#' @return A numeric vector with rolling skewness values.
#'
#' @export
#'
#' @examples
#' x <- as.double(c(1, 2, 3, 4, 5))
#' rolling_skewness(x, 3L)
rolling_skewness <- function(x, window_size, min_periods = window_size) {
  if (!is.double(x)) stop("x must be a double vector.", call. = FALSE)
  .check_window(window_size)
  mp <- .check_min_periods(min_periods, window_size)
  result <- .Call("rolling_skewness_c", x, as.integer(window_size),
                  PACKAGE = "robustrolling")
  .apply_min_periods(result, x, window_size, mp)
}

#' @title Rolling Kurtosis
#'
#' @description
#' Computes the rolling excess kurtosis (Fisher) over a numeric vector.
#' Requires at least 4 non-\code{NA} observations per window.
#'
#' @param x A numeric vector of type double.
#' @param window_size Positive integer window length.
#' @param min_periods Minimum number of non-\code{NA} observations required in
#'   a window to return a result. Defaults to \code{window_size}.
#'
#' @return A numeric vector with rolling excess kurtosis values.
#'
#' @export
#'
#' @examples
#' x <- as.double(c(1, 2, 3, 4, 5))
#' rolling_kurtosis(x, 4L)
rolling_kurtosis <- function(x, window_size, min_periods = window_size) {
  if (!is.double(x)) stop("x must be a double vector.", call. = FALSE)
  .check_window(window_size)
  mp <- .check_min_periods(min_periods, window_size)
  result <- .Call("rolling_kurtosis_c", x, as.integer(window_size),
                  PACKAGE = "robustrolling")
  .apply_min_periods(result, x, window_size, mp)
}
