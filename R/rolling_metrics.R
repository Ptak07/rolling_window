#' @title Rolling Sample Variance
#'
#' @description
#' Computes the rolling sample variance over a numeric vector.
#'
#' @param x A numeric vector of type double.
#' @param window_size Positive integer window length.
#'
#' @return
#' A numeric vector with rolling sample variance values. Entries are
#' `NaN` when fewer than two observations are available in the window.
#'
#' @export
#'
#' @examples
#' x <- as.double(c(1, 2, 3, 4))
#' rolling_variance(x, 3L)
rolling_variance <- function(x, window_size) {
  if (!is.double(x)) {
    stop("x must be a double vector.", call. = FALSE)
  }
  .Call("rolling_variance_c", x, as.integer(window_size), PACKAGE = "robustrolling")
}

#' @title Rolling Maximum
#'
#' @description
#' Computes the rolling maximum over a numeric vector using
#' a monotonic deque.
#'
#' @param x A numeric vector of type double.
#' @param window_size Positive integer window length.
#'
#' @return
#' A numeric vector with rolling maximum values.
#'
#' @export
#'
#' @examples
#' x <- as.double(c(1, 3, 2, 5, 4))
#' rolling_max(x, 3L)
rolling_max <- function(x, window_size) {
  if (!is.double(x)) {
    stop("x must be a double vector.", call. = FALSE)
  }
  .Call("rolling_max_c", x, as.integer(window_size), PACKAGE = "robustrolling")
}