Rolling Sample Variance
=======================

Computes the rolling sample variance over a numeric vector.

Usage
-----

.. code-block:: r

   rolling_variance(x, window_size, min_periods = window_size)

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``x``
     - A numeric vector of type double.
   * - ``window_size``
     - Positive integer window length.
   * - ``min_periods``
     - Minimum number of non-``NA`` observations required in a window to return a result. Defaults to ``window_size`` (pandas semantics). Positions with fewer non-``NA`` values yield ``NA``.

Returns
-------

A numeric vector with rolling sample variance values. Entries are
``NA`` when fewer than ``min_periods`` non-``NA`` observations are
present in the window, and ``NaN`` when variance is undefined (fewer
than two values).

Examples
--------

.. code-block:: r

   x <- as.double(c(1, 2, 3, 4))
   rolling_variance(x, 3L)

