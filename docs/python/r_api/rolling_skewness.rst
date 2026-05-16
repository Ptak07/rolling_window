Rolling Skewness
================

Computes the rolling adjusted Fisher-Pearson skewness over a numeric vector.
Requires at least 3 non-``NA`` observations per window.

Usage
-----

.. code-block:: r

   rolling_skewness(x, window_size, min_periods = window_size)

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
     - Minimum number of non-``NA`` observations required in a window to return a result. Defaults to ``window_size``.

Returns
-------

A numeric vector with rolling skewness values.

Examples
--------

.. code-block:: r

   x <- as.double(c(1, 2, 3, 4, 5))
   rolling_skewness(x, 3L)

