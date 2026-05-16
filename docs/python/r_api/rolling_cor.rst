Rolling Correlation
===================

Computes the rolling Pearson correlation between two numeric vectors.

Usage
-----

.. code-block:: r

   rolling_cor(x, y, window_size, min_periods = window_size)

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``x``
     - A numeric vector of type double.
   * - ``y``
     - A numeric vector of type double, same length as ``x``.
   * - ``window_size``
     - Positive integer window length.
   * - ``min_periods``
     - Minimum number of valid (non-``NA``) pairs required. Defaults to ``window_size``.

Returns
-------

A numeric vector with rolling correlation values.

Examples
--------

.. code-block:: r

   x <- as.double(c(1, 2, 3, 4, 5))
   y <- as.double(c(2, 4, 6, 8, 10))
   rolling_cor(x, y, 3L)

