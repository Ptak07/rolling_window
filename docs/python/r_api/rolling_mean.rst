Rolling Mean
============

Computes the rolling mean over a numeric vector.

Usage
-----

.. code-block:: r

   rolling_mean(x, window_size, min_periods = window_size)

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

A numeric vector with rolling mean values.

Examples
--------

.. code-block:: r

   x <- as.double(c(1, 2, 3, 4))
   rolling_mean(x, 3L)

