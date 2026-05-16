Rolling Minimum
===============

Computes the rolling minimum over a numeric vector using a monotonic deque.

Usage
-----

.. code-block:: r

   rolling_min(x, window_size, min_periods = window_size)

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

A numeric vector with rolling minimum values.

Examples
--------

.. code-block:: r

   x <- as.double(c(1, 3, 2, 5, 4))
   rolling_min(x, 3L)

