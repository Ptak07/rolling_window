Rolling Maximum
===============

Computes the rolling maximum over a numeric vector using a monotonic deque.

Usage
-----

.. code-block:: r

   rolling_max(x, window_size, min_periods = window_size)

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

A numeric vector with rolling maximum values.

Examples
--------

.. code-block:: r

   x <- as.double(c(1, 3, 2, 5, 4))
   rolling_max(x, 3L)

