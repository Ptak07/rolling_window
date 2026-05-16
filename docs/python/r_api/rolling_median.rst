Rolling Median
==============

Computes the rolling median over a numeric vector using an ordered multiset
with a tracked median iterator. Time complexity: O(log n) per element.

Usage
-----

.. code-block:: r

   rolling_median(x, window_size, min_periods = window_size)

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

A numeric vector with rolling median values.

Examples
--------

.. code-block:: r

   x <- as.double(c(1, 3, 2, 5, 4))
   rolling_median(x, 3L)

