robustrolling
=============

.. raw:: html

   <p class="hero">
   High-performance rolling-window statistics for R and Python — six
   algorithms implemented in C++17, exposed through idiomatic bindings in
   both languages, with O(1) or O(log n) updates per element.
   </p>

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api
   r_api/index

Algorithm overview
------------------

.. list-table::
   :header-rows: 1
   :widths: 26 22 12 20 20

   * - C++ class
     - Algorithm
     - Complexity
     - R function(s)
     - Python function(s)
   * - ``SlidingWelfordRing``
     - Welford online variance (ring buffer)
     - O(1)
     - ``rolling_variance``
     - ``rolling_variance``, ``SlidingWelford``
   * - ``MonotonicMax``
     - Monotonic deque maximum
     - O(1) amortised
     - ``rolling_max``
     - ``rolling_max``, ``MonotonicMax``
   * - ``MonotonicMin``
     - Monotonic deque minimum
     - O(1) amortised
     - ``rolling_min``
     - ``rolling_min``, ``MonotonicMin``
   * - ``MultisetMedian``
     - ``std::multiset`` tracked-iterator median
     - O(log n)
     - ``rolling_median``
     - ``rolling_median``, ``MultisetMedian``
   * - ``SlidingMoments``
     - Terriberry 4th-moment online algorithm
     - O(1)
     - ``rolling_mean``, ``rolling_skewness``, ``rolling_kurtosis``
     - ``rolling_mean``, ``rolling_skewness``, ``rolling_kurtosis``, ``SlidingMoments``
   * - ``SlidingCovariance``
     - 2-D Welford online covariance
     - O(1)
     - ``rolling_cov``, ``rolling_cor``
     - ``rolling_cov``, ``rolling_cor``, ``SlidingCovariance``

Install for R
-------------

.. code-block:: r

   # Requires a C++17 compiler (GCC ≥ 7, Clang ≥ 5, MSVC ≥ 2017)
   install.packages("remotes")
   remotes::install_github("IgorPtak/rolling_window")

Install for Python
------------------

.. code-block:: bash

   # Requires Python ≥ 3.8 and a C++17 compiler
   pip install git+https://github.com/IgorPtak/rolling_window.git#subdirectory=py_package

   # Or clone and install locally:
   git clone https://github.com/IgorPtak/rolling_window.git
   pip install rolling_window/py_package/
