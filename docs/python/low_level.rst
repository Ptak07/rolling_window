Low-level Classes
=================

.. py:currentmodule:: robustrolling

C++17 classes exposed via pybind11. Use them for streaming (one value at a
time) or to read multiple statistics from a single pass.

----

.. py:class:: MonotonicMax(window_size)

   Rolling maximum — monotonic deque, O(1) amortised.

   :param window_size: int

   .. py:method:: update(value: float)
   .. py:method:: get_max() -> float
   .. py:method:: process_batch(x: numpy.ndarray) -> numpy.ndarray

   .. code-block:: python

      mm = MonotonicMax(3)
      mm.update(1.0); mm.update(3.0); mm.update(2.0)
      mm.get_max()  # 3.0

----

.. py:class:: MonotonicMin(window_size)

   Rolling minimum — monotonic deque, O(1) amortised.

   :param window_size: int

   .. py:method:: update(value: float)
   .. py:method:: get_min() -> float
   .. py:method:: process_batch(x: numpy.ndarray) -> numpy.ndarray

----

.. py:class:: MultisetMedian(window_size)

   Rolling median — ``std::multiset`` with tracked iterator, O(log n).
   Even-size windows return the average of the two middle elements.

   :param window_size: int

   .. py:method:: update(value: float)
   .. py:method:: get_median() -> float
   .. py:method:: process_batch(x: numpy.ndarray) -> numpy.ndarray

----

.. py:class:: SlidingWelford(window_size)

   Rolling sample variance (ddof=1) — Welford algorithm with ring buffer,
   O(1).

   :param window_size: int

   .. py:method:: update(value: float)
   .. py:method:: get_variance() -> float
   .. py:method:: process_batch(x: numpy.ndarray) -> numpy.ndarray

   .. code-block:: python

      sw = SlidingWelford(3)
      for v in [1., 2., 3., 4.]:
          sw.update(v)
      sw.get_variance()  # 1.0

----

.. py:class:: SlidingMoments(window_size)

   Rolling mean, skewness, and excess kurtosis — Terriberry's 4th-moment
   algorithm, O(1).  Requires ≥ 3 observations for skewness, ≥ 4 for
   kurtosis.

   :param window_size: int

   .. py:method:: update(x: float)
   .. py:method:: reset()
   .. py:method:: current_size() -> int
   .. py:method:: get_mean() -> float
   .. py:method:: get_skewness() -> float
   .. py:method:: get_kurtosis() -> float
   .. py:method:: process_mean_batch(x: numpy.ndarray) -> numpy.ndarray
   .. py:method:: process_skewness_batch(x: numpy.ndarray) -> numpy.ndarray
   .. py:method:: process_kurtosis_batch(x: numpy.ndarray) -> numpy.ndarray

   .. code-block:: python

      sm = SlidingMoments(4)
      for v in [1., 2., 3., 4.]:
          sm.update(v)
      sm.get_mean(), sm.get_skewness(), sm.get_kurtosis()
      # (2.5, 0.0, -1.2)

----

.. py:class:: SlidingCovariance(window_size)

   Rolling sample covariance and Pearson correlation — 2-D Welford
   algorithm, O(1).

   :param window_size: int

   .. py:method:: update(x: float, y: float)
   .. py:method:: get_covariance() -> float
   .. py:method:: get_correlation() -> float
   .. py:method:: get_mean_x() -> float
   .. py:method:: get_mean_y() -> float
   .. py:method:: process_covariance_batch(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray
   .. py:method:: process_correlation_batch(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

   .. code-block:: python

      sc = SlidingCovariance(3)
      for x, y in [(1,2),(2,4),(3,6)]:
          sc.update(x, y)
      sc.get_covariance(), sc.get_correlation()
      # (2.0, 1.0)
