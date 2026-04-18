cmake_minimum_required(VERSION 3.14)
project(robust_rolling_python)

include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(robust_rolling_core py_wrapper.cpp)

target_include_directories(robust_rolling_core PRIVATE ../include)

target_sources(robust_rolling_core PRIVATE
  ../src_core/MonotonicMax.cpp
  ../src_core/SlidingWelfordRing.cpp
)

target_compile_options(robust_rolling_core PRIVATE -O3 -march=native -DNDEBUG)