cmake_minimum_required(VERSION 3.15)
project(robust_rolling_python LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(robust_rolling_core py_wrapper.cpp)

target_include_directories(robust_rolling_core PRIVATE ../include)

target_sources(robust_rolling_core PRIVATE
  ../src_core/MonotonicMax.cpp
  ../src_core/SlidingWelfordRing.cpp
)

target_compile_options(robust_rolling_core PRIVATE -O3 -DNDEBUG)

install(TARGETS robust_rolling_core LIBRARY DESTINATION .)