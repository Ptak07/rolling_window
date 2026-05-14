cmake_minimum_required(VERSION 3.14)
project(robust_rolling VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O3")

add_library(rolling_core 
    src_core/MultisetMedian.cpp
    src_core/MonotonicMax.cpp
    src_core/MonotonicMin.cpp
    src_core/SlidingWelford.cpp
    src_core/SlidingWelfordRing.cpp
    src_core/SlidingMoments.cpp
)
target_include_directories(rolling_core PUBLIC include)

# R integration for wrapper sources (headers, compiler flags and linker flags).
find_program(R_EXECUTABLE NAMES R REQUIRED)

execute_process(
    COMMAND ${R_EXECUTABLE} CMD config --cppflags
    OUTPUT_VARIABLE R_CPPFLAGS_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${R_EXECUTABLE} CMD config --ldflags
    OUTPUT_VARIABLE R_LDFLAGS_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

separate_arguments(R_CPPFLAGS NATIVE_COMMAND "${R_CPPFLAGS_RAW}")
separate_arguments(R_LDFLAGS NATIVE_COMMAND "${R_LDFLAGS_RAW}")

add_library(rolling_r_wrapper SHARED
    src/r_wrapper.cpp
)
target_include_directories(rolling_r_wrapper PRIVATE include)
target_link_libraries(rolling_r_wrapper PRIVATE rolling_core)
target_compile_options(rolling_r_wrapper PRIVATE ${R_CPPFLAGS})
target_link_options(rolling_r_wrapper PRIVATE ${R_LDFLAGS})

include(FetchContent)

# google tests 
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
)

#google benchmark
FetchContent_Declare(
    googlebenchmark 
    GIT_REPOSITORY https://github.com/google/benchmark.git 
    GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE) 
FetchContent_MakeAvailable(googlebenchmark)

# Required for Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing() 
add_executable(test_core
    tests/test_multiset_median.cxx
    tests/test_monotonic_max.cxx
    tests/test_monotonic_min.cxx
    tests/test_sliding_welford.cxx
    tests/test_sliding_welford_ring.cxx
    tests/test_sliding_moments.cxx
)
target_link_libraries(test_core PRIVATE rolling_core gtest gtest_main) 

include(GoogleTest)
gtest_discover_tests(test_core)

add_executable(bench_core benchmarks/bench_core.cxx)
target_link_libraries(bench_core PRIVATE rolling_core benchmark::benchmark benchmark::benchmark_main)