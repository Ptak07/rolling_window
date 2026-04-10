cmake_minimum_required(VERSION 3.14)
project(robust_rolling VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O3")

add_library(rolling_core 
    src/MultisetMedian.cxx
    src/MonotonicMax.cxx
    src/SlidingWelford.cxx
    src/SlidingWelfordRing.cxx
)
target_include_directories(rolling_core PUBLIC include)

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
add_executable(test_core tests/test_core.cxx) 
target_link_libraries(test_core PRIVATE rolling_core gtest gtest_main) 

include(GoogleTest)
gtest_discover_tests(test_core)

add_executable(bench_core benchmarks/bench_core.cxx)
target_link_libraries(bench_core PRIVATE rolling_core benchmark::benchmark benchmark::benchmark_main)