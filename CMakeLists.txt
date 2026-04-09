cmake_minimum_required(VERSION 3.14)
project(robust_rolling VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O3")

add_library(rolling_core src/SlidingMedian.cxx)
target_include_directories(rolling_core PUBLIC include)

add_executable(rolling_app src/main.cxx)
target_link_libraries(rolling_app PRIVATE rolling_core)