cmake_minimum_required(VERSION 2.8)
project(SalMetric. C CXX)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
set(SalMetric_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(${SalMetric_INCLUDE_DIR})
message(STATUS ${SalMetric_INCLUDE_DIR})

add_subdirectory(src)

