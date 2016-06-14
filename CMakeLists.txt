cmake_minimum_required(VERSION 3.5)
project(ml_linear_regression)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cc)
add_executable(ml_linear_regression ${SOURCE_FILES})