cmake_minimum_required(VERSION 3.5)
project(ml_linear_regression)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cc ml_base/utils.h ml_base/utils.cpp ml_base/linear_regression.cpp ml_base/linear_regression.h)

add_executable(ml_linear_regression ${SOURCE_FILES})