cmake_minimum_required(VERSION 3.28)
project(CPSC424)

set(CMAKE_CXX_STANDARD 17)

add_executable(CPSC424 homework1/matrixmult.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -mavx2 -mavx512f")