cmake_minimum_required(VERSION 3.28)
project(CPSC424)

set(CMAKE_CXX_STANDARD 17)

add_executable(homework1 homework1/matrixmult.cpp)
add_executable(homework1_1naive homework1/1_matrixmult_naive.cpp)
add_executable(homework1_1vectorization homework1/2_matrixmult_vectorization.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -mavx2 -mavx512f")