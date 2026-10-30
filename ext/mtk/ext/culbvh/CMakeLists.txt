cmake_minimum_required(VERSION 3.18)
project(culbvh LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")

find_package(TBB REQUIRED)
add_subdirectory(src)
