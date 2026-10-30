cmake_minimum_required(VERSION 3.12)
project(sqem LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

pybind11_add_module(sqem bindings.cpp)
target_link_libraries(sqem PRIVATE Eigen3::Eigen)
