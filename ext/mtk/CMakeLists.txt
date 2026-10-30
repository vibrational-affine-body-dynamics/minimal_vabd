cmake_minimum_required(VERSION 3.18)
project(mtk LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(cudss CONFIG REQUIRED)

add_subdirectory(ext/culbvh)

file(GLOB MTK_SOURCES CONFIGURE_DEPENDS medials/*.cpp medials/*.cu)
pybind11_add_module(mtk ${MTK_SOURCES} python/bindings.cpp)
target_include_directories(
    mtk PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${cudss_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/culbvh/src
)
target_link_libraries(
    mtk PRIVATE
    Eigen3::Eigen
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
    cudss
    culbvh
)
set_target_properties(
    mtk PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)
