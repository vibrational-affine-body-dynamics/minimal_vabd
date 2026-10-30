#pragma once


// #include <fmt/format.h>
// #include <glm/glm.hpp>
#include "helper_cuda.h"
// #include "helper_math.h"
#ifndef __CUDACC__

inline int max(int a, int b)
{
    return a > b ? a : b;
}

inline int min(int a, int b)
{
    return a < b ? a : b;
}
#endif
#include <cuda_runtime.h> 
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#define __CUDARUN__
// #include <thrust/device_ptr.h>
// #include <thrust/transform.h>
// #include <thrust/sort.h>
#include <Eigen/Core>

// #define CONST(type)				const type const
#define TID(id, maxID) 	uint id = blockIdx.x * blockDim.x + threadIdx.x; if (id >= maxID) return
#define TID_NO_RETURN(id, maxID) 	uint id = blockIdx.x * blockDim.x + threadIdx.x

#ifdef __CUDACC__ 
#define CUDA_CALL(func, totalThreads)  \
	if (totalThreads == 0) return; \
	uint func ## _numBlocks, func ## _numThreads; \
	ComputeGridSize(totalThreads, func ## _numBlocks, func ## _numThreads); \
	func <<<func ## _numBlocks, func ## _numThreads >>>
#define CUDA_CALL_S(func, totalThreads, stream)  \
	if (totalThreads == 0) return; \
	uint func ## _numBlocks, func ## _numThreads; \
	ComputeGridSize(totalThreads, func ## _numBlocks, func ## _numThreads); \
	func <<<func ## _numBlocks, func ## _numThreads, 0,stream>>>
#define CUDA_CALL_V(func, ...) \
	func <<<__VA_ARGS__>>>
// #else
// #define CUDA_CALL(func, totalThreads)
// #define CUDA_CALL_S(func, totalThreads, stream) 
// #define CUDA_CALL_V(func, ...)
#endif

#define __func__ __host__ __device__ inline

typedef unsigned int uint;
typedef unsigned short ushort;
#define __DOUBLE__SOLVER__
#ifdef __DOUBLE__SOLVER__
#define SCALAR_TYPE CUDA_R_64F
using scalar = double;
#else 
#define SCALAR_TYPE CUDA_R_32F
using scalar = float;
#endif
using vec2i = Eigen::Vector2i;
using vec3i = Eigen::Vector3i;
using vec4i = Eigen::Vector4i;

using vec2 = Eigen::Vector<scalar, 2>;
using vec3 = Eigen::Vector<scalar, 3>;
using vec4 = Eigen::Vector<scalar, 4>; 
using vec12 = Eigen::Vector<scalar, 12>;

using mat22 = Eigen::Matrix<scalar, 2, 2>;
using mat33 = Eigen::Matrix<scalar, 3, 3>;
using mat34 = Eigen::Matrix<scalar, 3, 4>;
using mat44 = Eigen::Matrix<scalar, 4, 4>;
using mat12 = Eigen::Matrix<scalar, 12, 12>;
using q4 = mat34;
using Mat = Eigen::Matrix<scalar, -1, -1>;
using Vec = Eigen::Vector<scalar, -1>; 
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define MIN_VALUE 5e-14
#define IS_double_ZERO(d) (abs(d) < MIN_VALUE)
#define Check_double_ZERO(d) (abs(d) > MIN_VALUE ? d : 0)

namespace Shay
{
	typedef unsigned int uint;

	const uint BLOCK_SIZE = 256;

	// __device__ inline float length2(glm::vec3 vec)
	// {
	// 	return glm::dot(vec, vec);
	// }

	inline void ComputeGridSize(const uint& n, uint& numBlocks, uint& numThreads)
	{
		if (n == 0)
		{
			//fmt::print("Error(Solver): numParticles is 0\n");
			numBlocks = 0;
			numThreads = 0;
			return;
		}
		numThreads = min(n, BLOCK_SIZE);
		numBlocks = (n % numThreads != 0) ? (n / numThreads + 1) : (n / numThreads);
	}

	template<class T>
	inline T* VtAllocBuffer(size_t elementCount)
	{
		T* devPtr = nullptr;
		#ifdef __CUDARUN__
		cudaMalloc((void **)&devPtr, elementCount * sizeof(T));
		#else
		// checkCudaErrors(cudaMallocManaged((void**)&devPtr, elementCount * sizeof(T)));
		devPtr = (T*)malloc(elementCount * sizeof(T));
		#endif
		cudaDeviceSynchronize(); // this is necessary, otherwise realloc can cause crash
		return devPtr;
	}

	inline void VtFreeBuffer(void* buffer)
	{
		#ifdef __CUDARUN__
		cudaDeviceSynchronize();
		cudaFree(buffer);
		#else

		free(buffer);
		#endif
	}

}