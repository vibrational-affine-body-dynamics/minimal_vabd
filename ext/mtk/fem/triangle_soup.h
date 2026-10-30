#pragma once
#include "common.cuh"
struct TriangleSoup
{
    // pointer to actual data
    vec3 *vertices; // Assumes device pointer
    vec2i *edges;
    vec3i *indices; // 1D array, size = #faces * 3
};
