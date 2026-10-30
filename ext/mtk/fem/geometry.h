// fem/geometry.h
#pragma once
#include <vector>
#include <string>
#include "buffer.h"

namespace fem
{

    struct GeometryBase
    {
        Shay::VtBuffer<vec3> xcs;
        Shay::VtBuffer<vec4i> T;
        Shay::VtBuffer<vec3i> indices;
        int n_nodes;
        int n_tets;
    };

    struct FEMMesh
    {
        int n_nodes = 0;
        int n_tets = 0;
        // Pointer to vertex positions
        vec3 *xcs = nullptr;
        // Tetrahedral indices (n_tets x 4)
        vec4i *T = nullptr;
    };

    struct TOBJLoader : public GeometryBase
    {
        TOBJLoader(const std::string &filename);
    };

    struct TOBJComplex : public GeometryBase
    {
        TOBJComplex(const std::vector<std::string> &meshes_filename,
                    const std::vector<mat44> &transforms);
        Eigen::MatrixXi F;
    };

}; // namespace fem