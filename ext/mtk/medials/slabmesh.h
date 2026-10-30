#pragma once
#include <Eigen/Dense>
#include "common.cuh"
class SlabMesh
{
public:
    Eigen::Matrix<scalar, -1, 3> V;  // vertices (nx3)
    Eigen::Matrix<int, -1, 2> E, E0; // unique edges
    Eigen::Matrix<int, -1, 3> F;     // faces (mx3)
    Vec R;               // radius per vertex

    int nv = 0, ne = 0, nf = 0;

    SlabMesh(const std::string &filename = "");
    void load(const std::string &filename);
    void export_ma(const std::string &filename) const;
    void export_ply(const std::string &filename) const;
};
