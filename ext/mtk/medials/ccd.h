#pragma once

#include "common.cuh"
#include <Eigen/Core>
#include <utility>

namespace MIPC
{
// Continuous collision detection for linearly moving medial primitives.
// Returns (earliest normalized time of impact, swept-AABB candidate count).
std::pair<scalar, int> medial_ccd(
    const Eigen::Matrix<scalar, -1, 3> &positions_start,
    const Eigen::Matrix<scalar, -1, 3> &positions_end,
    const Vec &radii,
    const Eigen::Matrix<int, -1, 2> &edges,
    const Eigen::Matrix<int, -1, 3> &faces,
    const Eigen::VectorXi &body_ids,
    const Eigen::Matrix<scalar, -1, 3> &static_positions,
    const Vec &static_radii,
    const Eigen::Matrix<int, -1, 2> &static_edges,
    const Eigen::Matrix<int, -1, 3> &static_faces,
    scalar tolerance = 1.0e-8);
}
