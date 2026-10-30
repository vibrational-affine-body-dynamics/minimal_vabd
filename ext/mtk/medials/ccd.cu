#include "medials/ccd.h"

#include "buffer.h"
#include "medials/detector.cuh"
#include "utils/buffer2eigen.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace Shay;

namespace MIPC
{
namespace
{
__device__ inline scalar clamp01(scalar value)
{
    return max(scalar(0), min(scalar(1), value));
}

__device__ inline vec3 lerp_position(const vec3 &start, const vec3 &velocity, scalar time)
{
    return start + time * velocity;
}

__device__ inline scalar vector_norm(const vec3 &value)
{
    return sqrt(value.dot(value));
}

__device__ inline scalar cone_relative_speed_bound(
    const vec2i &edge_a, const vec2i &edge_b,
    const vec3 *velocity_a, const vec3 *velocity_b)
{
    scalar bound = scalar(0);
    for (int endpoint_a = 0; endpoint_a < 2; ++endpoint_a)
    {
        for (int endpoint_b = 0; endpoint_b < 2; ++endpoint_b)
        {
            bound = max(
                bound,
                vector_norm(velocity_a[edge_a[endpoint_a]] -
                            velocity_b[edge_b[endpoint_b]]));
        }
    }
    return bound;
}

__device__ inline scalar slab_sphere_relative_speed_bound(
    const vec3i &face, int sphere,
    const vec3 *velocity_face, const vec3 *velocity_sphere)
{
    scalar bound = scalar(0);
    for (int vertex = 0; vertex < 3; ++vertex)
    {
        bound = max(
            bound,
            vector_norm(velocity_face[face[vertex]] -
                        velocity_sphere[sphere]));
    }
    return bound;
}

__device__ inline void atomic_min_toi(scalar *address, scalar value)
{
#ifdef __DOUBLE__SOLVER__
    atomicMin(reinterpret_cast<unsigned long long *>(address),
              static_cast<unsigned long long>(__double_as_longlong(value)));
#else
    atomicMin(reinterpret_cast<unsigned int *>(address), __float_as_uint(value));
#endif
}

__device__ inline bool intervals_overlap(scalar amin, scalar amax, scalar bmin, scalar bmax)
{
    return amin <= bmax && bmin <= amax;
}

__device__ inline void swept_edge_bounds(
    const vec2i &edge, const vec3 *start, const vec3 *end, const scalar *radii,
    vec3 &lower, vec3 &upper)
{
    const int ids[2] = {edge[0], edge[1]};
    lower = vec3(1.0e30, 1.0e30, 1.0e30);
    upper = vec3(-1.0e30, -1.0e30, -1.0e30);
    for (int endpoint = 0; endpoint < 2; ++endpoint)
    {
        int id = ids[endpoint];
        scalar radius = radii[id];
        for (int axis = 0; axis < 3; ++axis)
        {
            lower[axis] = min(lower[axis], min(start[id][axis], end[id][axis]) - radius);
            upper[axis] = max(upper[axis], max(start[id][axis], end[id][axis]) + radius);
        }
    }
}

__device__ inline void swept_slab_bounds(
    const vec3i &face, const vec3 *start, const vec3 *end, const scalar *radii,
    vec3 &lower, vec3 &upper)
{
    lower = vec3(1.0e30, 1.0e30, 1.0e30);
    upper = vec3(-1.0e30, -1.0e30, -1.0e30);
    for (int vertex = 0; vertex < 3; ++vertex)
    {
        int id = face[vertex];
        scalar radius = radii[id];
        for (int axis = 0; axis < 3; ++axis)
        {
            lower[axis] = min(lower[axis], min(start[id][axis], end[id][axis]) - radius);
            upper[axis] = max(upper[axis], max(start[id][axis], end[id][axis]) + radius);
        }
    }
}

__device__ inline void swept_sphere_bounds(
    int id, const vec3 *start, const vec3 *end, const scalar *radii,
    vec3 &lower, vec3 &upper)
{
    scalar radius = radii[id];
    for (int axis = 0; axis < 3; ++axis)
    {
        lower[axis] = min(start[id][axis], end[id][axis]) - radius;
        upper[axis] = max(start[id][axis], end[id][axis]) + radius;
    }
}

__device__ inline bool boxes_overlap(
    const vec3 &lower_a, const vec3 &upper_a,
    const vec3 &lower_b, const vec3 &upper_b)
{
    return intervals_overlap(lower_a[0], upper_a[0], lower_b[0], upper_b[0]) &&
           intervals_overlap(lower_a[1], upper_a[1], lower_b[1], upper_b[1]) &&
           intervals_overlap(lower_a[2], upper_a[2], lower_b[2], upper_b[2]);
}

__device__ inline scalar cone_gap(
    const vec2i &edge_a, const vec2i &edge_b,
    const vec3 *start_a, const vec3 *velocity_a, const scalar *radii_a,
    const vec3 *start_b, const vec3 *velocity_b, const scalar *radii_b,
    scalar time)
{
    vec3 a0 = lerp_position(start_a[edge_a[0]], velocity_a[edge_a[0]], time);
    vec3 a1 = lerp_position(start_a[edge_a[1]], velocity_a[edge_a[1]], time);
    vec3 b0 = lerp_position(start_b[edge_b[0]], velocity_b[edge_b[0]], time);
    vec3 b1 = lerp_position(start_b[edge_b[1]], velocity_b[edge_b[1]], time);
    // compute_distance_cone_cone minimizes the paper's quadratic medial
    // distance f = ||c1-c2||^2 - (r1+r2)^2.  Its minimizer supplies the
    // corresponding physical surface gap used for conservative advancement;
    // both definitions have exactly the same contact (zero) set.
    vec3 distance = compute_distance_cone_cone(
        a0, a1, b0, b1,
        radii_a[edge_a[0]], radii_a[edge_a[1]],
        radii_b[edge_b[0]], radii_b[edge_b[1]]);
    scalar alpha = clamp01(distance[1]);
    scalar beta = clamp01(distance[2]);
    vec3 point_a = a1 + alpha * (a0 - a1);
    vec3 point_b = b1 + beta * (b0 - b1);
    scalar radius_a = radii_a[edge_a[1]] + alpha *
        (radii_a[edge_a[0]] - radii_a[edge_a[1]]);
    scalar radius_b = radii_b[edge_b[1]] + beta *
        (radii_b[edge_b[0]] - radii_b[edge_b[1]]);
    return vector_norm(point_a - point_b) - radius_a - radius_b;
}

__device__ inline scalar cone_toi(
    const vec2i &edge_a, const vec2i &edge_b,
    const vec3 *start_a, const vec3 *velocity_a, const scalar *radii_a,
    const vec3 *start_b, const vec3 *velocity_b, const scalar *radii_b,
    scalar tolerance)
{
    // The gap is invariant under common translation.  Bounding it with the
    // sum of absolute endpoint speeds can therefore be arbitrarily loose: two
    // separated primitives moving together at high speed exhausted the
    // iteration budget and the old code returned a false early impact.  Any
    // pair of points on the two segments has a relative velocity in the convex
    // hull of these four endpoint differences, so their maximum norm is a
    // conservative and translation-invariant Lipschitz bound.
    scalar speed_bound = cone_relative_speed_bound(
        edge_a, edge_b, velocity_a, velocity_b);
    scalar time = 0.0;
    scalar gap = cone_gap(edge_a, edge_b, start_a, velocity_a, radii_a,
                          start_b, velocity_b, radii_b, time);
    // IPC requires a strictly positive starting gap.  Once the CCD
    // reintegration uses a positive-gap barrier, an active pair must be
    // reported at t=0 rather than delegated to the penetration-tolerant DCD
    // energy.
    if (!isfinite(gap))
        return scalar(1);
    if (gap <= tolerance)
        return scalar(0);
    for (int iteration = 0; iteration < 64; ++iteration)
    {
        if (gap <= tolerance)
            return time;
        if (speed_bound <= tolerance)
            return scalar(1);
        scalar advance = scalar(0.9) * (gap - tolerance) / speed_bound;
        advance = max(advance, scalar(1.0e-12));
        time += advance;
        if (time > scalar(1))
            return scalar(1);
        gap = cone_gap(edge_a, edge_b, start_a, velocity_a, radii_a,
                       start_b, velocity_b, radii_b, time);
        if (!isfinite(gap))
            return scalar(1);
    }
    // Reaching the iteration limit only proves that [0, time] is safe; it
    // does not prove an impact at `time`.  Returning it as a TOI manufactured
    // a collision in free space.  Treat an unresolved candidate as a miss;
    // genuine impacts above are returned only after the gap test succeeds.
    return scalar(1);
}

__device__ inline scalar slab_sphere_gap(
    const vec3i &face, int sphere,
    const vec3 *start_face, const vec3 *velocity_face, const scalar *radii_face,
    const vec3 *start_sphere, const vec3 *velocity_sphere, const scalar *radii_sphere,
    scalar time)
{
    vec3 a0 = lerp_position(start_face[face[0]], velocity_face[face[0]], time);
    vec3 a1 = lerp_position(start_face[face[1]], velocity_face[face[1]], time);
    vec3 a2 = lerp_position(start_face[face[2]], velocity_face[face[2]], time);
    vec3 b = lerp_position(start_sphere[sphere], velocity_sphere[sphere], time);
    vec3 distance = compute_distance_slab_sphere(
        a0, a1, a2, b,
        radii_face[face[0]], radii_face[face[1]], radii_face[face[2]],
        radii_sphere[sphere]);
    scalar alpha = clamp01(distance[1]);
    scalar beta = clamp01(distance[2]);
    if (alpha + beta > scalar(1))
    {
        scalar sum = alpha + beta;
        alpha /= sum;
        beta /= sum;
    }
    vec3 point = a2 + alpha * (a0 - a2) + beta * (a1 - a2);
    scalar radius = radii_face[face[2]] +
        alpha * (radii_face[face[0]] - radii_face[face[2]]) +
        beta * (radii_face[face[1]] - radii_face[face[2]]);
    return vector_norm(point - b) - radius - radii_sphere[sphere];
}

__device__ inline scalar slab_sphere_toi(
    const vec3i &face, int sphere,
    const vec3 *start_face, const vec3 *velocity_face, const scalar *radii_face,
    const vec3 *start_sphere, const vec3 *velocity_sphere, const scalar *radii_sphere,
    scalar tolerance)
{
    scalar speed_bound = slab_sphere_relative_speed_bound(
        face, sphere, velocity_face, velocity_sphere);
    scalar time = 0.0;
    scalar gap = slab_sphere_gap(face, sphere, start_face, velocity_face,
                                 radii_face, start_sphere, velocity_sphere,
                                 radii_sphere, time);
    if (!isfinite(gap))
        return scalar(1);
    if (gap <= tolerance)
        return scalar(0);
    for (int iteration = 0; iteration < 64; ++iteration)
    {
        if (gap <= tolerance)
            return time;
        if (speed_bound <= tolerance)
            return scalar(1);
        scalar advance = scalar(0.9) * (gap - tolerance) / speed_bound;
        advance = max(advance, scalar(1.0e-12));
        time += advance;
        if (time > scalar(1))
            return scalar(1);
        gap = slab_sphere_gap(face, sphere, start_face, velocity_face,
                              radii_face, start_sphere, velocity_sphere,
                              radii_sphere, time);
        if (!isfinite(gap))
            return scalar(1);
    }
    // Do not turn conservative-advancement non-convergence into a contact.
    return scalar(1);
}

__global__ void dynamic_cone_ccd_kernel(
    const vec3 *start, const vec3 *end, const vec3 *velocity,
    const scalar *radii, const vec2i *edges, const int *body_ids,
    int edge_count, scalar tolerance, scalar *toi, int *candidate_count)
{
    TID(tid, edge_count * edge_count);
    int first = tid / edge_count;
    int second = tid % edge_count;
    if (first >= second)
        return;
    vec2i edge_a = edges[first];
    vec2i edge_b = edges[second];
    if (body_ids[edge_a[0]] == body_ids[edge_b[0]] ||
        radii[edge_a[0]] < 0 || radii[edge_a[1]] < 0 ||
        radii[edge_b[0]] < 0 || radii[edge_b[1]] < 0)
        return;
    vec3 lower_a, upper_a, lower_b, upper_b;
    swept_edge_bounds(edge_a, start, end, radii, lower_a, upper_a);
    swept_edge_bounds(edge_b, start, end, radii, lower_b, upper_b);
    if (!boxes_overlap(lower_a, upper_a, lower_b, upper_b))
        return;
    atomicAdd(candidate_count, 1);
    scalar impact = cone_toi(edge_a, edge_b, start, velocity, radii,
                             start, velocity, radii, tolerance);
    if (impact < scalar(1))
        atomic_min_toi(toi, impact);
}

__global__ void static_cone_ccd_kernel(
    const vec3 *start, const vec3 *end, const vec3 *velocity,
    const scalar *radii, const vec2i *edges, int edge_count,
    const vec3 *static_positions, const vec3 *static_velocity,
    const scalar *static_radii, const vec2i *static_edges, int static_edge_count,
    scalar tolerance, scalar *toi, int *candidate_count)
{
    TID(tid, edge_count * static_edge_count);
    int moving_id = tid / static_edge_count;
    int static_id = tid % static_edge_count;
    vec2i moving_edge = edges[moving_id];
    vec2i fixed_edge = static_edges[static_id];
    if (radii[moving_edge[0]] < 0 || radii[moving_edge[1]] < 0 ||
        static_radii[fixed_edge[0]] < 0 || static_radii[fixed_edge[1]] < 0)
        return;
    vec3 lower_a, upper_a, lower_b, upper_b;
    swept_edge_bounds(moving_edge, start, end, radii, lower_a, upper_a);
    swept_edge_bounds(fixed_edge, static_positions, static_positions,
                      static_radii, lower_b, upper_b);
    if (!boxes_overlap(lower_a, upper_a, lower_b, upper_b))
        return;
    atomicAdd(candidate_count, 1);
    scalar impact = cone_toi(moving_edge, fixed_edge, start, velocity, radii,
                             static_positions, static_velocity, static_radii,
                             tolerance);
    if (impact < scalar(1))
        atomic_min_toi(toi, impact);
}

__global__ void dynamic_slab_sphere_ccd_kernel(
    const vec3 *start, const vec3 *end, const vec3 *velocity,
    const scalar *radii, const vec3i *faces, const int *body_ids,
    int face_count, int vertex_count, scalar tolerance,
    scalar *toi, int *candidate_count)
{
    TID(tid, face_count * vertex_count);
    int face_id = tid / vertex_count;
    int sphere = tid % vertex_count;
    vec3i face = faces[face_id];
    if (body_ids[face[0]] == body_ids[sphere] || radii[sphere] < 0 ||
        radii[face[0]] < 0 || radii[face[1]] < 0 || radii[face[2]] < 0)
        return;
    vec3 lower_a, upper_a, lower_b, upper_b;
    swept_slab_bounds(face, start, end, radii, lower_a, upper_a);
    swept_sphere_bounds(sphere, start, end, radii, lower_b, upper_b);
    if (!boxes_overlap(lower_a, upper_a, lower_b, upper_b))
        return;
    atomicAdd(candidate_count, 1);
    scalar impact = slab_sphere_toi(face, sphere, start, velocity, radii,
                                    start, velocity, radii, tolerance);
    if (impact < scalar(1))
        atomic_min_toi(toi, impact);
}

__global__ void static_slab_sphere_ccd_kernel(
    const vec3 *start, const vec3 *end, const vec3 *velocity,
    const scalar *radii, const vec3i *faces, int face_count, int vertex_count,
    const vec3 *static_positions, const vec3 *static_velocity,
    const scalar *static_radii, const vec3i *static_faces,
    int static_face_count, int static_vertex_count, scalar tolerance,
    scalar *toi, int *candidate_count)
{
    int moving_face_pairs = face_count * static_vertex_count;
    int fixed_face_pairs = static_face_count * vertex_count;
    TID(tid, moving_face_pairs + fixed_face_pairs);
    vec3 lower_a, upper_a, lower_b, upper_b;
    scalar impact = scalar(1);
    if (tid < moving_face_pairs)
    {
        vec3i face = faces[tid / static_vertex_count];
        int sphere = tid % static_vertex_count;
        swept_slab_bounds(face, start, end, radii, lower_a, upper_a);
        swept_sphere_bounds(sphere, static_positions, static_positions,
                            static_radii, lower_b, upper_b);
        if (!boxes_overlap(lower_a, upper_a, lower_b, upper_b))
            return;
        atomicAdd(candidate_count, 1);
        impact = slab_sphere_toi(face, sphere, start, velocity, radii,
                                 static_positions, static_velocity, static_radii,
                                 tolerance);
    }
    else
    {
        int pair = tid - moving_face_pairs;
        vec3i face = static_faces[pair / vertex_count];
        int sphere = pair % vertex_count;
        swept_slab_bounds(face, static_positions, static_positions,
                          static_radii, lower_a, upper_a);
        swept_sphere_bounds(sphere, start, end, radii, lower_b, upper_b);
        if (!boxes_overlap(lower_a, upper_a, lower_b, upper_b))
            return;
        atomicAdd(candidate_count, 1);
        impact = slab_sphere_toi(face, sphere, static_positions, static_velocity,
                                 static_radii, start, velocity, radii, tolerance);
    }
    if (impact < scalar(1))
        atomic_min_toi(toi, impact);
}
}

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
    scalar tolerance)
{
    if (positions_start.rows() != positions_end.rows() ||
        positions_start.rows() != radii.rows() ||
        positions_start.rows() != body_ids.rows())
        throw std::invalid_argument("dynamic medial CCD array sizes do not match");
    if (static_positions.rows() != static_radii.rows())
        throw std::invalid_argument("static medial CCD array sizes do not match");

    wp_array<vec3> start, end, velocity;
    wp_array<scalar> radius;
    wp_array<vec2i> edge;
    wp_array<vec3i> face;
    wp_array<int> bodies;
    WpArray(positions_start, start);
    WpArray(positions_end, end);
    Eigen::Matrix<scalar, -1, 3> velocity_matrix = positions_end - positions_start;
    WpArray(velocity_matrix, velocity);
    radius.copy(std::vector<scalar>(radii.data(), radii.data() + radii.size()));
    WpArray(edges, edge);
    WpArray(faces, face);
    bodies.copy(std::vector<int>(body_ids.data(), body_ids.data() + body_ids.size()));

    wp_array<vec3> fixed, fixed_velocity;
    wp_array<scalar> fixed_radius;
    wp_array<vec2i> fixed_edge;
    wp_array<vec3i> fixed_face;
    WpArray(static_positions, fixed);
    fixed_velocity.resize(static_positions.rows());
    fixed_velocity.zero_();
    fixed_radius.copy(std::vector<scalar>(
        static_radii.data(), static_radii.data() + static_radii.size()));
    WpArray(static_edges, fixed_edge);
    WpArray(static_faces, fixed_face);

    wp_array<scalar> device_toi(1);
    wp_array<int> candidates(1);
    scalar initial_toi = scalar(1);
    cudaMemcpy(device_toi.data(), &initial_toi, sizeof(scalar), cudaMemcpyHostToDevice);
    candidates.zero_();

    int edge_count = static_cast<int>(edges.rows());
    int face_count = static_cast<int>(faces.rows());
    int vertex_count = static_cast<int>(positions_start.rows());
    if (edge_count > 0)
    {
        uint blocks, threads;
        ComputeGridSize(edge_count * edge_count, blocks, threads);
        dynamic_cone_ccd_kernel<<<blocks, threads>>>(
            start, end, velocity, radius, edge, bodies, edge_count,
            tolerance, device_toi, candidates);
    }
    if (face_count > 0)
    {
        uint blocks, threads;
        ComputeGridSize(face_count * vertex_count, blocks, threads);
        dynamic_slab_sphere_ccd_kernel<<<blocks, threads>>>(
            start, end, velocity, radius, face, bodies, face_count, vertex_count,
            tolerance, device_toi, candidates);
    }

    int static_edge_count = static_cast<int>(static_edges.rows());
    int static_face_count = static_cast<int>(static_faces.rows());
    int static_vertex_count = static_cast<int>(static_positions.rows());
    if (edge_count > 0 && static_edge_count > 0)
    {
        uint blocks, threads;
        ComputeGridSize(edge_count * static_edge_count, blocks, threads);
        static_cone_ccd_kernel<<<blocks, threads>>>(
            start, end, velocity, radius, edge, edge_count,
            fixed, fixed_velocity, fixed_radius, fixed_edge, static_edge_count,
            tolerance, device_toi, candidates);
    }
    int static_slab_pairs = face_count * static_vertex_count +
                            static_face_count * vertex_count;
    if (static_slab_pairs > 0)
    {
        uint blocks, threads;
        ComputeGridSize(static_slab_pairs, blocks, threads);
        static_slab_sphere_ccd_kernel<<<blocks, threads>>>(
            start, end, velocity, radius, face, face_count, vertex_count,
            fixed, fixed_velocity, fixed_radius, fixed_face,
            static_face_count, static_vertex_count, tolerance,
            device_toi, candidates);
    }

    cudaDeviceSynchronize();
    scalar result_toi;
    int result_candidates;
    cudaMemcpy(&result_toi, device_toi.data(), sizeof(scalar), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result_candidates, candidates.data(), sizeof(int), cudaMemcpyDeviceToHost);
    return {std::max(scalar(0), std::min(scalar(1), result_toi)), result_candidates};
}
}
