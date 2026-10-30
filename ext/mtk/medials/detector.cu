#include "medials/collision_detector.h"
#include "medials/detector.cuh"
using namespace Shay;
__global__ void kernel_slab_sphere_collision(
    MedialGeometry geo,
    SlabSphereCollisionList ss_list,
    int per_mesh_verts, int n_faces, int n_vertices)
{

    TID(tid, n_faces * n_vertices);
    int face_id = tid / n_vertices;
    int vert_id = tid % n_vertices;
    slab_sphere_collision_set(geo, ss_list, face_id, vert_id, per_mesh_verts);
}
void launch_slab_sphere_collision(
    MedialGeometry geo,
    SlabSphereCollisionList ss_list,
    int per_mesh_verts, int n_faces, int n_vertices)
{
    int launch_count = n_faces * n_vertices;
    CUDA_CALL(kernel_slab_sphere_collision, launch_count)(geo, ss_list, per_mesh_verts, n_faces, n_vertices);
}

__global__ void kernel_cone_cone_collision(
    MedialGeometry geo,
    ConeConeCollisionList cc_list,
    int per_mesh_verts, int n_edges)
{
    TID(tid, n_edges * n_edges);
    int edge_id_a = tid / n_edges;
    int edge_id_b = tid % n_edges;
    cone_cone_collision_set(geo, cc_list, edge_id_a, edge_id_b, per_mesh_verts);
}
void launch_cone_cone_collision(
    MedialGeometry geo,
    ConeConeCollisionList cc_list,
    int per_mesh_verts, int n_edges)
{
    int launch_count = n_edges * n_edges;
    CUDA_CALL(kernel_cone_cone_collision, launch_count)(geo, cc_list, per_mesh_verts, n_edges);
}

__global__ void kernel_sphere_ground_collision(
    MedialGeometry geo,
    SphereGroundCollisionList g_list,
    scalar ground_plane,
    int per_mesh_verts, int n_vertices)
{
    TID(tid, n_vertices);
    int vert_id = tid;
    sphere_ground_set(geo, g_list, ground_plane, vert_id);
}

void launch_sphere_ground_collision(
    MedialGeometry geo,
    SphereGroundCollisionList g_list,
    scalar ground_plane,
    int per_mesh_verts, int n_vertices)
{
    int launch_count = n_vertices;
    CUDA_CALL(kernel_sphere_ground_collision, launch_count)(geo, g_list, ground_plane, per_mesh_verts, n_vertices);
}
__global__ void kernel_sphere_static_collision(
    MedialGeometry geo,
    TriangleSoup triangles_soup,
    SlabSphereCollisionList collision_list,
    const int *neighbors, // assuming neighbors is a flat array
    int n_pairs,
    int2 *culled_pairs)
{
    TID(tid, n_pairs);
    sphere_static_collision(geo, triangles_soup, collision_list, neighbors, tid, culled_pairs);
}
void launch_sphere_static_collision(
    MedialGeometry geo,
    TriangleSoup triangles_soup,
    SlabSphereCollisionList collision_list,
    const int *neighbors, // assuming neighbors is a flat array
    int n_pairs,
    int2 *culled_pairs)
{
    int launch_count = n_pairs;
    CUDA_CALL(kernel_sphere_static_collision, launch_count)(geo, triangles_soup, collision_list, neighbors, n_pairs, culled_pairs);
}

__global__ void kernel_cone_static_collision(
    MedialGeometry geo,
    TriangleSoup triangles_soup,
    ConeConeCollisionList collision_list,
    int n_pairs,
    int2 *culled_pairs)
{
    TID(tid, geo.n_verts);
    int vert_id = tid;
    cone_static_collision(geo, triangles_soup, collision_list, vert_id, culled_pairs);
}

void launch_cone_static_collision(
    MedialGeometry geo,
    TriangleSoup triangles_soup,
    ConeConeCollisionList collision_list,
    int n_pairs,
    int2 *culled_pairs)
{
    int launch_count = n_pairs;

    CUDA_CALL(kernel_cone_static_collision, launch_count)(geo, triangles_soup, collision_list, n_pairs, culled_pairs);
}

__global__ void kernel_cone_aabb(
    MedialGeometry geo,
    aabb *aabbs)
{
    TID(tid, geo.n_edges);

    vec2i ex = geo.edges[tid];
    vec3 c0 = geo.vertices[ex[0]];
    vec3 c1 = geo.vertices[ex[1]];
    scalar r0 = geo.radius[ex[0]];
    scalar r1 = geo.radius[ex[1]];

    vec3 l0 = c0 - vec3(r0, r0, r0);
    vec3 l1 = c1 - vec3(r1, r1, r1);
    vec3 u0 = c0 + vec3(r0, r0, r0);
    vec3 u1 = c1 + vec3(r1, r1, r1);
    vec3 l = l0.cwiseMin(l1);
    vec3 u = u0.cwiseMax(u1);
    aabbs[tid] = {make_float3(l[0], l[1], l[2]), make_float3(u[0], u[1], u[2])};
}

__global__ void kernel_sphere_aabb(
    MedialGeometry geo,
    aabb *aabbs)
{
    TID(tid, geo.n_verts);

    scalar r = geo.radius[tid];
    vec3 c = geo.vertices[tid];
    vec3 l = c - vec3(r, r, r);
    vec3 u = c + vec3(r, r, r);
    aabbs[tid] = {make_float3(l[0], l[1], l[2]), make_float3(u[0], u[1], u[2])};
}

void launch_cone_aabb(
    MedialGeometry geo,
    aabb *aabbs)
{
    CUDA_CALL(kernel_cone_aabb, geo.n_edges)(geo, aabbs);
}

void launch_sphere_aabb(
    MedialGeometry geo,
    aabb *aabbs)
{
    CUDA_CALL(kernel_sphere_aabb, geo.n_verts)(geo, aabbs);
}

__global__ void kernel_test_culled_pairs_cone_cone(
    int n_pairs,
    int2 *pairs,
    MedialGeometry geo,
    ConeConeCollisionList collision_list,
    int per_mesh_verts)
{

    TID(tid, n_pairs);
    int i = pairs[tid].x;
    int j = pairs[tid].y;

    int ii = min(i, j);
    int jj = max(i, j);
    cone_cone_collision_set(
        geo, collision_list, ii, jj, per_mesh_verts);
}
void launch_test_culled_pairs_cone_cone(
    int n_pairs,
    int2 *pairs,
    MedialGeometry geo,
    ConeConeCollisionList collision_list,
    int per_mesh_verts)
{
    int launch_cont = n_pairs;
    CUDA_CALL(kernel_test_culled_pairs_cone_cone, launch_cont)(n_pairs, pairs, geo, collision_list, per_mesh_verts);
}

__global__ void kernel_static_triangle_aabb(
    TriangleSoup triangle_soup,
    aabb *aabbs,
    int n_triangles)
{
    TID(tid, n_triangles);
    vec3i ti = triangle_soup.indices[tid];
    vec3 t0 = triangle_soup.vertices[ti[0]];
    vec3 t1 = triangle_soup.vertices[ti[1]];
    vec3 t2 = triangle_soup.vertices[ti[2]];

    vec3 l = t0.cwiseMin(t1).cwiseMin(t2);
    vec3 u = t0.cwiseMax(t1).cwiseMax(t2);

    aabbs[tid] = {make_float3(l[0], l[1], l[2]), make_float3(u[0], u[1], u[2])};
}

__global__ void kernel_static_edge_aabb(
    TriangleSoup triangle_soup,
    aabb *aabbs,
    int n_edges)
{
    TID(tid, n_edges);
    vec2i ei = triangle_soup.edges[tid];
    vec3 e0 = triangle_soup.vertices[ei[0]];
    vec3 e1 = triangle_soup.vertices[ei[1]];

    vec3 l = e0.cwiseMin(e1);
    vec3 u = e0.cwiseMax(e1);

    aabbs[tid] = {make_float3(l[0], l[1], l[2]), make_float3(u[0], u[1], u[2])};
}

void launch_static_triangle_aabb(
    TriangleSoup triangle_soup,
    aabb *aabbs,
    int n_triangles)
{
    int launch_count = n_triangles;
    CUDA_CALL(kernel_static_triangle_aabb, launch_count)(triangle_soup, aabbs, n_triangles);
}

void launch_static_edge_aabb(
    TriangleSoup triangle_soup,
    aabb *aabbs,
    int n_static_edges)
{
    int launch_count = n_static_edges;
    CUDA_CALL(kernel_static_edge_aabb, launch_count)(triangle_soup, aabbs, n_static_edges);
}

__global__ void kernel_cc_fill_gh(
    MedialGeometry geo,
    ConeConeCollisionList cc_set,
    TripletsDev triplets)
{
    int launch_count = cc_set.cnt[0];
    TID(tid, launch_count);
    __shared__ int offset; 
    if (threadIdx.x == 0) {
        offset = triplets.cnt[0];
    }
    __syncthreads(); 
    if (tid == 0) {
        atomicAdd(triplets.cnt, launch_count * 16);
    }

    int i0 = cc_set.a[tid][0];
    int i1 = cc_set.a[tid][1];
    int i2 = cc_set.a[tid][2];
    int i3 = cc_set.a[tid][3];
    vec4i id = cc_set.a[tid];

    vec3 c0 = geo.vertices[i0];
    vec3 c1 = geo.vertices[i1];
    vec3 c2 = geo.vertices[i2];
    vec3 c3 = geo.vertices[i3];

    scalar r0 = geo.radius[i0];
    scalar r1 = geo.radius[i1];
    scalar r2 = geo.radius[i2];
    scalar r3 = geo.radius[i3];

    auto distab = compute_distance_cone_cone(c0, c1, c2, c3, r0, r1, r2, r3);
    scalar dist = distab[0];
    scalar alpha = distab[1];
    scalar beta = distab[2];

    vec3 sC1 = c0 - c1;
    vec3 sC2 = c3 - c2;
    vec3 sC3 = c1 - c3;

    vec3 v = 2.0 * (alpha * sC1 + beta * sC2 + sC3);

    scalar *g0 = triplets.g + i0 * 3;
    scalar *g1 = triplets.g + i1 * 3;
    scalar *g2 = triplets.g + i2 * 3;
    scalar *g3 = triplets.g + i3 * 3;

    vec12 diff;

    diff[0] = alpha * v.data()[0];
    diff[1] = alpha * v.data()[1];
    diff[2] = alpha * v.data()[2];

    diff[3] = (1.0 - alpha) * v.data()[0];
    diff[4] = (1.0 - alpha) * v.data()[1];
    diff[5] = (1.0 - alpha) * v.data()[2];

    diff[6] = -beta * v.data()[0];
    diff[7] = -beta * v.data()[1];
    diff[8] = -beta * v.data()[2];

    diff[9] = -(1.0 - beta) * v.data()[0];
    diff[10] = -(1.0 - beta) * v.data()[1];
    diff[11] = -(1.0 - beta) * v.data()[2];

    for (int i = 0; i < 4; i++)
    {
        vec3 diffi = diff.segment<3>(i * 3);
        for (int j = 0; j < 3; j++)
        {
            atomicAdd(triplets.g + id[i] * 3 + j, diffi[j] * 2.0 * dist);
        }
        for (int j = 0; j < 4; j++)
        {
            vec3 diffj = diff.segment<3>(j * 3);
            int idx = i * 4 + j + tid * 16 + offset;
            mat33 block_ij = 2 * diffi * diffj.transpose();
            triplets.rows[idx] = id[i];
            triplets.cols[idx] = id[j];
            triplets.blocks[idx] = block_ij;
        }
    }
}
void launch_cc_fill_gh(
    MedialGeometry geo,
    ConeConeCollisionList cc_set,
    TripletsDev triplets)
{
    int launch_count = CC_SET_SIZE;
    CUDA_CALL(kernel_cc_fill_gh, launch_count)(geo, cc_set, triplets);
}

__global__ void kernel_ss_fill_gh(
    MedialGeometry geo,
    SlabSphereCollisionList ss_set,
    TripletsDev triplets)
{
    int launch_count = ss_set.cnt[0];
    TID(tid, launch_count);
    __shared__ int offset; 
    if (threadIdx.x == 0) {
        offset = triplets.cnt[0];
    }
    __syncthreads(); 
    if (tid == 0) {
        atomicAdd(triplets.cnt, launch_count * 16);
    }
    auto id = ss_set.a[tid];
    const vec3 &b = geo.vertices[id[3]];

    vec3 c0 = geo.vertices[id[0]];
    vec3 c1 = geo.vertices[id[1]];
    vec3 c2 = geo.vertices[id[2]];

    scalar r0 = geo.radius[id[0]];
    scalar r1 = geo.radius[id[1]];
    scalar r2 = geo.radius[id[2]];
    scalar rb = geo.radius[id[3]];

    auto distab = compute_distance_slab_sphere(c0, c1, c2, b, r0, r1, r2, rb);
    scalar dist = distab[0];
    scalar alpha = distab[1];
    scalar beta = distab[2];
    vec12 diff;

    vec3 sC1 = c0 - c2;
    vec3 sC2 = c1 - c2;
    vec3 sC3 = c2 - b;

    vec3 v = 2.0 * (alpha * sC1 + beta * sC2 + sC3);

    diff[0] = (alpha * v.data()[0]);
    diff[1] = (alpha * v.data()[1]);
    diff[2] = (alpha * v.data()[2]);

    diff[3] = (beta * v.data()[0]);
    diff[4] = (beta * v.data()[1]);
    diff[5] = (beta * v.data()[2]);

    diff[6] = ((1.0 - alpha - beta) * v.data()[0]);
    diff[7] = ((1.0 - alpha - beta) * v.data()[1]);
    diff[8] = ((1.0 - alpha - beta) * v.data()[2]);

    diff[9] = (-v.data()[0]);
    diff[10] = (-v.data()[1]);
    diff[11] = (-v.data()[2]);

    for (int i = 0; i < 4; i++)
    {
        vec3 diffi = diff.segment<3>(i * 3);
        for (int j = 0; j < 3; j++)
        {
            atomicAdd(triplets.g + id[i] * 3 + j, diffi[j] * 2.0 * dist);
        }
        for (int j = 0; j < 4; j++)
        {
            vec3 diffj = diff.segment<3>(j * 3);
            int idx = i * 4 + j + tid * 16 + offset;
            mat33 block_ij = 2 * diffi * diffj.transpose();
            triplets.rows[idx] = id[i];
            triplets.cols[idx] = id[j];
            triplets.blocks[idx] = block_ij;
        }
    }
}
void launch_ss_fill_gh(
    MedialGeometry geo,
    SlabSphereCollisionList ss_set,
    TripletsDev triplets)
{
    int launch_count = SS_SET_SIZE;
    CUDA_CALL(kernel_ss_fill_gh, launch_count)(geo, ss_set, triplets);
}


__global__ void kernel_g_fill_gh(
    
    MedialGeometry geo, 
    SphereGroundCollisionList g_set,
    TripletsDev triplets
){
    int launch_count = g_set.cnt[0];
    TID(tid, launch_count);
    __shared__ int offset; 
    if (threadIdx.x == 0) {
        offset = triplets.cnt[0];
    }
    __syncthreads(); 
    if (tid == 0) {
        atomicAdd(triplets.cnt, launch_count);
    }
    scalar h = g_set.hr[tid][0];
    scalar r = g_set.hr[tid][1];    
    scalar h11 = 12 * (h - r) * (h - r);
    mat33 hh = mat33::Zero(); 
    hh(1, 1) = h11;
    scalar gg1 = (h - r) * h11 / 3; 
    vec3 gg(0.0, gg1, 0.0);

    int id = g_set.a[tid];
    atomicAdd(triplets.g + id * 3 + 1, gg1);
    int idx = offset + tid; 

    triplets.rows[idx] = id; 
    triplets.cols[idx] = id;
    triplets.blocks[idx] = hh;
}
void launch_g_fill_gh(
    MedialGeometry geo, 
    SphereGroundCollisionList g_set,
    TripletsDev triplets
) {
    int launch_count = G_SET_SIZE;
    CUDA_CALL(kernel_g_fill_gh, launch_count)(geo, g_set, triplets);
}