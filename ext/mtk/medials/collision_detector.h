#pragma once
#include "common.cuh"
#include "buffer.h"
#include "medials/constraints.h"
#include <vector>
#include <Eigen/Core>
#include <string>
#include <set>
#include "utils/linalg.cuh"
#include "fem/triangle_soup.h"
#include "fem/geometry.h"
#include <Eigen/Sparse>
#include "culbvh.cuh"
using aabb = culbvh::Bound<float>;
static const int
    CC_SET_SIZE = 4096,
    SS_SET_SIZE = 4096,
    G_SET_SIZE = 1024,
    CC_BUF_SIZE = 65536 * 8,
    ST_BUF_SIZE = 65536 * 8;
struct MedialGeometry
{
    vec3 *vertices;
    scalar *radius;
    vec2i *edges;
    vec3i *faces;
    int *connectivity;
    int n_verts, n_edges, n_faces;
};

__func__ int atomic_add(int *address, int value)
{
#if defined(__CUDA_ARCH__)
    return atomicAdd(address, value);
#else
    int old = *address;
    *address += value;
    return old;
#endif
}
__func__ scalar atomic_add(scalar *address, scalar value)
{
#if defined(__CUDA_ARCH__)
    return atomicAdd(address, value);
#else
    int old = *address;
    *address += value;
    return old;
#endif
}

struct ConeConeCollisionList
{
    vec4i *a;
    scalar *dist;
    scalar *E;
    int *cnt;
    __func__ void append(const vec4i &element, scalar dist)
    {
        int id = atomic_add(cnt, 1);
        if (id < CC_SET_SIZE)
        {
            a[id] = element;
            this->dist[id] = dist;
        }
        atomic_add(E, dist * dist);
    }
};

struct SlabSphereCollisionList
{
    vec4i *a;
    scalar *E;
    int *cnt;
    __func__ void append(const vec4i &element, scalar dist)
    {
        int id = atomic_add(cnt, 1);
        if (id < SS_SET_SIZE)
            a[id] = element;
        atomic_add(E, dist * dist);
    }
};

struct SphereGroundCollisionList
{
    int *a;
    int *cnt;
    scalar *E;
    vec2 *hr;
    __func__ void append(int element, const vec2 &hr, scalar dist_sqr)
    {
        int id = atomic_add(cnt, 1);
        if (id < G_SET_SIZE)
        {
            a[id] = element;
            this->hr[id] = hr;
        }
        atomic_add(E, dist_sqr * dist_sqr);
    }
};

struct StaticCollisionList
{
    vec2i *a;
    int *cnt;
    scalar *E;
    __func__ void append(const vec2i &element, scalar dist_sqr)
    {
        int id = atomic_add(cnt, 1);
        a[id] = element;
        atomic_add(E, dist_sqr * dist_sqr);
    }
};

__func__ bool is_1_ring(int a, int b, int c, int d)
{
    return (a == c || a == d || b == c || b == d);
}

__func__ int to_hash(int x, int y, MedialGeometry geo)
{
    int i = min(x, y);
    int j = max(x, y);
    int n_verts = geo.n_verts;
    return i * n_verts + j;
}

__func__ int lower_bound(int *arr, int start, int end, int value)
{
    int low = start;
    int high = end - 1;
    int result = -1; // Default if no element <= value

    while (low <= high)
    {
        int mid = (low + high) / 2;
        if (arr[mid] <= value)
        {
            result = mid;  // This index is a candidate
            low = mid + 1; // Try to find a larger one
        }
        else
        {
            high = mid - 1;
        }
    }
    return result; // -1 if none found, or index of largest b <= value
}

__func__ bool is_connected(MedialGeometry geo, int x, int y)
{
    int h = to_hash(x, y, geo);
    int f = lower_bound(geo.connectivity, 0, geo.n_edges, h);
    return h == geo.connectivity[f];
}
__func__ bool is_2_ring(MedialGeometry geo, int ea0, int ea1, int eb0, int eb1)
{
    return (is_connected(geo, ea0, eb0) || is_connected(geo, ea0, eb1) || is_connected(geo, ea1, eb0) || is_connected(geo, ea1, eb1));
}

// __func__ void cone_cone_collision_set(
//     MedialGeometry geo,
//     ConeConeCollisionList cc_list,
//     int i, int j,
//     int per_mesh_verts);
// __func__ void cone_cone_collision_set_static(
//     MedialGeometry geo,
//     MedialGeometry geo_static,
//     ConeConeCollisionList cc_list,
//     int i, int j);
// __func__ void slab_sphere_collision_set(
//     MedialGeometry geo,
//     SlabSphereCollisionList ss_list,
//     int i, int j,
//     int per_mesh_verts);
// __func__ void sphere_ground_set(
//     const MedialGeometry geo,
//     SphereGroundCollisionList g_list,
//     scalar ground_plane,
//     int i);

// __func__ void sphere_static_collision(
//     MedialGeometry geo,
//     TriangleSoup triangles_soup,
//     StaticCollisionList collision_list,
//     const int *neighbors, // assuming neighbors is a flat array
//     int i,
//     int n_triangles);

__func__ vec3 plane_normal_wp(const vec3 &v0, const vec3 &v1, const vec3 &v2)
{
    vec3 c = cross(v1 - v0, v2 - v0);
    return c.normalized();
}

struct TripletsDev
{
    int *rows, *cols, *cnt;
    mat33 *blocks;
    scalar *g;
};

struct MedialCollisionDetector
{
    Mat V, Vt;
    Mat V_rest;
    Vec R;
    Eigen::MatrixXi E, F;
    Eigen::VectorXi body;

    std::vector<MIPC::MipcConeConeConstraint> cc_set;
    std::vector<MIPC::MipcSlabSphereConstraint> ss_set;

    scalar ground = 0.0;

    MedialGeometry medial_geo;

    std::set<int> indices_set;

    wp_array<vec3> medial_geo_vertices;
    wp_array<scalar> medial_geo_radius;
    wp_array<vec2i> medial_geo_edges;
    wp_array<vec3i> medial_geo_faces;
    wp_array<int> medial_geo_connectivity;

    int n_vertices, n_edges, n_faces;

    ConeConeCollisionList ee_set;
    wp_array<vec4i> ee_set_a;
    wp_array<int> ee_set_cnt;
    wp_array<scalar> ee_set_dist;
    wp_array<scalar> ee_set_E;

    SlabSphereCollisionList pt_set;
    wp_array<vec4i> pt_set_a;
    wp_array<int> pt_set_cnt;
    wp_array<scalar> pt_set_dist;
    wp_array<scalar> pt_set_E;

    SphereGroundCollisionList g_set;
    wp_array<int> g_set_a;
    wp_array<int> g_set_cnt;
    wp_array<scalar> g_set_E;
    wp_array<vec2> g_set_hr;
    MedialCollisionDetector(int n_meshes, const Eigen::Matrix<scalar, -1, 3> &V, const Vec &R, const Eigen::Matrix<int, -1, 2> &E, const Eigen::Matrix<int, -1, 3> &F, scalar ground = 0.0, fem::TOBJComplex *static_objects = nullptr);
    void refit(const Mat &V, const Vec &R);
    void refit_host();
    void analyze(Vec &g, Eigen::SparseMatrix<scalar> &H, std::vector<int> &indices);
    void analyze(
        Vec &g, Eigen::SparseMatrix<scalar> &H, std::vector<int> &indices,
        std::vector<int> &rows, std::vector<int> &cols, std::vector<mat33> &blocks);

    TripletsDev analyze_dev(Vec &g);
    scalar collision_set();

    // call after refit()
    scalar collision_set_ss();
    scalar collision_set_cc();
    std::vector<vec4i> get_cc_set();
    std::vector<vec4i> get_ss_set();
    void analyze_ss(Vec &g, Mat &H, std::vector<int> &idx_ret);
    void analyze_cc(Vec &g, Mat &H, std::vector<int> &idx_ret);
    void define_static_mesh();

    inline int get_nss()
    {
        return ee_set.cnt[0];
    }
    inline int get_ncc()
    {
        return pt_set.cnt[0];
    }

    inline scalar energy()
    {
        return collision_set();
    }

    inline MIPC::CollideMedialSphere sphere(int i)
    {
        return MIPC::CollideMedialSphere(V.block<1, 3>(i, 0), V_rest.block<1, 3>(i, 0), R(i), i);
    }

    inline MIPC::CollideMedialSphere static_sphere(int i)
    {
        return MIPC::CollideMedialSphere(static_V[i], static_V[i], 0.0, i);
    }

    bool ss_collision = true, cc_collision = true, sg_collision = true, cc_static_collision = true;
    int per_mesh_verts = 30, n_meshes;

    // static mesh stuff
    Eigen::Matrix<int, -1, 2> E_static;
    wp_array<int> static_neighbors;
    wp_array<vec2i> static_edges;

    int n_edges_static = 0, n_triangles_static = 0;

    fem::TOBJComplex *static_objects = nullptr;
    TriangleSoup static_soup;
    wp_array<vec3i> static_soup_indices;
    wp_array<vec3> static_soup_vertices;
    wp_array<vec2i> static_soup_edges;

    // static collision lists
    ConeConeCollisionList c_static_set;
    wp_array<vec4i> c_static_set_a;
    wp_array<int> c_static_set_cnt;
    wp_array<scalar> c_static_set_dist;
    wp_array<scalar> c_static_set_E;

    // StaticCollisionList static_set;
    // wp_array<vec2i> static_set_a;
    // wp_array<int> static_set_cnt;
    // wp_array<scalar> static_set_E;
    // wp_array<vec3> static_set_v0;

    SlabSphereCollisionList p_static_set;
    wp_array<vec4i> p_static_set_a;
    wp_array<int> p_static_set_cnt;
    wp_array<scalar> p_static_set_dist;
    wp_array<scalar> p_static_set_E;

    inline int hashb(int i, int j)
    {
        return min(i, j) * n_meshes + max(i, j);
    }

    inline void unhash(int h, int &i, int &j)
    {
        i = h / n_meshes;
        j = h % n_meshes;
    }

    // bvh
    culbvh::LBVH cone_bvh, sphere_bvh, slab_bvh, static_triangle_bvh, static_edge_bvh;
    void compute_cone_bvh();
    void compute_sphere_bvh();
    void compute_slab_bvh();
    Shay::VtBuffer<aabb> cone_aabbs, sphere_aabbs, slab_aabbs, static_triangle_aabbs, static_edge_aabbs;
    Shay::VtBuffer<int2> d_res, d_res_sphere, d_res_cone;
    std::vector<vec3> static_V;

    Shay::VtBuffer<scalar> g_dev;
    TripletsDev triplets_dev;
    Shay::VtBuffer<int> triplet_rows, triplet_cols, triplet_cnt;
    Shay::VtBuffer<mat33> triplet_blocks;

    bool refit_cone_cone = false; 
};

void launch_slab_sphere_collision(
    MedialGeometry geo,
    SlabSphereCollisionList ss_list,
    int per_mesh_verts, int n_faces, int n_vertices);

void launch_cone_cone_collision(
    MedialGeometry geo,
    ConeConeCollisionList cc_list,
    int per_mesh_verts, int n_edges);

void launch_sphere_ground_collision(
    MedialGeometry geo,
    SphereGroundCollisionList g_list,
    scalar ground_plane,
    int per_mesh_verts, int n_vertices);

void launch_sphere_static_collision(
    MedialGeometry geo,
    TriangleSoup triangles_soup,
    SlabSphereCollisionList collision_list,
    const int *neighbors, // assuming neighbors is a flat array
    int n_pairs,
    int2 *culled_pairs);

void launch_cone_static_collision(
    MedialGeometry geo,
    TriangleSoup triangles_soup,
    ConeConeCollisionList collision_list,
    int n_pairs,
    int2 *culled_pairs);

void launch_sphere_aabb(
    MedialGeometry geo,
    aabb *aabbs);

void launch_cone_aabb(
    MedialGeometry geo,
    aabb *aabbs);

void launch_static_triangle_aabb(
    TriangleSoup triangle_soup,
    aabb *aabbs,
    int n_static_triangles);

void launch_static_edge_aabb(
    TriangleSoup triangle_soup,
    aabb *aabbs,
    int n_static_edges);

void launch_test_culled_pairs_cone_cone(
    int n_pairs,
    int2 *pairs,
    MedialGeometry geo,
    ConeConeCollisionList collision_list,
    int per_mesh_verts);

void launch_cc_fill_gh(
    MedialGeometry geo,
    ConeConeCollisionList cc_set,
    TripletsDev triplets);
void launch_ss_fill_gh(
    MedialGeometry geo,
    SlabSphereCollisionList ss_set,
    TripletsDev triplets);
void launch_g_fill_gh(
    MedialGeometry geo,
    SphereGroundCollisionList g_set,
    TripletsDev triplets
);