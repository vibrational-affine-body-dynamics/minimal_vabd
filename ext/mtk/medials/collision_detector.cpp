#include "common.cuh"
#include "medials/collision_detector.h"
#include "utils/buffer2eigen.h"
#include "medials/collision_cell_export.h"
#include <algorithm>
#include "medials/distance.h"
#include <map>
#include "utils/timer.h"
#include "utils/sparse.tpp"
#include "detector.cuh"
#include <algorithm>
#include <map>
#include <set>
using namespace std;
using namespace MIPC;
using namespace Eigen;

namespace
{
Eigen::MatrixXi unique_triangle_edges(const Eigen::MatrixXi &faces)
{
    std::set<std::pair<int, int>> edges;
    for (int face = 0; face < faces.rows(); ++face)
        for (int local = 0; local < 3; ++local)
        {
            int a = faces(face, local);
            int b = faces(face, (local + 1) % 3);
            edges.emplace(std::min(a, b), std::max(a, b));
        }
    Eigen::MatrixXi result(static_cast<int>(edges.size()), 2);
    int row = 0;
    for (const auto &edge : edges)
    {
        result(row, 0) = edge.first;
        result(row, 1) = edge.second;
        ++row;
    }
    return result;
}

Eigen::MatrixXi triangle_adjacency(const Eigen::MatrixXi &faces)
{
    Eigen::MatrixXi adjacency = Eigen::MatrixXi::Constant(faces.rows(), 3, -1);
    std::map<std::pair<int, int>, std::pair<int, int>> owner;
    for (int face = 0; face < faces.rows(); ++face)
        for (int local = 0; local < 3; ++local)
        {
            int a = faces(face, local);
            int b = faces(face, (local + 1) % 3);
            auto key = std::make_pair(std::min(a, b), std::max(a, b));
            auto found = owner.find(key);
            if (found == owner.end())
                owner.emplace(key, std::make_pair(face, local));
            else
            {
                adjacency(face, local) = found->second.first;
                adjacency(found->second.first, found->second.second) = face;
            }
        }
    return adjacency;
}
}

#define __CONE_STATIC_COLLISION__
#define __SPHERE_STATIC_COLLISION__
inline int to_hash_np(int x, int y, int n_verts)
{
    int i = min(x, y);
    int j = max(x, y);
    return i * n_verts + j;
}

inline vec3 plane_normal(const vec3 &v0, const vec3 &v1, const vec3 &v2)
{
    vec3 n = cross(v1 - v0, v2 - v0);
    return n / n.norm();
}
MedialCollisionDetector::MedialCollisionDetector(int n_mesehes, const Eigen::Matrix<scalar, -1, 3> &V, const Vec &R, const Eigen::Matrix<int, -1, 2> &E, const Eigen::Matrix<int, -1, 3> &F, scalar ground, fem::TOBJComplex *static_objects) : V(V), R(R), E(E), F(F), ground(ground), V_rest(V), n_meshes(n_meshes), static_objects(static_objects)
{
    per_mesh_verts = V.rows() / n_mesehes;
    // per_mesh_verts = 211;
    WpArray(V, medial_geo_vertices);
    medial_geo_radius.resize(R.rows());
    cudaMemcpy(medial_geo_radius.data(), R.data(), sizeof(scalar) * R.rows(), cudaMemcpyDefault);
    WpArray(E, medial_geo_edges);
    WpArray(F, medial_geo_faces);

    n_vertices = V.rows();
    n_edges = E.rows();
    n_faces = F.rows();

    // set up medial geometry
    medial_geo.n_verts = n_vertices;
    medial_geo.n_edges = n_edges;
    medial_geo.n_faces = n_faces;
    n_meshes = n_vertices / per_mesh_verts;

    medial_geo.vertices = medial_geo_vertices.data();
    medial_geo.radius = medial_geo_radius.data();
    medial_geo.edges = medial_geo_edges.data();
    medial_geo.faces = medial_geo_faces.data();

    vector<int> conn(E.rows());

    for (int i = 0; i < n_edges; i++)
    {
        conn[i] = to_hash_np(E(i, 0), E(i, 1), n_vertices);
    }
    sort(conn.begin(), conn.end());

    medial_geo_connectivity.resize(n_edges);
    cudaMemcpy(medial_geo_connectivity.data(), conn.data(), sizeof(int) * n_edges, cudaMemcpyDefault);
    medial_geo.connectivity = medial_geo_connectivity.data();
    // done setting up medial geometry

    ee_set_a.resize(CC_SET_SIZE);
    ee_set_cnt.resize(1);
    ee_set_E.resize(1);
    ee_set_dist.resize(CC_SET_SIZE);

    pt_set_a.resize(SS_SET_SIZE);
    pt_set_cnt.resize(1);
    pt_set_E.resize(1);

    g_set_a.resize(G_SET_SIZE);
    g_set_cnt.resize(1);
    g_set_E.resize(1);
    g_set_hr.resize(G_SET_SIZE);

    pt_set.a = pt_set_a.data();
    pt_set.cnt = pt_set_cnt.data();
    pt_set.E = pt_set_E.data();

    ee_set.a = ee_set_a.data();
    ee_set.cnt = ee_set_cnt.data();
    ee_set.E = ee_set_E.data();
    ee_set.dist = ee_set_dist.data();

    g_set.a = g_set_a.data();
    g_set.cnt = g_set_cnt.data();
    g_set.E = g_set_E.data();
    g_set.hr = g_set_hr.data();

    int triplet_size = (CC_SET_SIZE + SS_SET_SIZE) * 16 + G_SET_SIZE;

    // triplets
    triplet_cols.resize(triplet_size);
    triplet_rows.resize(triplet_size);
    triplet_blocks.resize(triplet_size);
    triplet_cnt.resize(1);
    triplets_dev.rows = triplet_rows;
    triplets_dev.cols = triplet_cols;
    triplets_dev.blocks = triplet_blocks;
    triplets_dev.cnt = triplet_cnt;
    
    g_dev.resize(n_vertices * 3);
    triplets_dev.g = g_dev; 
    
    
    static_objects = static_objects;
    define_static_mesh();

    // bvh
    cone_aabbs.resize(n_edges);
    d_res.resize(CC_BUF_SIZE);

    sphere_aabbs.resize(n_vertices);
    d_res_sphere.resize(ST_BUF_SIZE);

    d_res_cone.resize(CC_BUF_SIZE);
}
void MedialCollisionDetector::compute_cone_bvh()
{
    launch_cone_aabb(medial_geo, cone_aabbs);
    if (refit_cone_cone) {
        cone_bvh.refit();
    }
    else {
        cone_bvh.compute(cone_aabbs, n_edges);
    }
}

void MedialCollisionDetector::compute_sphere_bvh()
{
    launch_sphere_aabb(medial_geo, sphere_aabbs);
    sphere_bvh.compute(sphere_aabbs, n_vertices);
}
void MedialCollisionDetector::define_static_mesh()
{

    if (static_objects != nullptr)
    {

        E_static = unique_triangle_edges(static_objects->F);
        static_edges.resize(E_static.rows());
        WpArray(E_static, static_edges);

        c_static_set_a.resize(CC_SET_SIZE);
        c_static_set_cnt.resize(1);
        c_static_set_E.resize(1);
        c_static_set_dist.resize(CC_SET_SIZE);

        c_static_set.a = c_static_set_a.data();
        c_static_set.cnt = c_static_set_cnt.data();
        c_static_set.E = c_static_set_E.data();
        c_static_set.dist = c_static_set_dist.data();

        p_static_set_a.resize(SS_SET_SIZE);
        p_static_set_cnt.resize(1);
        p_static_set_E.resize(1);

        p_static_set.a = p_static_set_a;
        p_static_set.cnt = p_static_set_cnt.data();
        p_static_set.E = p_static_set_E.data();

        n_triangles_static = static_objects->F.rows();
        n_edges_static = E_static.rows();

        MatrixXi TT = triangle_adjacency(static_objects->F);
        static_neighbors.resize(TT.rows() * 3);
        cudaMemcpy(static_neighbors.data(), TT.data(), sizeof(int) * TT.rows() * 3, cudaMemcpyDefault);
        static_soup.vertices = static_objects->xcs;
        static_soup.edges = static_edges;
        static_soup.indices = static_objects->indices;

        // host-side buffers
        static_V = static_objects->xcs.get();

        // bvh for static triangles
        static_triangle_aabbs.resize(n_triangles_static);
        launch_static_triangle_aabb(static_soup, static_triangle_aabbs, n_triangles_static);
        cudaDeviceSynchronize();
        static_triangle_bvh.compute(static_triangle_aabbs, n_triangles_static);

        // bvh for static edges
        static_edge_aabbs.resize(n_edges_static);
        launch_static_edge_aabb(static_soup, static_edge_aabbs, n_edges_static);
        cudaDeviceSynchronize();
        static_edge_bvh.compute(static_edge_aabbs, n_edges_static);
    }
}
void MedialCollisionDetector::refit(const Mat &V, const Vec &R)
{
    this->V = V;
    this->R = R;

    from_eigen(medial_geo_vertices, V);
}

void MedialCollisionDetector::refit_host()
{
    // from_eigen(medial_geo_vertices, V);
    Vt.resize(3, n_vertices);
    cudaMemcpy(Vt.data(), medial_geo_vertices.data(), sizeof(vec3) * n_vertices, cudaMemcpyDeviceToHost);
    V = Vt.transpose();
}
scalar MedialCollisionDetector::collision_set_ss()
{
    pt_set_cnt.zero_();
    pt_set_E.zero_();
    launch_slab_sphere_collision(medial_geo, pt_set, per_mesh_verts, n_faces, n_vertices);
    cudaDeviceSynchronize();
    return pt_set_E.get()[0];
}

scalar MedialCollisionDetector::collision_set_cc()
{
    Timer timer;
    timer.start();
    ee_set_cnt.zero_();
    ee_set_E.zero_();
    // launch_cone_cone_collision(medial_geo, ee_set, per_mesh_verts, n_edges);

    compute_cone_bvh();

    timer.end("         cone-cone AABB & refit");
    timer.start();
    int tmp = cone_bvh.query(d_res, CC_BUF_SIZE);
    // printf("culled pairs %d\n", tmp);
    timer.end("         cone-cone bvh query");
    timer.start();
    launch_test_culled_pairs_cone_cone(tmp, d_res, medial_geo, ee_set, per_mesh_verts);
    timer.end("         cone-cone collision kernel");
    return ee_set_E.get()[0];
}
scalar MedialCollisionDetector::collision_set()
{
    // refit();
    scalar ret = 0.0;
#ifdef __CUDARUN__
    g_set_cnt.zero_();
    g_set_E.zero_();
    Timer timer;
    if (ss_collision)
    {
        timer.start();
        ret += collision_set_ss();
        timer.end("     slab-sphere collision");
    }
    if (cc_collision)
    {
        timer.start();
        ret += collision_set_cc();
        timer.end("     cone-cone collision");
    }
    if (sg_collision)
    {
        timer.start();
        launch_sphere_ground_collision(medial_geo, g_set, ground, per_mesh_verts, n_vertices);
        cudaDeviceSynchronize();
        ret += g_set_E.get()[0];
        timer.end("     sphere-ground collision");
    }

    // cout << "pt_set cnt = " << pt_set_cnt.get()[0] << endl;
    // cout << "ee_set cnt = " << ee_set_cnt.get()[0] << endl;
    // cout << "g_set cnt = " << g_set_cnt.get()[0] << endl;

    if (static_objects != nullptr)
    {
#ifdef __SPHERE_STATIC_COLLISION__
        timer.start();
        compute_sphere_bvh();
        int n_pairs = sphere_bvh.query(d_res_sphere, ST_BUF_SIZE, &static_triangle_bvh);
        p_static_set_cnt.zero_();
        p_static_set_E.zero_();
        launch_sphere_static_collision(medial_geo, static_soup, p_static_set, static_neighbors, n_pairs, d_res_sphere);
        ret += p_static_set_E.get()[0];
        timer.end("     sphere-static collision");
#endif

#ifdef __CONE_STATIC_COLLISION__
        timer.start();
        int n_pairs_cone = cone_bvh.query(d_res_cone, CC_BUF_SIZE, &static_edge_bvh);
        c_static_set_cnt.zero_();
        c_static_set_E.zero_();
        launch_cone_static_collision(medial_geo, static_soup, c_static_set, n_pairs_cone, d_res_cone);
        ret += c_static_set_E.get()[0];
        timer.end("     cone-static collision");
#endif
    }
#else
    *(pt_set.cnt) = 0;
    *(pt_set.E) = 0.0;
    *(ee_set.cnt) = 0;
    *(ee_set.E) = 0.0;
    *(g_set.cnt) = 0;
    *(g_set.E) = 0.0;
    if (ss_collision)
    {
        for (int i = 0; i < n_faces; i++)
            for (int j = 0; j < n_vertices; j++)
                slab_sphere_collision_set(medial_geo, pt_set, i, j, per_mesh_verts);

        ret += pt_set.E[0];
    }
    if (cc_collision)
    {
        for (int i = 0; i < n_edges; i++)
            for (int j = i + 1; j < n_edges; j++)
            {
                cone_cone_collision_set(medial_geo, ee_set, i, j, per_mesh_verts);
            }

        ret += ee_set.E[0];
    }
    if (sg_collision)
    {
        for (int i = 0; i < n_vertices; i++)
        {
            sphere_ground_set(medial_geo, g_set, ground, i);
        }
        ret += g_set.E[0];
    }
    cout << "pt_set cnt = " << pt_set_cnt[0] << endl;
    cout << "ee_set cnt = " << ee_set_cnt[0] << endl;
    cout << "g_set cnt = " << g_set_cnt[0] << endl;
#endif

    return ret;
}
std::vector<vec4i> MedialCollisionDetector::get_cc_set()
{
    int ncc = ee_set_cnt.get()[0];
    vector<vec4i> cc_id(ncc);
    cudaMemcpy(cc_id.data(), ee_set.a, sizeof(vec4i) * ncc, cudaMemcpyDefault);
    return cc_id;
}

std::vector<vec4i> MedialCollisionDetector::get_ss_set()
{
    int nss = pt_set_cnt.get()[0];
    vector<vec4i> ss_id(nss);
    cudaMemcpy(ss_id.data(), pt_set.a, sizeof(vec4i) * nss, cudaMemcpyDefault);
    return ss_id;
}
void MedialCollisionDetector::analyze(Vec &g, Eigen::SparseMatrix<scalar> &H, std::vector<int> &idx_ret)
{
    vector<int> rows, cols;
    vector<mat33> blocks;
    analyze(g, H, idx_ret, rows, cols, blocks);
}


TripletsDev MedialCollisionDetector::analyze_dev(Vec &g)
{
    triplet_cnt.zero_();
    g_dev.zero_();
    launch_cc_fill_gh(medial_geo, ee_set, triplets_dev);
    cudaDeviceSynchronize();

    launch_ss_fill_gh(medial_geo, pt_set, triplets_dev);
    cudaDeviceSynchronize();

    launch_g_fill_gh(medial_geo, g_set, triplets_dev);

    g.resize(n_vertices * 3);
    cudaMemcpy(g.data(), g_dev.data(), sizeof(scalar) * n_vertices * 3, cudaMemcpyDefault);
    return triplets_dev;
}

void MedialCollisionDetector::analyze(
    Vec &g, Eigen::SparseMatrix<scalar> &H, std::vector<int> &idx_ret,
    std::vector<int> &rows, std::vector<int> &cols, std::vector<mat33> &blocks)
{
    refit_host();
    Timer timer;
    timer.start();
    Vec b(n_vertices * 3);
    b.setZero();
    indices_set.clear();

    int ng = g_set_cnt.get()[0];
    auto ss_id = get_ss_set();
    auto cc_id = get_cc_set();
    vector<int> sg_id(ng);
    vector<vec2> sg_hr(ng);
    cudaMemcpy(sg_id.data(), g_set.a, sizeof(int) * ng, cudaMemcpyDefault);
    cudaMemcpy(sg_hr.data(), g_set.hr, sizeof(vec2) * ng, cudaMemcpyDefault);

    cc_set.clear();
    ss_set.clear();
    // timer.start();
    for (auto &id : ss_id)
    {
        CollideMedialSphere s0(sphere(id[0])), s1(sphere(id[1])), s2(sphere(id[2])), s3(sphere(id[3]));
        MipcSlabSphereConstraint cons(s0, s1, s2, s3);

        scalar dist = abs(cons.computeDistance());
        // ss_set.push_back(cons);

        vec12 g;
        mat12 h;
        cons.get_dist_gh(g, h);
        int e0 = id[0], e1 = id[1], e2 = id[2], e3 = id[3];
        b.segment<3>(e0 * 3) += g.segment<3>(0) * 2.0 * dist;
        b.segment<3>(e1 * 3) += g.segment<3>(3) * 2.0 * dist;
        b.segment<3>(e2 * 3) += g.segment<3>(6) * 2.0 * dist;
        b.segment<3>(e3 * 3) += g.segment<3>(9) * 2.0 * dist;

        // h = 2 * dist * h + 2 * g * g.transpose();
        h = 2 * g * g.transpose();

        // indices_set.insert(e0);
        // indices_set.insert(e1);
        // indices_set.insert(e2);
        // indices_set.insert(e3);
        for (int ii = 0; ii < 4; ii++)
            for (int jj = 0; jj < 4; jj++)
            {
                rows.push_back(id[ii]);
                cols.push_back(id[jj]);
                blocks.push_back(h.block<3, 3>(ii * 3, jj * 3));
            }
    }

    // timer.end("analyze ss collision");
    // timer.start();
    // for (auto &id : cc_id)
    // #pragma omp parallel for
    for (int kk = 0; kk < cc_id.size(); kk++)
    {
        vec4i &id = cc_id[kk];
        CollideMedialSphere s0(sphere(id[0])), s1(sphere(id[1])), s2(sphere(id[2])), s3(sphere(id[3]));
        MipcConeConeConstraint cons(s0, s1, s2, s3);

        scalar dist = abs(cons.computeDistance());
        // cc_set.push_back(cons);
        vec12 g;
        mat12 h;
        cons.get_dist_gh(g, h);
        int e0 = id[0], e1 = id[1], e2 = id[2], e3 = id[3];
        b.segment<3>(e0 * 3) += g.segment<3>(0) * 2.0 * dist;
        b.segment<3>(e1 * 3) += g.segment<3>(3) * 2.0 * dist;
        b.segment<3>(e2 * 3) += g.segment<3>(6) * 2.0 * dist;
        b.segment<3>(e3 * 3) += g.segment<3>(9) * 2.0 * dist;

        h = 2 * g * g.transpose();

        // indices_set.insert(e0);
        // indices_set.insert(e1);
        // indices_set.insert(e2);
        // indices_set.insert(e3);
        for (int ii = 0; ii < 4; ii++)
            for (int jj = 0; jj < 4; jj++)
            {
                rows.push_back(id[ii]);
                cols.push_back(id[jj]);
                blocks.push_back(h.block<3, 3>(ii * 3, jj * 3));
            }
    }

    // timer.end("analyze cc collision");
    // timer.start();

    // #pragma omp parallel for
    for (int i = 0; i < sg_id.size(); i++)
    {
        scalar h = sg_hr[i](0), r = sg_hr[i](1);
        mat33 hh = mat33::Zero();
        scalar h11 = 12 * (h - r) * (h - r);
        hh(1, 1) = h11;

        scalar gg1 = (h - r) * h11 / 3;
        vec3 gg(0.0, gg1, 0.0);

        int id = sg_id[i];
        b.segment<3>(id * 3) += gg;

        rows.push_back(id);
        cols.push_back(id);
        blocks.push_back(hh);
        // indices_set.insert(id);
    }

    // timer.end("analyze sg collision");
    // timer.start();

    if (static_objects != nullptr)
    {
#ifdef __SPHERE_STATIC_COLLISION__
        vector<vec4i> st_id;
        int n_st = p_static_set_cnt.get()[0];
        st_id.resize(n_st);
        cudaMemcpy(st_id.data(), p_static_set.a, sizeof(vec4i) * n_st, cudaMemcpyDefault);
        for (auto &id : st_id)
        {
            int i = id[3];
            int it0 = id[0];
            int it1 = id[1];
            int it2 = id[2];

            vec3 t0 = static_V[it0];
            vec3 t1 = static_V[it1];
            vec3 t2 = static_V[it2];

            vec3 tn = plane_normal(t0, t1, t2);
            scalar h = tn.dot(this->V.row(i).transpose() - t0);
            scalar r = this->R[i];
            mat33 hh = mat33::Zero();

            scalar h11 = 12 * (h - r) * (h - r);
            hh += tn * tn.transpose() * h11;
            scalar gg1 = (h - r) * h11 / 3;
            vec3 gg = gg1 * tn;
            b.segment<3>(i * 3) += gg;
            rows.push_back(i);
            cols.push_back(i);
            blocks.push_back(hh);
            // indices_set.insert(i);
        }
#endif
#ifdef __CONE_STATIC_COLLISION__
        vector<vec4i> ce_id;
        int n_ce = c_static_set_cnt.get()[0];
        ce_id.resize(n_ce);
        cudaMemcpy(ce_id.data(), c_static_set.a, sizeof(vec4i) * n_ce, cudaMemcpyDefault);
        for (auto &id : ce_id)
        {
            CollideMedialSphere s0(sphere(id[0])), s1(sphere(id[1])), s2(static_sphere(id[2])), s3(static_sphere(id[3]));
            MipcConeConeConstraint cons(s0, s1, s2, s3);

            scalar dist = abs(cons.computeDistance());
            // cc_set.push_back(cons);
            vec12 g;
            mat12 h;
            cons.get_dist_gh(g, h);
            int e0 = id[0], e1 = id[1];
            b.segment<3>(e0 * 3) += g.segment<3>(0) * 2.0 * dist;
            b.segment<3>(e1 * 3) += g.segment<3>(3) * 2.0 * dist;

            h = 2 * dist * h + 2 * g * g.transpose();

            // indices_set.insert(e0);
            // indices_set.insert(e1);
            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                {
                    rows.push_back(id[ii]);
                    cols.push_back(id[jj]);
                    blocks.push_back(h.block<3, 3>(ii * 3, jj * 3));
                }
        }
#endif
    }
    // vector<int> idx;
    // idx.assign(indices_set.begin(), indices_set.end());

    // int H_dim = idx.size() * 3;
    // // H.resize(H_dim, H_dim);
    // std::map<int, int> idx_inv;
    // for (int i = 0; i < idx.size(); i++)
    // {
    //     idx_inv[idx[i]] = i;
    // }

    // for (int i = 0; i < rows.size(); i++)
    //{
    //     int r = rows[i];
    //     int c = cols[i];

    //    int ii = idx_inv[r];
    //    int jj = idx_inv[c];

    //    auto block = blocks[i];
    //    H.block<3, 3>(ii * 3, jj * 3) += block;
    //}
    timer.end("     analyze: gen submatrices");
    timer.start();
    H.resize(n_vertices * 3, n_vertices * 3);
    // sparse_utils::gen_empty_sm<3>(rows, cols, blocks, H);
    timer.end("     analyze: fill H");
    timer.start();
    // std::set<int> body_indices;
    // for (int i = 0; i < rows.size(); i++)
    // {
    //     int br = rows[i] / per_mesh_verts;
    //     int bc = cols[i] / per_mesh_verts;
    //     body_indices.insert(hashb(br, bc));
    // }

    // for (auto &bij : body_indices)
    // {
    //     int r, c;
    //     unhash(bij, r, c);
    //     idx_ret.push_back(r);
    //     idx_ret.push_back(c);
    //     if (r != c)
    //     {
    //         idx_ret.push_back(c);
    //         idx_ret.push_back(r);
    //     }
    // }
    vector<vec3> lowers(n_meshes), uppers(n_meshes);
    const auto intersects = [](const vec3 &l0, const vec3 &u0, const vec3 &l1, const vec3 &u1)
    {
        return (l0(0) <= u1(0) && u0(0) >= l1(0)) &&
               (l0(1) <= u1(1) && u0(1) >= l1(1)) &&
               (l0(2) <= u1(2) && u0(2) >= l1(2));
    };
    for (int i = 0; i < n_meshes; i++)
    {
        vec3 li(FLT_MAX, FLT_MAX, FLT_MAX);
        vec3 ui = -li;
        for (int j = 0; j < per_mesh_verts; j++)
        {
            int ii = i * per_mesh_verts + j;
            vec3 vi = V.row(ii).transpose();
            scalar ri = R(ii) + 0.1;
            vec3 inflate = vec3(ri, ri, ri);
            li = li.cwiseMin(vi - inflate);
            ui = ui.cwiseMax(vi + inflate);
        }
        lowers[i] = li;
        uppers[i] = ui;
    }
    for (int i = 0; i < n_meshes; i++)
    {
        // if (lowers[i](1) < ground)
        if (true)
        {
            idx_ret.push_back(i);
            idx_ret.push_back(i);
        }
        for (int j = 0; j < i; j++)
        {
            if (intersects(lowers[i], uppers[i], lowers[j], uppers[j]))
            {
                idx_ret.push_back(i);
                idx_ret.push_back(j);
                idx_ret.push_back(j);
                idx_ret.push_back(i);
            }
        }
    }
    timer.end("     analyze: fill indices");
    // idx_ret.resize(idx.size() * 3);
    // for (int i = 0; i < idx.size(); i++)
    // {
    //     int r = idx[i];
    //     idx_ret[3 * i] = (3 * r);
    //     idx_ret[3 * i + 1] = 3 * r + 1;
    //     idx_ret[3 * i + 2] = 3 * r + 2;
    // }

    // g.resize(H_dim);
    // for (int i = 0; i < H_dim; i++)
    // {
    //     g[i] = b[idx_ret[i]];
    // }
    g = b;

    // timer.end("analyze: fill H and g");
}

void MedialCollisionDetector::analyze_cc(Vec &g, Mat &H, std::vector<int> &idx_ret)
{
#ifndef __CUDARUN__
    Timer timer;
    Vec b(n_vertices * 3);
    b.setZero();
    vector<int> rows, cols;
    vector<mat33> blocks;
    indices_set.clear();

    int ncc = ee_set.cnt[0];
    vector<vec4i> cc_id(ncc);
    cudaMemcpy(cc_id.data(), ee_set.a, sizeof(vec4i) * ncc, cudaMemcpyDefault);
    cout << "cc checkpoint 0" << endl;
    cc_set.clear();
    timer.start();
    for (auto &id : cc_id)
    {
        CollideMedialSphere s0(sphere(id[0])), s1(sphere(id[1])), s2(sphere(id[2])), s3(sphere(id[3]));
        MipcConeConeConstraint cons(s0, s1, s2, s3);

        cout << "cc checkpoint 1" << endl;
        scalar dist = abs(cons.computeDistance());
        // cc_set.push_back(cons);
        vec12 g;
        mat12 h;
        cout << "cc checkpoint 2" << endl;
        cons.get_dist_gh(g, h);
        int e0 = id[0], e1 = id[1], e2 = id[2], e3 = id[3];
        b.segment<3>(e0 * 3) += g.segment<3>(0) * 2.0 * dist;
        b.segment<3>(e1 * 3) += g.segment<3>(3) * 2.0 * dist;
        b.segment<3>(e2 * 3) += g.segment<3>(6) * 2.0 * dist;
        b.segment<3>(e3 * 3) += g.segment<3>(9) * 2.0 * dist;

        h = 2 * dist * h + 2 * g * g.transpose();
        indices_set.insert(e0);
        indices_set.insert(e1);
        indices_set.insert(e2);
        indices_set.insert(e3);
        cout << "cc checkpoint 3" << endl;
        for (int ii = 0; ii < 4; ii++)
            for (int jj = 0; jj < 4; jj++)
            {
                rows.push_back(id[ii]);
                cols.push_back(id[jj]);
                blocks.push_back(h.block<3, 3>(ii * 3, jj * 3));
            }
    }

    timer.end("analyze cc collision");
    vector<int> idx;
    idx.assign(indices_set.begin(), indices_set.end());

    int H_dim = idx.size() * 3;
    H.resize(H_dim, H_dim);
    H.setZero();
    std::map<int, int> idx_inv;
    for (int i = 0; i < idx.size(); i++)
    {
        idx_inv[idx[i]] = i;
    }

    for (int i = 0; i < rows.size(); i++)
    {
        int r = rows[i];
        int c = cols[i];

        int ii = idx_inv[r];
        int jj = idx_inv[c];

        auto block = blocks[i];
        H.block<3, 3>(ii * 3, jj * 3) += block;
    }

    idx_ret.resize(idx.size() * 3);
    for (int i = 0; i < idx.size(); i++)
    {
        int r = idx[i];
        idx_ret[3 * i] = (3 * r);
        idx_ret[3 * i + 1] = 3 * r + 1;
        idx_ret[3 * i + 2] = 3 * r + 2;
    }
    g.resize(H_dim);
    for (int i = 0; i < H_dim; i++)
    {
        g[i] = b[idx_ret[i]];
    }
#endif
}

void MedialCollisionDetector::analyze_ss(Vec &g, Mat &H, std::vector<int> &idx_ret)
{
#ifndef __CUDARUN__
    Timer timer;
    Vec b(n_vertices * 3);
    b.setZero();
    vector<int> rows, cols;
    vector<mat33> blocks;
    indices_set.clear();

    int nss = pt_set.cnt[0];
    vector<vec4i> ss_id(nss);
    cudaMemcpy(ss_id.data(), pt_set.a, sizeof(vec4i) * nss, cudaMemcpyDefault);
    cout << "ss checkpoint 0" << endl;

    ss_set.clear();
    timer.start();
    for (auto &id : ss_id)
    {
        CollideMedialSphere s0(sphere(id[0])), s1(sphere(id[1])), s2(sphere(id[2])), s3(sphere(id[3]));
        MipcSlabSphereConstraint cons(s0, s1, s2, s3);
        cout << "ss checkpoint 1" << endl;

        scalar dist = abs(cons.computeDistance());
        // ss_set.push_back(cons);

        vec12 g;
        mat12 h;
        cout << "ss checkpoint 2" << endl;

        cons.get_dist_gh(g, h);
        int e0 = id[0], e1 = id[1], e2 = id[2], e3 = id[3];
        b.segment<3>(e0 * 3) += g.segment<3>(0) * 2.0 * dist;
        b.segment<3>(e1 * 3) += g.segment<3>(3) * 2.0 * dist;
        b.segment<3>(e2 * 3) += g.segment<3>(6) * 2.0 * dist;
        b.segment<3>(e3 * 3) += g.segment<3>(9) * 2.0 * dist;

        h = 2 * dist * h + 2 * g * g.transpose();

        indices_set.insert(e0);
        indices_set.insert(e1);
        indices_set.insert(e2);
        indices_set.insert(e3);
        cout << "ss checkpoint 3" << endl;

        for (int ii = 0; ii < 4; ii++)
            for (int jj = 0; jj < 4; jj++)
            {
                rows.push_back(id[ii]);
                cols.push_back(id[jj]);
                blocks.push_back(h.block<3, 3>(ii * 3, jj * 3));
            }
    }

    timer.end("analyze ss collision");

    vector<int> idx;
    idx.assign(indices_set.begin(), indices_set.end());

    int H_dim = idx.size() * 3;
    H.resize(H_dim, H_dim);
    H.setZero();
    std::map<int, int> idx_inv;
    for (int i = 0; i < idx.size(); i++)
    {
        idx_inv[idx[i]] = i;
    }

    for (int i = 0; i < rows.size(); i++)
    {
        int r = rows[i];
        int c = cols[i];

        int ii = idx_inv[r];
        int jj = idx_inv[c];

        auto block = blocks[i];
        H.block<3, 3>(ii * 3, jj * 3) += block;
    }

    idx_ret.resize(idx.size() * 3);
    for (int i = 0; i < idx.size(); i++)
    {
        int r = idx[i];
        idx_ret[3 * i] = (3 * r);
        idx_ret[3 * i + 1] = 3 * r + 1;
        idx_ret[3 * i + 2] = 3 * r + 2;
    }
    g.resize(H_dim);
    for (int i = 0; i < H_dim; i++)
    {
        g[i] = b[idx_ret[i]];
    }
    timer.end("analyze: fill H and g");
#endif
}
