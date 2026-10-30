#pragma once
#include "common.cuh"
#define ZERO 1e-6
__func__ void point_triangle_distance_wp(
    const vec3 &v0, const vec3 &v1, const vec3 &v2, const vec3 &v, int &dist_type, scalar &dist)
{
    vec3 e1 = v1 - v0;
    vec3 e2 = v2 - v0;
    vec3 e3 = v2 - v1;

    vec3 n = cross(e1, e2);
    vec3 na = cross(v2 - v1, v - v1);
    vec3 nb = cross(v0 - v2, v - v2);
    vec3 nc = cross(v1 - v0, v - v0);

    vec3 bary(n.dot(na), n.dot(nb), n.dot(nc));
    bary /= n.squaredNorm();

    vec3 bary_abs = bary.cwiseAbs();
    scalar bary_sum = bary_abs.sum();

    scalar ret = FLT_MAX;
    int type = 0;

    if (bary_sum - 1.0 < ZERO)
    {
        vec3 n_hat = n.normalized();
        scalar d = n_hat.dot(v - v0);
        ret = fabs(d);
        type = 0;
    }
    else
    {
        vec3 ev = v - v0;
        vec3 ev3 = v - v1;
        vec3 e1_hat = e1.normalized();
        vec3 e2_hat = e2.normalized();
        vec3 e3_hat = e3.normalized();

        scalar e1_dot = e1_hat.dot(ev);
        scalar e2_dot = e2_hat.dot(ev);
        scalar e3_dot = e3_hat.dot(ev3);

        scalar e1_norm = e1.norm();
        scalar e2_norm = e2.norm();
        scalar e3_norm = e3.norm();

        vec3 edge_dists(FLT_MAX, FLT_MAX, FLT_MAX);

        if (e1_dot > 0.0 && e1_dot < e1_norm)
            edge_dists[0] = (v - (v0 + e1_hat * e1_dot)).norm();
        if (e2_dot > 0.0 && e2_dot < e2_norm)
            edge_dists[1] = (v - (v0 + e2_hat * e2_dot)).norm();
        if (e3_dot > 0.0 && e3_dot < e3_norm)
            edge_dists[2] = (v - (v1 + e3_hat * e3_dot)).norm();

        vec3 vert_dists((v - v0).norm(), (v - v1).norm(), (v - v2).norm());

        Eigen::MatrixXf::Index min_id;
        scalar v_min = vert_dists.minCoeff();
        scalar e_min = edge_dists.minCoeff(&min_id);

        type = (v_min < e_min) ? 1 : min_id + 2;
        ret = fminf(v_min, e_min);
    }

    dist_type = type;
    dist = ret;
}

__func__ bool point_projects_inside_triangle(const vec3 &v0, const vec3 &v1, const vec3 &v2, const vec3 &v)
{
    vec3 e1 = v1 - v0;
    vec3 e2 = v2 - v0;
    vec3 n = cross(e1, e2);

    vec3 na = cross(v2 - v1, v - v1);
    vec3 nb = cross(v0 - v2, v - v2);
    vec3 nc = cross(v1 - v0, v - v0);

    vec3 bary(n.dot(na), n.dot(nb), n.dot(nc));
    bary /= n.squaredNorm();

    return bary.cwiseAbs().sum() - 1.0 < ZERO;
}
__func__ bool inside_collision_cell(
    const TriangleSoup &triangle_soup,
    const int *neighbors,
    int tid,
    const vec3 &p)
{
    bool ret = true;

    vec3 v0 = triangle_soup.vertices[triangle_soup.indices[tid][0]];
    vec3 v1 = triangle_soup.vertices[triangle_soup.indices[tid][1]];
    vec3 v2 = triangle_soup.vertices[triangle_soup.indices[tid][2]];

    vec3 v[3] = {v0, v1, v2};
    vec3 n = plane_normal_wp(v0, v1, v2);

    vec3 nn[3];

    for (int x = 0; x < 3; ++x)
    {
        int tn = neighbors[tid * 3 + x];
        if (tn != -1)
        {
            vec3 vn0 = triangle_soup.vertices[triangle_soup.indices[tn][0]];
            vec3 vn1 = triangle_soup.vertices[triangle_soup.indices[tn][1]];
            vec3 vn2 = triangle_soup.vertices[triangle_soup.indices[tn][2]];

            nn[x] = plane_normal_wp(vn0, vn1, vn2);
        }
        else
        {
            ret = false;
        }
    }

    if (ret)
    {
        for (int x = 0; x < 3; ++x)
        {
            vec3 ne = (nn[x] + n).normalized();

            vec3 eij = v[(x + 1) % 3] - v[x];
            vec3 neb = cross(ne, eij);
            vec3 neb_hat = neb.normalized();

            scalar deplane = neb_hat.dot(p - v[x]);
            if (deplane < 0.0)
            {
                ret = false;
            }
        }
    }

    return ret;
}
