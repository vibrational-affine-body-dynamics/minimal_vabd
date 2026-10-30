#pragma once
#include "medials/collision_detector.h"
#include "medials/collision_cell_export.h"
__func__ vec3 compute_distance_cone_cone(
    const vec3 &c0, const vec3 &c1, const vec3 &c2, const vec3 &c3,
    scalar r0, scalar r1, scalar r2, scalar r3)
{
    vec3 sC1 = c0 - c1;
    vec3 sC2 = c3 - c2;
    vec3 sC3 = c1 - c3;

    scalar sR1 = r0 - r1;
    scalar sR2 = r2 - r3;
    scalar sR3 = r1 + r3;

    scalar A = sC1.dot(sC1) - sR1 * sR1;
    scalar B = 2.0 * (sC1.dot(sC2) - sR1 * sR2);
    scalar C = sC2.dot(sC2) - sR2 * sR2;
    scalar D = 2.0 * (sC1.dot(sC3) - sR1 * sR3);
    scalar E = 2.0 * (sC2.dot(sC3) - sR2 * sR3);
    scalar F = sC3.dot(sC3) - sR3 * sR3;

    scalar delta = 4 * A * C - B * B;

    scalar temp_alpha, temp_beta;
    scalar temp_dist;

    scalar alpha = 0.0, beta = 0.0;
    scalar distance = CDMM::valueOfQuadraticSurface2D(alpha, beta, A, B, C, D, E, F);

    temp_alpha = 1.0, temp_beta = 0.0;
    temp_dist = CDMM::valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
    if (distance > temp_dist)
    {
        distance = temp_dist;
        alpha = temp_alpha;
        beta = temp_beta;
    }

    temp_alpha = 0.0, temp_beta = 1.0;
    temp_dist = CDMM::valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
    if (distance > temp_dist)
    {
        distance = temp_dist;
        alpha = temp_alpha;
        beta = temp_beta;
    }

    temp_alpha = 1.0, temp_beta = 1.0;
    temp_dist = CDMM::valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
    if (distance > temp_dist)
    {
        distance = temp_dist;
        alpha = temp_alpha;
        beta = temp_beta;
    }

    temp_alpha = 0.0;
    temp_beta = -E / (2.0 * C);
    if (temp_beta > 0.0 && temp_beta < 1.0)
    {
        temp_dist = CDMM::valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
        if (distance > temp_dist)
        {
            distance = temp_dist;
            alpha = temp_alpha;
            beta = temp_beta;
        }
    }

    temp_alpha = 1.0;
    temp_beta = -(B + E) / (2.0 * C);
    if (temp_beta > 0.0 && temp_beta < 1.0)
    {
        temp_dist = CDMM::valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
        if (distance > temp_dist)
        {
            distance = temp_dist;
            alpha = temp_alpha;
            beta = temp_beta;
        }
    }

    temp_alpha = -D / (2.0 * A);
    temp_beta = 0.0;
    if (temp_alpha > 0.0 && temp_alpha < 1.0)
    {
        temp_dist = CDMM::valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
        if (distance > temp_dist)
        {
            distance = temp_dist;
            alpha = temp_alpha;
            beta = temp_beta;
        }
    }

    temp_alpha = -(B + D) / (2.0 * A);
    temp_beta = 1.0;
    if (temp_alpha > 0.0 && temp_alpha < 1.0)
    {
        temp_dist = CDMM::valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
        if (distance > temp_dist)
        {
            distance = temp_dist;
            alpha = temp_alpha;
            beta = temp_beta;
        }
    }

    if (delta != 0.0)
    {
        temp_alpha = (B * E - 2.0 * C * D) / delta;
        temp_beta = (B * D - 2.0 * A * E) / delta;
        if (temp_alpha > 0.0 && temp_alpha < 1.0 && temp_beta > 0.0 && temp_beta < 1.0)
        {
            temp_dist = CDMM::valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
            if (distance > temp_dist)
            {
                distance = temp_dist;
                alpha = temp_alpha;
                beta = temp_beta;
            }
        }
    }

    return vec3{distance, alpha, beta};
}

__func__ vec3 compute_distance_slab_sphere(const vec3 &c0, const vec3 &c1, const vec3 &c2, const vec3 &c3,
                                             scalar r0, scalar r1, scalar r2, scalar r3)
{
    vec3 sC1 = c0 - c2;
    vec3 sC2 = c1 - c2;
    vec3 sC3 = c2 - c3;

    scalar sR1 = r0 - r2;
    scalar sR2 = r1 - r2;
    scalar sR3 = r2 + r3;

    scalar A = sC1.dot(sC1) - sR1 * sR1;
    scalar B = 2.0 * (sC1.dot(sC2) - sR1 * sR2);
    scalar C = sC2.dot(sC2) - sR2 * sR2;
    scalar D = 2.0 * (sC1.dot(sC3) - sR1 * sR3);
    scalar E = 2.0 * (sC2.dot(sC3) - sR2 * sR3);
    scalar F = sC3.dot(sC3) - sR3 * sR3;

    scalar delta = 4 * A * C - B * B;

    scalar temp_alpha, temp_beta;
    scalar temp_dist;
    scalar alpha = 0.0, beta = 0.0;
    scalar distance = CDMM::valueOfQuadraticSurface2D(alpha, beta, A, B, C, D, E, F);

    temp_alpha = 1.0;
    temp_beta = 0.0;
    temp_dist = CDMM::valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
    if (distance > temp_dist)
    {
        distance = temp_dist;
        alpha = temp_alpha;
        beta = temp_beta;
    }

    temp_alpha = 0.0;
    temp_beta = 1.0;
    temp_dist = CDMM::valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
    if (distance > temp_dist)
    {
        distance = temp_dist;
        alpha = temp_alpha;
        beta = temp_beta;
    }

    temp_alpha = 0.0;
    temp_beta = -E / (2.0 * C);
    if (temp_beta > 0.0 && temp_beta < 1.0)
    {
        temp_dist = CDMM::valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
        if (distance > temp_dist)
        {
            distance = temp_dist;
            alpha = temp_alpha;
            beta = temp_beta;
        }
    }

    temp_alpha = -D / (2.0 * A);
    temp_beta = 0.0;
    if (temp_alpha > 0.0 && temp_alpha < 1.0)
    {
        temp_dist = CDMM::valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
        if (distance > temp_dist)
        {
            distance = temp_dist;
            alpha = temp_alpha;
            beta = temp_beta;
        }
    }

    temp_alpha = 0.5 * (2.0 * C + E - B - D) / (A - B + C);
    temp_beta = 1.0 - temp_alpha;
    if (temp_alpha > 0.0 && temp_alpha < 1.0)
    {
        temp_dist = CDMM::valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
        if (distance > temp_dist)
        {
            distance = temp_dist;
            alpha = temp_alpha;
            beta = temp_beta;
        }
    }

    // can be ignored
    if (delta != 0.0)
    {
        temp_alpha = (B * E - 2.0 * C * D) / delta;
        temp_beta = (B * D - 2.0 * A * E) / delta;
        if (temp_alpha > 0.0 && temp_alpha < 1.0 && temp_beta > 0.0 && temp_beta < 1.0 && temp_alpha + temp_beta < 1.0)
        {
            temp_dist = CDMM::valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
            if (distance > temp_dist)
            {
                distance = temp_dist;
                alpha = temp_alpha;
                beta = temp_beta;
            }
        }
    }

    return {distance, alpha, beta};
}

__func__ void cone_cone_collision_set(
    MedialGeometry geo,
    ConeConeCollisionList cc_list,
    int i, int j,
    int per_mesh_verts)
{
    if (i < j)
    {
        const vec2i &ex = geo.edges[i];
        const vec2i &ey = geo.edges[j];

        vec3 c0 = geo.vertices[ex[0]];
        vec3 c1 = geo.vertices[ex[1]];
        vec3 c2 = geo.vertices[ey[0]];
        vec3 c3 = geo.vertices[ey[1]];

        scalar r0 = geo.radius[ex[0]];
        scalar r1 = geo.radius[ex[1]];
        scalar r2 = geo.radius[ey[0]];
        scalar r3 = geo.radius[ey[1]];

        auto distab = compute_distance_cone_cone(c0, c1, c2, c3, r0, r1, r2, r3);
        scalar dist = distab[0];

        bool hack = (ex[0] / per_mesh_verts) == (ey[0] / per_mesh_verts);
        // bool refuse_cond =
        //     is_1_ring(ex[0], ex[1], ey[0], ey[1]) ||
        //     is_2_ring(geo, ex[0], ex[1], ey[0], ey[1]) ||
        //     hack;
        bool refuse_cond = hack;

        if (dist < 0.0f && !refuse_cond)
        {
            vec4i col(ex[0], ex[1], ey[0], ey[1]);
            cc_list.append(col, dist);
        }
    }
}

__func__ void slab_sphere_collision_set(
    MedialGeometry geo,
    SlabSphereCollisionList ss_list,
    int i, int j,
    int per_mesh_verts)
{
    const auto &slab = geo.faces[i];
    const vec3 &b = geo.vertices[j];

    vec3 c0 = geo.vertices[slab[0]];
    vec3 c1 = geo.vertices[slab[1]];
    vec3 c2 = geo.vertices[slab[2]];

    scalar r0 = geo.radius[slab[0]];
    scalar r1 = geo.radius[slab[1]];
    scalar r2 = geo.radius[slab[2]];
    scalar rb = geo.radius[j];

    auto distab = compute_distance_slab_sphere(c0, c1, c2, b, r0, r1, r2, rb);
    scalar dist = distab[0];

    bool hack = (j / per_mesh_verts) == (slab[0] / per_mesh_verts);
    bool refuse_cond = (slab[0] == j || slab[1] == j || slab[2] == j || hack);

    if (dist < 0.0f && !refuse_cond)
    {
        vec4i col(slab[0], slab[1], slab[2], j);
        ss_list.append(col, dist);
    }
}
__func__ void sphere_ground_set(
    MedialGeometry geo,
    SphereGroundCollisionList g_list,
    scalar ground_plane,
    int i)
{
    const vec3 &b = geo.vertices[i];
    scalar r = geo.radius[i];

    scalar d = b[1] - ground_plane;
    scalar dist = d - r;

    if (dist < 0.0)
    {
        vec2 hr(d, r);
        g_list.append(i, hr, dist * dist);
    }
}

__func__ void cone_cone_collision_set_static(
    MedialGeometry geo,
    MedialGeometry geo_static,
    ConeConeCollisionList cc_list,
    int i, int j)
{
    const vec2i &ex = geo.edges[i];
    vec3 c0 = geo.vertices[ex[0]];
    vec3 c1 = geo.vertices[ex[1]];
    scalar r0 = geo.radius[ex[0]];
    scalar r1 = geo.radius[ex[1]];

    const vec2i &ey = geo_static.edges[j];
    vec3 c2 = geo_static.vertices[ey[0]];
    vec3 c3 = geo_static.vertices[ey[1]];
    scalar r2 = geo_static.radius[ey[0]];
    scalar r3 = geo_static.radius[ey[1]];

    auto distab = compute_distance_cone_cone(c0, c1, c2, c3, r0, r1, r2, r3);
    auto dist = distab[0];
    if (dist < 0.0f)
    {
        vec4i col(ex[0], ex[1], ey[0], ey[1]);
        cc_list.append(col, dist);
    }
}

__func__ void cone_static_collision(
    MedialGeometry geo,
    TriangleSoup triangles_soup,
    ConeConeCollisionList collision_list,
    int ii,
    int2 *culled_pairs)
{
    int i = culled_pairs[ii].x;
    int j = culled_pairs[ii].y;

    const vec2i &ex = geo.edges[i];
    vec3 c0 = geo.vertices[ex[0]];
    vec3 c1 = geo.vertices[ex[1]];
    scalar r0 = geo.radius[ex[0]];
    scalar r1 = geo.radius[ex[1]];

    // for (int j = 0; j < n_edges_static; j++)
    {
        const vec2i &ey = triangles_soup.edges[j];
        vec3 c2 = triangles_soup.vertices[ey[0]];
        vec3 c3 = triangles_soup.vertices[ey[1]];
        scalar r2 = .0;
        scalar r3 = .0;

        auto distab = compute_distance_cone_cone(c0, c1, c2, c3, r0, r1, r2, r3);
        scalar dist = distab[0];
        if (dist < 0.0f)
        {
            vec4i col(ex[0], ex[1], ey[0], ey[1]);
            collision_list.append(col, dist);
        }
    }
}
__func__ void sphere_static_collision(
    MedialGeometry geo,
    TriangleSoup triangles_soup,
    SlabSphereCollisionList collision_list,
    const int *neighbors, // assuming neighbors is a flat array
    int ii,
    int2 *culled_pairs)
{

    int2 pi = culled_pairs[ii];
    int i = pi.x;
    int y = pi.y;

    vec3 xi = geo.vertices[i];
    scalar ri = geo.radius[i];

    vec3 low = xi - vec3(ri, ri, ri);
    vec3 high = xi + vec3(ri, ri, ri);

    // MeshQuery query = mesh_query_aabb(triangles_soup.mesh_id, low, high);

    // int y = 0;
    // while (mesh_query_aabb_next(query, y)) {
    // for (y = 0; y < n_triangles; y++)
    {
        vec3i tt = triangles_soup.indices[y];
        int t0 = tt[0];
        int t1 = tt[1];
        int t2 = tt[2];

        vec3 xt0 = triangles_soup.vertices[t0];
        vec3 xt1 = triangles_soup.vertices[t1];
        vec3 xt2 = triangles_soup.vertices[t2];

        scalar distance;
        int dist_type;
        point_triangle_distance_wp(xt0, xt1, xt2, xi, dist_type, distance);

        vec4i element(tt[0], tt[1], tt[2], i);
        if (distance < ri)
        {
            vec3 n = plane_normal_wp(xt0, xt1, xt2);
            if (n.dot(xi - xt0) < 0.0f)
            {
                distance = -distance;
            }

            scalar dmr = distance - ri;

            if (point_projects_inside_triangle(xt0, xt1, xt2, xi) ||
                inside_collision_cell(triangles_soup, neighbors, y, xi))
            {
                collision_list.append(element, dmr * dmr);
            }
        }
    }
}