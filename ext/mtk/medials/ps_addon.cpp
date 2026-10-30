#include "medials/ps_addon.h"
#include "utils/linalg.cuh"
using namespace std;
using namespace Eigen;
static const int n_segments = 10, n_arcs = 10;

void get_frames(const vec3 &v0, const vec3 &v1, scalar r0, scalar r1, vec3 &fx, vec3 &fy)
{
    // fx cross fy = v1 - v0 normlized
    static const vec3 up(0.0, 1.0, 0.0);
    scalar sin_phi = (r1 - r0) / (v1 - v0).norm();
    scalar c = sqrt(1 - sin_phi * sin_phi);
    vec3 axis = (v1 - v0).normalized();
    // auto xx = axis.cross(up);
    auto xx = cross(axis, up);
    if (xx.squaredNorm() > 1e-3)
    {
    }
    else
    {
        // xx = axis.cross(vec3(1.0, 0.0, 0.0));
        xx = cross(axis, vec3(1.0, 0.0, 0.0));
    }
    fx = xx.normalized();
    // fy = axis.cross(fx);
    fy = cross(axis, fx);

    // fx = fx * c + sin_phi * axis;
    // fy = fy * c + sin_phi * axis;
}

inline vec3 position(scalar seg, scalar theta, const vec3 &v0, const vec3 &v1, scalar r0, scalar r1, const vec3 &fx, const vec3 &fy)
{
    vec3 axis = (v1 - v0).normalized();
    vec3 aa = seg * v1 + (1 - seg) * v0;
    scalar rr = seg * r1 + (1 - seg) * r0;
    scalar sin_phi = (r1 - r0) / (v1 - v0).norm();
    scalar c = sqrt(1 - sin_phi * sin_phi);
    return aa + rr * ((cos(theta) * fx + sin(theta) * fy) * c - axis * sin_phi);
    // return aa + rr * (cos(theta) * fx + sin(theta) * fy);
}

inline int idx(int i, int j)
{
    i = i % n_arcs;
    j = j % n_segments;
    return i + n_arcs * j;
}

inline vec3i gen_lower(int i, int j)
{
    vec3i ret;
    ret << idx(i, j), idx(i + 1, j), idx(i, j + 1);
    return ret;
}
inline vec3i gen_upper(int i, int j)
{
    vec3i ret;
    ret << idx(i, j + 1), idx(i + 1, j), idx(i + 1, j + 1);
    return ret;
}
vector<vec3i> gen_indices_default()
{
    vector<vec3i> ret;
    for (int j = 0; j < n_segments - 1; j++)
    {
        for (int i = 0; i < n_arcs; i++)
        {
            ret.push_back(gen_lower(i, j));
            ret.push_back(gen_upper(i, j));
        }
    }
    return ret;
}
vector<vec3i> gen_indices(int cnt)
{
    static const vector<vec3i> indices_default = gen_indices_default();
    vector<vec3i> indices = indices_default;
    vec3i offset;
    offset << cnt, cnt, cnt;
    for (auto &e : indices)
    {
        e += offset;
    }
    return indices;
}

inline void points2mat(const vector<vec3> &points, Mat &V)
{
    V.conservativeResize(points.size(), 3);
    for (int i = 0; i < points.size(); i++)
    {
        V.row(i) = points[i];
    }
}

inline void faces2mat(const vector<vec3i> &faces, Eigen::MatrixXi &F)
{
    F.conservativeResize(faces.size(), 3);
    for (int i = 0; i < faces.size(); i++)
    {
        F.row(i) = faces[i];
    }
}

void cone_points(const Mat &V, const Vec &R, const Eigen::MatrixXi &E, Mat &V_ret, Eigen::MatrixXi &F_ret)
{
    vector<vec3> points;
    vector<vec3i> faces;
    for (int ee = 0; ee < E.rows(); ee++)
    {
        int e0 = E(ee, 0), e1 = E(ee, 1);
        // vec3 v0 = V.row(e0), v1 = V.row(e1);
        vec3 v0 = V.block<1, 3>(e0, 0), v1 = V.block<1, 3>(e1, 0);
        scalar r0 = R(e0), r1 = R(e1);

        vec3 x_frame, y_frame;
        get_frames(v0, v1, r0, r1, x_frame, y_frame);
        for (int ss = 0; ss < n_segments; ss++)
        {
            for (int tt = 0; tt < n_arcs; tt++)
            {
                scalar theta = 2 * M_PI * (scalar)tt / n_arcs;
                scalar seg = (scalar)ss / (n_segments - 1);
                auto p = position(seg, theta, v0, v1, r0, r1, x_frame, y_frame);
                points.push_back(p);
            }
        }
        int cnt = ee * n_segments * n_arcs;
        auto ff = gen_indices(cnt);
        faces.insert(faces.end(), ff.begin(), ff.end());
    }

    points2mat(points, V_ret);
    faces2mat(faces, F_ret);
}

void update_cone_points(const Mat &V, const Vec &R, const Eigen::MatrixXi &E, Mat &V_ret)
{
    vector<vec3> points;
    for (int ee = 0; ee < E.rows(); ee++)
    {
        int e0 = E(ee, 0), e1 = E(ee, 1);
        vec3 v0 = V.row(e0), v1 = V.row(e1);
        scalar r0 = R(e0), r1 = R(e1);

        vec3 x_frame, y_frame;
        get_frames(v0, v1, r0, r1, x_frame, y_frame);
        for (int ss = 0; ss < n_segments; ss++)
        {
            for (int tt = 0; tt < n_arcs; tt++)
            {
                scalar theta = 2 * M_PI * (scalar)tt / n_arcs;
                scalar seg = (scalar)ss / (n_segments - 1);
                auto p = position(seg, theta, v0, v1, r0, r1, x_frame, y_frame);
                points.push_back(p);
            }
        }
    }
    points2mat(points, V_ret);
}