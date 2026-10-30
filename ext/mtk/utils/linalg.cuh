#pragma once
#include "common.cuh"

__func__ vec3 cross(const vec3 &a, const vec3 &b)
{
    vec3 ret;
    ret << a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0];
    return ret;
}
__func__ scalar determinant(const mat33 &m)
{
    vec3 a, b, c;
    a << m(0, 0), m(0, 1), m(0, 2);
    b << m(1, 0), m(1, 1), m(1, 2);
    c << m(2, 0), m(2, 1), m(2, 2);

    return a.dot(cross(b, c));
}

__func__ mat33 inverse(const mat33 &m)
{
    scalar det = determinant(m);

    if (det != scalar(0.0f))
    {
        mat33 b;

        b(0, 0) = m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1);
        b(1, 0) = m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2);
        b(2, 0) = m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0);

        b(0, 1) = m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2);
        b(1, 1) = m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0);
        b(2, 1) = m(0, 1) * m(2, 0) - m(0, 0) * m(2, 1);

        b(0, 2) = m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1);
        b(1, 2) = m(0, 2) * m(1, 0) - m(0, 0) * m(1, 2);
        b(2, 2) = m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);

        return b * 1.0 / det;
    }
    else
    {
        return mat33::Zero();
    }
}

__func__ mat22 inverse(const mat22 &A)
{
    scalar det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);

    // You might want to add a small epsilon check here for robustness
    if (abs(det) < 1e-12)
    {
        // Return identity or zero matrix or assert depending on your needs
        return mat22::Zero(); // or mat22::Identity();
    }

    scalar inv_det = 1.0 / det;

    mat22 Ainv;
    Ainv(0, 0) = A(1, 1) * inv_det;
    Ainv(0, 1) = -A(0, 1) * inv_det;
    Ainv(1, 0) = -A(1, 0) * inv_det;
    Ainv(1, 1) = A(0, 0) * inv_det;

    return Ainv;
}