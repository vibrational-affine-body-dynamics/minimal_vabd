#pragma once
#include "buffer.h"
#include "cnpy/cnpy.h"
#include <Eigen/Sparse>
inline Eigen::Matrix<scalar, -1, 3> to_eigen(const Shay::VtBuffer<vec3> &buf)
{
    Eigen::Matrix<scalar, -1, 3> ret;
    auto host = buf.get();
    ret.resize(host.size(), 3);
    for (int i = 0; i < host.size(); i++)
    {
        ret.row(i) = host[i];
    }
    return ret;
}
inline std::vector<vec3> to_host_vector(const Eigen::Matrix<scalar, -1, 3> &m)
{
    std::vector<vec3> ret;
    ret.resize(m.rows());
    for (int i = 0; i < ret.size(); i++)
    {
        ret[i] = m.row(i);
    }
    return ret;
}

inline void from_eigen(Shay::VtBuffer<vec3> &buf, const Eigen::Matrix<scalar, -1, 3> &m)
{
    auto host = to_host_vector(m);
    buf.copy(host);
}

inline void from_eigen(std::vector<vec3> &buf, const Eigen::Matrix<scalar, -1, 3> &m)
{
    auto host = to_host_vector(m);
    cudaMemcpy(buf.data(), host.data(), sizeof(vec3) * host.size(), cudaMemcpyDefault);
}

// inline void assign(wp_array<vec3> &arr, const Eigen::Matrix<scalar, -1 , 3> &m) {
//     arr = to_host_vector(m);
// }

template <typename Derived>
inline void WpArray(const Eigen::MatrixBase<Derived> &mat, wp_array<Eigen::Matrix<typename Derived::Scalar, Derived::ColsAtCompileTime, 1>> &ret)
{
    using Scalar = typename Derived::Scalar;
    constexpr int cols = Derived::ColsAtCompileTime;

    static_assert(cols != Eigen::Dynamic, "Matrix must have a fixed number of columns.");

    using VecType = Eigen::Matrix<Scalar, cols, 1>;
    std::vector<VecType> result;
    result.reserve(mat.rows());
    for (int i = 0; i < mat.rows(); ++i)
    {
        result.emplace_back(mat.row(i).transpose());
    }
#ifdef __CUDARUN__
    ret.copy(result);
#else
    ret = result;
#endif
}

inline Mat arr_to_mat(cnpy::NpyArray &W)
{
    int r = W.shape[0], c = 1;
    if (W.shape.size() > 1) c = W.shape[1];
    Mat W_medial;
    if (W.word_size == 4)
    {
        float *d = W.data<float>();
        if (W.fortran_order)
        {
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat(d, r, c);
            W_medial = mat.cast<scalar>();
        }
        else
        {

            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(d, r, c);
            W_medial = mat.cast<scalar>();
        }
    }
    else
    {
        double *d = W.data<double>();
        if (W.fortran_order)
        {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat(d, r, c);
            // W_medial = mat.cast<scalar>();
            W_medial = mat.cast<scalar>();
        }
        else
        {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(d, r, c);
            // W_medial = mat.cast<scalar>();
            W_medial = mat.cast<scalar>();
        }
    }
    return W_medial;
}

inline Mat load_npy(const std::string &file)
{
    auto W = cnpy::npy_load(file);
    return arr_to_mat(W);
}

inline void save_sparse(const Eigen::SparseMatrix<scalar> &Um, const std::string &name)
{
    auto *t = Um.valuePtr();
    uint nnz = Um.nonZeros();
    const int *inner = Um.innerIndexPtr();
    const int *outer = Um.outerIndexPtr();
    cnpy::npy_save<int>(name + "_outer.npy", outer, {uint(Um.cols())}, "w");
    cnpy::npy_save<int>(name + "_indices.npy", inner, {uint(nnz)}, "w");
    cnpy::npy_save<scalar>(name + "_values.npy", t, {uint(nnz)}, "w");
}

inline void save_vector(const Vec &b, const std::string &name)
{
    cnpy::npy_save<scalar>(name, &b(0), {static_cast<unsigned long long>(b.size())}, "w");
}
inline void save_dense(const Mat &A, const std::string &name)
{
    Eigen::Matrix<scalar, -1, -1, Eigen::RowMajor> A_row_major = A;
    cnpy::npy_save<scalar>(name, &A_row_major(0, 0), {uint(A.rows()), uint(A.cols())}, "w");
}