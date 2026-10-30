#pragma once
#include "common.cuh"
#include <set>
#include <map>
#include <array>
#include <vector>

namespace sparse_utils
{
    template <int N>
    inline int starting_offset(int i, int j, const std::map<std::array<int, 2>, int> &lut, int *outers)
    {
        auto it = lut.find({i, j});
        int k = it->second;
        return k * N + outers[j * N];
    }

    template <int N>
    inline int stride(int j, int *outers)
    {
        return outers[j * N + 1] - outers[j * N];
        // full 12x12 matrix, no overflow issue
    }

    template <int N>
    inline void put(scalar *values, int offset, int _stride, const Eigen::Matrix<scalar, N, N> &block)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int ofs = offset + _stride * j + i;
                values[ofs] += block(i, j);
            }
        }
    }

    inline void putNM(int N, int M, scalar *values, int offset, int _stride, const Eigen::Matrix<scalar, -1, -1> &block)
    {   
        int MN = M * N;
        #pragma omp parallel for 
        for (int ii = 0; ii < MN; ii++){
            int i = ii / M; 
            int j = ii % M;
            int ofs = offset + _stride * j + i;
            values[ofs] += block(i, j);
        }
        // for (int i = 0; i < N; i++)
        // {
        //     for (int j = 0; j < M; j++)
        //     {
        //         int ofs = offset + _stride * j + i;
        //         values[ofs] += block(i, j);
        //     }
        // }
    }


    template <int N>
    void fill_sm(const std::vector<int> &rows, const std::vector<int> &cols,
                 const std::vector<Eigen::Matrix<scalar, N, N>> &vals,
                 Eigen::SparseMatrix<scalar> &sparse_hess, const std::map<std::array<int, 2>, int> &lut)
    {
        auto outers = sparse_hess.outerIndexPtr();
        auto values = sparse_hess.valuePtr();
        for (int i = 0; i < vals.size(); i++)
        {
            int ii = rows[i];
            int jj = cols[i];

            int oij = starting_offset<N>(ii, jj, lut, outers);
            int stride_j = stride<N>(jj, outers);
            put<N>(values, oij, stride_j, vals[i]);
        }
    }
    template <int N>
    void fill_sm_parallel(const std::vector<int> &rows, const std::vector<int> &cols,
                          const std::vector<Eigen::Matrix<scalar, N, N>> &vals,
                          Eigen::SparseMatrix<scalar> &sparse_hess, const std::map<std::array<int, 2>, int> &lut)
    {
        auto outers = sparse_hess.outerIndexPtr();
        auto values = sparse_hess.valuePtr();
#pragma omp parallel for
        for (int i = 0; i < vals.size(); i++)
        {
            int ii = rows[i];
            int jj = cols[i];

            int oij = starting_offset<N>(ii, jj, lut, outers);
            int stride_j = stride<N>(jj, outers);
            put<N>(values, oij, stride_j, vals[i]);
        }
    }
    template <int N>
    void gen_empty_sm(
        const std::vector<int> &rows,
        const std::vector<int> &cols,
        const std::vector<Eigen::Matrix<scalar, N, N>> &vals,
        Eigen::SparseMatrix<scalar> &sparse_hess, std::map<std::array<int, 2>, int> &lut,
        bool parallel = false)
    {
        if (rows.size() == 0)
            return;
        std::set<std::array<int, 2>> cset;
        // std::map<std::array<int, 2>, int> lut;
        for (int i = 0; i < rows.size(); i++)
        {
            cset.insert({cols[i], rows[i]});
        }

        auto old = cset.begin();
        auto old_col = (*old)[0];
        for (int i = 0; i < old_col; i++)
        {
            for (int c = 0; c < N; c++)
                sparse_hess.startVec(i * N + c);
        }
        for (auto it = cset.begin();; it++)
        {
            if (it == cset.end() || (*it)[0] != old_col)
            {
                for (int c = 0; c < N; c++)
                {

                    auto cc = c + old_col * N;
                    sparse_hess.startVec(cc);
                    int k = 0;
                    for (auto kt = old; kt != it; ++kt)
                    {
                        lut[{(*kt)[1], (*kt)[0]}] = k++;
                        auto row = (*kt)[1];
                        for (int r = 0; r < N; r++)
                        {
                            auto rr = row * N + r;
                            sparse_hess.insertBack(rr, cc) = 0.0;
                        }
                    }
                }
                if (it == cset.end())
                    break;

                while (++old_col < (*it)[0])
                {
                    for (int c = 0; c < N; c++)
                        sparse_hess.startVec(c + old_col * N);
                }
                old = it;
                old_col = (*it)[0];
            }
        }
        sparse_hess.makeCompressed();
        // spdlog::info("\nsparse matrix : #non-zero blocks = {}", cset.size());
        // cout << sparse_hess;
        if (parallel)
            fill_sm_parallel<N>(rows, cols, vals, sparse_hess, lut);
        else
            fill_sm<N>(rows, cols, vals, sparse_hess, lut);
    }

    template <int N>
    void gen_empty_sm(
        const std::vector<int> &rows,
        const std::vector<int> &cols,
        const std::vector<Eigen::Matrix<scalar, N, N>> &vals,
        Eigen::SparseMatrix<scalar> &sparse_hess,
        bool parallel = false)
    {
        std::map<std::array<int, 2>, int> lut;
        gen_empty_sm<N>(rows, cols, vals, sparse_hess, lut, parallel);
    }

    template <typename Scalar>
    void hstack(const Eigen::SparseMatrix<Scalar, Eigen::ColMajor> &A,
                const Eigen::SparseMatrix<Scalar, Eigen::ColMajor> &B,
                Eigen::SparseMatrix<Scalar, Eigen::ColMajor> &C)
    {
        assert(A.rows() == B.rows() && "Number of rows must match");

        const int m = A.rows();
        const int nA = A.cols(), nB = B.cols();
        const int nnzA = A.nonZeros();
        const int nnzB = B.nonZeros();

        C.resize(m, nA + nB);
        C.reserve(nnzA + nnzB);

        // Copy values
        std::copy(A.valuePtr(), A.valuePtr() + nnzA, C.valuePtr());
        std::copy(B.valuePtr(), B.valuePtr() + nnzB, C.valuePtr() + nnzA);

        // Copy inner indices
        std::copy(A.innerIndexPtr(), A.innerIndexPtr() + nnzA, C.innerIndexPtr());
        std::copy(B.innerIndexPtr(), B.innerIndexPtr() + nnzB, C.innerIndexPtr() + nnzA);

        // Construct outer index (column pointers)
        auto *outer = C.outerIndexPtr();
        const auto *outerA = A.outerIndexPtr();
        const auto *outerB = B.outerIndexPtr();

        for (int i = 0; i <= nA; ++i)
        {
            outer[i] = outerA[i];
        }
        for (int i = 1; i <= nB; ++i)
        {
            outer[nA + i] = nnzA + outerB[i];
        }
    }

    template <typename Scalar>
    void block_diag(const Eigen::SparseMatrix<Scalar, Eigen::ColMajor> &A,
                    const Eigen::SparseMatrix<Scalar, Eigen::ColMajor> &B,
                    Eigen::SparseMatrix<Scalar, Eigen::ColMajor> &C)
    {
        const int mA = A.rows(), mB = B.rows();
        const int nA = A.cols(), nB = B.cols();
        const int nnzA = A.nonZeros();
        const int nnzB = B.nonZeros();

        C.resize(mA + mB, nA + nB);
        C.reserve(nnzA + nnzB);

        // Copy values
        std::copy(A.valuePtr(), A.valuePtr() + nnzA, C.valuePtr());
        std::copy(B.valuePtr(), B.valuePtr() + nnzB, C.valuePtr() + nnzA);

        // Copy inner indices
        std::copy(A.innerIndexPtr(), A.innerIndexPtr() + nnzA, C.innerIndexPtr());

        // std::copy(B.innerIndexPtr(), B.innerIndexPtr() + nnzB, C.innerIndexPtr() + nnzA);

        auto *inner = C.innerIndexPtr();
        auto *innerB = B.innerIndexPtr();
        #pragma omp parallel for
        for (int i = 0; i < nnzB; i++)
        {
            inner[i + nnzA] = innerB[i] + nA;
        }

        // Construct outer index (column pointers)
        auto *outer = C.outerIndexPtr();
        const auto *outerA = A.outerIndexPtr();
        const auto *outerB = B.outerIndexPtr();
        #pragma omp parallel for
        for (int i = 0; i <= nA; ++i)
        {
            outer[i] = outerA[i];
        }
        #pragma omp parallel for
        for (int i = 1; i <= nB; ++i)
        {
            outer[nA + i] = nnzA + outerB[i];
        }
    }

    template <typename Scalar>
    void select_rows_csr_direct(const Eigen::SparseMatrix<Scalar, Eigen::RowMajor> &A,
                                const std::vector<int> &row_indices,
                                Eigen::SparseMatrix<Scalar, Eigen::RowMajor> &B)
    {
        assert(A.isCompressed());

        int out_rows = row_indices.size();
        int cols = A.cols();

        // Estimate nnz
        int nnz = 0;
        for (int row : row_indices)
        {
            nnz += A.outerIndexPtr()[row + 1] - A.outerIndexPtr()[row];
        }

        B.resize(out_rows, cols);
        B.reserve(nnz);

        for (int i = 0; i < row_indices.size(); ++i)
        {
            int src_row = row_indices[i];
            B.startVec(i);
            for (typename Eigen::SparseMatrix<Scalar, Eigen::RowMajor>::InnerIterator it(A, src_row); it; ++it)
            {
                B.insertBack(i, it.col()) = it.value();
            }
        }

        B.finalize();
    }
};
