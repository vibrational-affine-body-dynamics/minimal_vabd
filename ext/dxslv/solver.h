#pragma once
#include <cstdint>
#include <stdexcept>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "cudss.h"

using Vec = Eigen::VectorXd;
using Veci = Eigen::VectorXi;

struct SolverBase
{
    Veci outers;
    Veci indices;
    Vec values;
    int nnz, n;
    SolverBase(const Veci &outers, const Veci &indices, const Vec &values)
        : outers(outers), indices(indices), values(values), n(outers.rows() - 1), nnz(indices.rows()) {}
    SolverBase(const Eigen::SparseMatrix<double> &A) : n(A.outerSize()), nnz(A.nonZeros())
    {
        const int *outer_ptr = A.outerIndexPtr();
        const int *indices_ptr = A.innerIndexPtr();
        const double *values_ptr = A.valuePtr();

        outers.resize(A.outerSize() + 1);
        indices.resize(A.nonZeros());
        values.resize(A.nonZeros());

        memcpy(outers.data(), outer_ptr, (A.outerSize() + 1) * sizeof(int));
        memcpy(indices.data(), indices_ptr, A.nonZeros() * sizeof(int));
        memcpy(values.data(), values_ptr, A.nonZeros() * sizeof(double));
    }
    virtual void analyze_pattern() = 0;
    virtual void factorize() = 0;
    virtual void refactorize(const Vec &new_values) = 0;
    virtual Vec solve(const Vec &b) const = 0;
};

struct CUSolver : SolverBase
{
    CUSolver(const Veci &outers, const Veci &indices, const Vec &values);
    CUSolver(const Eigen::SparseMatrix<double> &A);

    cudssHandle_t handle;
    cudssConfig_t solver_config;
    cudssData_t solver_data;
    cudssMatrix_t A, x, b;

    void analyze_pattern() override;
    void factorize() override;
    void refactorize(const Vec &new_values) override;
    void refactorize_dev(uintptr_t new_values_ptr);
    Vec solve(const Vec &b) const override;
    void solve_dev(uintptr_t b_ptr, uintptr_t x_ptr) const; 
    int stage = 0; // 0: not analyzed, 1: analyzed, 2: factorized

    void init();
    ~CUSolver()
    {
        cudssMatrixDestroy(A);
        cudssMatrixDestroy(b);
        cudssMatrixDestroy(x);
        cudaFree(outers_d);
        cudaFree(indices_d);
        cudaFree(values_d);
        cudaFree(b_d);
        cudaFree(x_d);
        cudssDataDestroy(handle, solver_data);
        cudssConfigDestroy(solver_config);
        cudssDestroy(handle);
    }
    int *outers_d, *indices_d;
    double *values_d, *b_d, *x_d;
};

// A device-only counterpart to CUSolver.  The constructor and numerical
// operations accept CUDA device pointers, so no host Eigen arrays are needed.
struct CUSolverDevice
{
    CUSolverDevice(uintptr_t outers_ptr, uintptr_t indices_ptr,
                   uintptr_t values_ptr, int n, int nnz);

    void analyze_pattern();
    void factorize();
    void refactorize(uintptr_t new_values_ptr);
    void solve(uintptr_t b_ptr, uintptr_t x_ptr) const;

    ~CUSolverDevice();

    CUSolverDevice(const CUSolverDevice &) = delete;
    CUSolverDevice &operator=(const CUSolverDevice &) = delete;

private:
    void init(uintptr_t outers_ptr, uintptr_t indices_ptr, uintptr_t values_ptr);

    int n, nnz;
    int stage = 0; // 0: not analyzed, 1: analyzed, 2: factorized
    cudssHandle_t handle = nullptr;
    cudssConfig_t solver_config = nullptr;
    cudssData_t solver_data = nullptr;
    cudssMatrix_t A = nullptr, x = nullptr, b = nullptr;
    int *outers_d = nullptr, *indices_d = nullptr;
    double *values_d = nullptr, *b_d = nullptr, *x_d = nullptr;
};

// Float32 device-only counterpart to CUSolverDevice. All matrix values and
// right-hand-side/solution buffers are expected to be CUDA float pointers.
struct CUSolverDeviceF
{
    CUSolverDeviceF(uintptr_t outers_ptr, uintptr_t indices_ptr,
                    uintptr_t values_ptr, int n, int nnz);

    void analyze_pattern();
    void factorize();
    void refactorize(uintptr_t new_values_ptr);
    void solve(uintptr_t b_ptr, uintptr_t x_ptr) const;

    ~CUSolverDeviceF();

    CUSolverDeviceF(const CUSolverDeviceF &) = delete;
    CUSolverDeviceF &operator=(const CUSolverDeviceF &) = delete;

private:
    void init(uintptr_t outers_ptr, uintptr_t indices_ptr, uintptr_t values_ptr);

    int n, nnz;
    int stage = 0; // 0: not analyzed, 1: analyzed, 2: factorized
    cudssHandle_t handle = nullptr;
    cudssConfig_t solver_config = nullptr;
    cudssData_t solver_data = nullptr;
    cudssMatrix_t A = nullptr, x = nullptr, b = nullptr;
    int *outers_d = nullptr, *indices_d = nullptr;
    float *values_d = nullptr, *b_d = nullptr, *x_d = nullptr;
};
