#include "solver.h"
CUSolver::CUSolver(const Veci &outers, const Veci &indices, const Vec &values)
    : SolverBase(outers, indices, values)
{
    init();
}
CUSolver::CUSolver(const Eigen::SparseMatrix<double> &A)
    : SolverBase(A)
{
    init();
}
void CUSolver::init()
{
    cudssCreate(&handle);
    cudssConfigCreate(&solver_config);
    cudssDataCreate(handle, &solver_data);

    cudaMalloc((void **)&outers_d, (n + 1) * sizeof(int));
    cudaMalloc((void **)&indices_d, nnz * sizeof(int));
    cudaMalloc((void **)&values_d, nnz * sizeof(double));
    cudaMalloc((void **)&b_d, n * sizeof(double));
    cudaMalloc((void **)&x_d, n * sizeof(double));

    cudaMemcpy(outers_d, outers.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(indices_d, indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(values_d, values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice);

    int64_t nrows = n, ncols = n;
    int ldb = ncols, ldx = nrows;
    int nrhs = 1;
    cudssMatrixCreateDn(&b, ncols, nrhs, ldb, b_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    cudssMatrixCreateDn(&x, nrows, nrhs, ldx, x_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);

    /* Create a matrix object for the sparse input matrix. */
    cudssMatrixType_t mtype = CUDSS_MTYPE_SPD;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t base = CUDSS_BASE_ZERO;
    cudssMatrixCreateCsr(&A, nrows, ncols, nnz, outers_d, NULL, indices_d, values_d, CUDA_R_32I, CUDA_R_64F, mtype, mview, base);
}

void CUSolver::factorize()
{
    assert(stage >= 1);
    /* Factorization */
    cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solver_config, solver_data, A, x, b);
    stage = 2;
}

void CUSolver::refactorize(const Vec &new_values) {
    assert(stage >= 1);
    cudaMemcpy(values_d, new_values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice);

    cudssExecute(handle, CUDSS_PHASE_REFACTORIZATION, solver_config, solver_data, A, x, b);
    stage = 2;
}

void CUSolver::refactorize_dev(uintptr_t new_values_ptr) {
    assert(stage >= 1);
    cudaMemcpy(values_d, reinterpret_cast<double*>(new_values_ptr), nnz * sizeof(double), cudaMemcpyDeviceToDevice);

    cudssExecute(handle, CUDSS_PHASE_REFACTORIZATION, solver_config, solver_data, A, x, b);
    stage = 2;
}

Vec CUSolver::solve(const Vec &bb) const
{
    Vec xe;
    xe.resize(n);
    assert(stage == 2);
    int nrhs = 1;

    cudaMemcpy(b_d, bb.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    /* Solving */
    cudssExecute(handle, CUDSS_PHASE_SOLVE, solver_config, solver_data, A, x, b);
    cudaMemcpy(xe.data(), x_d, nrhs * n * sizeof(double), cudaMemcpyDeviceToHost);
    return xe;
}

void CUSolver::solve_dev(uintptr_t b_ptr, uintptr_t x_ptr) const
{
    assert(stage == 2);
    int nrhs = 1;

    cudaMemcpy(b_d, reinterpret_cast<double*>(b_ptr), n * sizeof(double), cudaMemcpyDeviceToDevice);
    /* Solving */
    cudssExecute(handle, CUDSS_PHASE_SOLVE, solver_config, solver_data, A, x, b);
    cudaMemcpy(reinterpret_cast<double*>(x_ptr), x_d, nrhs * n * sizeof(double), cudaMemcpyDeviceToDevice);
}

void CUSolver::analyze_pattern()
{
    cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solver_config, solver_data, A, x, b);
    stage = 1;
}

CUSolverDevice::CUSolverDevice(uintptr_t outers_ptr, uintptr_t indices_ptr,
                               uintptr_t values_ptr, int n, int nnz)
    : n(n), nnz(nnz)
{
    if (n <= 0 || nnz <= 0 || outers_ptr == 0 || indices_ptr == 0 || values_ptr == 0)
        throw std::invalid_argument("CUSolverDevice requires non-null device pointers and positive n and nnz");
    init(outers_ptr, indices_ptr, values_ptr);
}

void CUSolverDevice::init(uintptr_t outers_ptr, uintptr_t indices_ptr, uintptr_t values_ptr)
{
    cudssCreate(&handle);
    cudssConfigCreate(&solver_config);
    cudssDataCreate(handle, &solver_data);

    cudaMalloc((void **)&outers_d, (n + 1) * sizeof(int));
    cudaMalloc((void **)&indices_d, nnz * sizeof(int));
    cudaMalloc((void **)&values_d, nnz * sizeof(double));
    cudaMalloc((void **)&b_d, n * sizeof(double));
    cudaMalloc((void **)&x_d, n * sizeof(double));

    cudaMemcpy(outers_d, reinterpret_cast<const int *>(outers_ptr),
               (n + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(indices_d, reinterpret_cast<const int *>(indices_ptr),
               nnz * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(values_d, reinterpret_cast<const double *>(values_ptr),
               nnz * sizeof(double), cudaMemcpyDeviceToDevice);

    const int64_t size = n;
    cudssMatrixCreateDn(&b, size, 1, size, b_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    cudssMatrixCreateDn(&x, size, 1, size, x_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    cudssMatrixCreateCsr(&A, size, size, nnz, outers_d, nullptr, indices_d,
                         values_d, CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_SPD,
                         CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO);
}

void CUSolverDevice::analyze_pattern()
{
    cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solver_config, solver_data, A, x, b);
    stage = 1;
}

void CUSolverDevice::factorize()
{
    if (stage < 1)
        throw std::logic_error("analyze_pattern() must be called before factorize()");
    cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solver_config, solver_data, A, x, b);
    stage = 2;
}

void CUSolverDevice::refactorize(uintptr_t new_values_ptr)
{
    if (stage < 1)
        throw std::logic_error("analyze_pattern() must be called before refactorize()");
    if (new_values_ptr == 0)
        throw std::invalid_argument("new_values_ptr must be a non-null CUDA device pointer");

    cudaMemcpy(values_d, reinterpret_cast<const double *>(new_values_ptr),
               nnz * sizeof(double), cudaMemcpyDeviceToDevice);
    cudssExecute(handle, CUDSS_PHASE_REFACTORIZATION, solver_config, solver_data, A, x, b);
    stage = 2;
}

void CUSolverDevice::solve(uintptr_t b_ptr, uintptr_t x_ptr) const
{
    if (stage != 2)
        throw std::logic_error("factorize() or refactorize() must be called before solve()");
    if (b_ptr == 0 || x_ptr == 0)
        throw std::invalid_argument("b_ptr and x_ptr must be non-null CUDA device pointers");

    cudaMemcpy(b_d, reinterpret_cast<const double *>(b_ptr),
               n * sizeof(double), cudaMemcpyDeviceToDevice);
    cudssExecute(handle, CUDSS_PHASE_SOLVE, solver_config, solver_data, A, x, b);
    cudaMemcpy(reinterpret_cast<double *>(x_ptr), x_d,
               n * sizeof(double), cudaMemcpyDeviceToDevice);
}

CUSolverDevice::~CUSolverDevice()
{
    if (A) cudssMatrixDestroy(A);
    if (b) cudssMatrixDestroy(b);
    if (x) cudssMatrixDestroy(x);
    if (outers_d) cudaFree(outers_d);
    if (indices_d) cudaFree(indices_d);
    if (values_d) cudaFree(values_d);
    if (b_d) cudaFree(b_d);
    if (x_d) cudaFree(x_d);
    if (solver_data) cudssDataDestroy(handle, solver_data);
    if (solver_config) cudssConfigDestroy(solver_config);
    if (handle) cudssDestroy(handle);
}

CUSolverDeviceF::CUSolverDeviceF(uintptr_t outers_ptr, uintptr_t indices_ptr,
                                 uintptr_t values_ptr, int n, int nnz)
    : n(n), nnz(nnz)
{
    if (n <= 0 || nnz <= 0 || outers_ptr == 0 || indices_ptr == 0 || values_ptr == 0)
        throw std::invalid_argument("CUSolverDeviceF requires non-null device pointers and positive n and nnz");
    init(outers_ptr, indices_ptr, values_ptr);
}

void CUSolverDeviceF::init(uintptr_t outers_ptr, uintptr_t indices_ptr, uintptr_t values_ptr)
{
    cudssCreate(&handle);
    cudssConfigCreate(&solver_config);
    cudssDataCreate(handle, &solver_data);

    cudaMalloc((void **)&outers_d, (n + 1) * sizeof(int));
    cudaMalloc((void **)&indices_d, nnz * sizeof(int));
    cudaMalloc((void **)&values_d, nnz * sizeof(float));
    cudaMalloc((void **)&b_d, n * sizeof(float));
    cudaMalloc((void **)&x_d, n * sizeof(float));

    cudaMemcpy(outers_d, reinterpret_cast<const int *>(outers_ptr),
               (n + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(indices_d, reinterpret_cast<const int *>(indices_ptr),
               nnz * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(values_d, reinterpret_cast<const float *>(values_ptr),
               nnz * sizeof(float), cudaMemcpyDeviceToDevice);

    const int64_t size = n;
    cudssMatrixCreateDn(&b, size, 1, size, b_d, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR);
    cudssMatrixCreateDn(&x, size, 1, size, x_d, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR);
    cudssMatrixCreateCsr(&A, size, size, nnz, outers_d, nullptr, indices_d,
                         values_d, CUDA_R_32I, CUDA_R_32F, CUDSS_MTYPE_SPD,
                         CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO);
}

void CUSolverDeviceF::analyze_pattern()
{
    cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solver_config, solver_data, A, x, b);
    stage = 1;
}

void CUSolverDeviceF::factorize()
{
    if (stage < 1)
        throw std::logic_error("analyze_pattern() must be called before factorize()");
    cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solver_config, solver_data, A, x, b);
    stage = 2;
}

void CUSolverDeviceF::refactorize(uintptr_t new_values_ptr)
{
    if (stage < 1)
        throw std::logic_error("analyze_pattern() must be called before refactorize()");
    if (new_values_ptr == 0)
        throw std::invalid_argument("new_values_ptr must be a non-null CUDA device pointer");

    cudaMemcpy(values_d, reinterpret_cast<const float *>(new_values_ptr),
               nnz * sizeof(float), cudaMemcpyDeviceToDevice);
    cudssExecute(handle, CUDSS_PHASE_REFACTORIZATION, solver_config, solver_data, A, x, b);
    stage = 2;
}

void CUSolverDeviceF::solve(uintptr_t b_ptr, uintptr_t x_ptr) const
{
    if (stage != 2)
        throw std::logic_error("factorize() or refactorize() must be called before solve()");
    if (b_ptr == 0 || x_ptr == 0)
        throw std::invalid_argument("b_ptr and x_ptr must be non-null CUDA device pointers");

    cudaMemcpy(b_d, reinterpret_cast<const float *>(b_ptr),
               n * sizeof(float), cudaMemcpyDeviceToDevice);
    cudssExecute(handle, CUDSS_PHASE_SOLVE, solver_config, solver_data, A, x, b);
    cudaMemcpy(reinterpret_cast<float *>(x_ptr), x_d,
               n * sizeof(float), cudaMemcpyDeviceToDevice);
}

CUSolverDeviceF::~CUSolverDeviceF()
{
    if (A) cudssMatrixDestroy(A);
    if (b) cudssMatrixDestroy(b);
    if (x) cudssMatrixDestroy(x);
    if (outers_d) cudaFree(outers_d);
    if (indices_d) cudaFree(indices_d);
    if (values_d) cudaFree(values_d);
    if (b_d) cudaFree(b_d);
    if (x_d) cudaFree(x_d);
    if (solver_data) cudssDataDestroy(handle, solver_data);
    if (solver_config) cudssConfigDestroy(solver_config);
    if (handle) cudssDestroy(handle);
}
