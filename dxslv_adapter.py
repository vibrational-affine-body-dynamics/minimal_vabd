"""Validated cuDSS solves through the optional dxslv CUDA-pointer wrapper."""

import warnings

import numpy as np
import warp as wp
from scipy.sparse import csr_matrix, triu
from scipy.sparse.linalg import spsolve

try:
    from dxslv import CUSolverDevice
except ImportError:  # The open-source runtime remains usable without cuDSS.
    CUSolverDevice = None


class DxslvSolver:
    """Cache cuDSS analysis while a symmetric sparsity pattern is unchanged."""

    def __init__(self, device="cuda:0", residual_tolerance=1.0e-7):
        self.device = device
        self.residual_tolerance = float(residual_tolerance)
        self.available = CUSolverDevice is not None and wp.is_cuda_available()
        self.solver = None
        self.pattern = None
        self.outers = None
        self.indices = None
        self.values = None
        self.rhs = None
        self.solution = None
        self.fallbacks = 0
        self.gpu_solves = 0
        self.refinements = 0
        self._warned = False

    @staticmethod
    def _upper_csr(matrix):
        upper = triu(csr_matrix(matrix), format="csr")
        upper.sum_duplicates()
        upper.eliminate_zeros()
        upper.sort_indices()
        return upper

    def _new_pattern(self, upper):
        return (
            upper.shape[0],
            upper.indptr.tobytes(),
            upper.indices.tobytes(),
        )

    def _initialize(self, upper, pattern):
        self.outers = wp.array(
            np.asarray(upper.indptr, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self.indices = wp.array(
            np.asarray(upper.indices, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self.values = wp.array(
            np.asarray(upper.data, dtype=np.float64),
            dtype=wp.float64,
            device=self.device,
        )
        n = upper.shape[0]
        self.rhs = wp.zeros(n, dtype=wp.float64, device=self.device)
        self.solution = wp.zeros(n, dtype=wp.float64, device=self.device)
        wp.synchronize_device(self.device)
        self.solver = CUSolverDevice(
            self.outers.ptr,
            self.indices.ptr,
            self.values.ptr,
            n,
            upper.nnz,
        )
        self.solver.analyze_pattern()
        self.solver.factorize()
        self.pattern = pattern

    def _gpu_solve(self, matrix, rhs):
        upper = self._upper_csr(matrix)
        if upper.shape[0] == 0 or upper.nnz == 0:
            raise ValueError("dxslv requires a nonempty system")
        pattern = self._new_pattern(upper)
        if pattern != self.pattern:
            self._initialize(upper, pattern)
        else:
            self.values.assign(np.asarray(upper.data, dtype=np.float64))
            self.solver.refactorize(self.values.ptr)

        self.rhs.assign(np.asarray(rhs, dtype=np.float64))
        self.solution.zero_()
        self.solver.solve(self.rhs.ptr, self.solution.ptr)
        wp.synchronize_device(self.device)
        result = self.solution.numpy()
        relative_residual = np.inf
        for refinement in range(3):
            correction_rhs = rhs - np.asarray(matrix @ result).reshape(-1)
            relative_residual = np.linalg.norm(correction_rhs) / max(
                np.linalg.norm(rhs), 1.0
            )
            if relative_residual <= self.residual_tolerance:
                break
            if refinement == 2:
                break
            self.rhs.assign(correction_rhs)
            self.solution.zero_()
            self.solver.solve(self.rhs.ptr, self.solution.ptr)
            wp.synchronize_device(self.device)
            result += self.solution.numpy()
            self.refinements += 1
        if (not np.isfinite(result).all() or
                not np.isfinite(relative_residual) or
                relative_residual > self.residual_tolerance):
            raise RuntimeError(
                f"dxslv residual {relative_residual:.3e} exceeds "
                f"{self.residual_tolerance:.3e}"
            )
        self.gpu_solves += 1
        return result

    def solve(self, matrix, rhs):
        """Prefer cuDSS; use SciPy when unavailable or validation fails."""
        rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
        if self.available:
            try:
                return self._gpu_solve(matrix, rhs)
            except Exception as exc:
                self.fallbacks += 1
                # A failed factorization may leave cuDSS state unusable.
                self.solver = None
                self.pattern = None
                if not self._warned:
                    warnings.warn(
                        f"dxslv solve failed; using SciPy fallback: {exc}",
                        RuntimeWarning,
                    )
                    self._warned = True
        return np.asarray(spsolve(matrix, rhs)).reshape(-1)
