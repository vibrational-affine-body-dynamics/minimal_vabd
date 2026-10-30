"""Generalized eigenanalysis shared by the weight-generation scripts."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from scipy.sparse.linalg import eigsh


def solve_modes(K, M, model="squishy", output_dir="data/eigs", count=30,
                backend="auto", matlab_function="nullspace", force=False):
    """Solve K x = lambda M x, caching the same MAT files as the paper code."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_file = output_dir / f"{model}.mat"
    mode_file = output_dir / f"Q_{model}.mat"
    savemat(matrix_file, {"K": K, "M": M}, long_field_names=True)

    needs_solve = force or not mode_file.exists()
    if not needs_solve:
        needs_solve = loadmat(mode_file)["Vv"].shape[1] < min(int(count), K.shape[0] - 1)
    if needs_solve:
        matlab_error = None
        if backend in ("auto", "matlab"):
            try:
                import matlab.engine

                engine = matlab.engine.start_matlab()
                try:
                    engine.addpath(str(Path(__file__).resolve().parent), nargout=0)
                    engine.feval(matlab_function, model, str(output_dir.resolve()),
                                 float(count), nargout=0)
                finally:
                    engine.quit()
            except Exception as exc:
                matlab_error = exc
                if backend == "matlab":
                    raise

        if backend == "scipy" or (backend == "auto" and matlab_error is not None):
            k = min(int(count), K.shape[0] - 1)
            values, vectors = eigsh(K, k=k, M=M, which="SM")
            order = np.argsort(values)
            values, vectors = values[order], vectors[:, order]
            savemat(mode_file, {"Vv": vectors, "D": np.diag(values)})

    result = loadmat(mode_file)
    # This mirrors the lma2w scripts, which use the real part of numerical
    # complex pairs returned by MATLAB for the nominally real FEM system.
    return result["D"].real.astype(np.float64), result["Vv"].real.astype(np.float64)


def normalize_columns(array):
    scale = np.linalg.norm(array, axis=0, ord=np.inf, keepdims=True)
    scale[scale == 0.0] = 1.0
    return array / scale
