"""Generate SQEM targets for the 2,000-point squishy neural-MAT set."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import igl
import numpy as np

from g2m.medial import SlabMesh
from g2m.naive_fitter import Fitter
from utils.tobj import import_tobj


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "pqsample" / "36d_2000_pi.npy"
DEFAULT_Q = ROOT / "data" / "pqsample" / "squishy_q_36d_2000.npy"
DEFAULT_P = ROOT / "data" / "pqsample" / "p_squishy_q_36d_2000.npy"


def canonical_quaternions(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 36:
        raise ValueError(f"expected (N, 36) Poisson samples, got {points.shape}")
    q = points.reshape(-1, 9, 4).copy()
    norms = np.linalg.norm(q, axis=2, keepdims=True)
    if np.any(norms < 1.0e-12):
        raise ValueError("Poisson samples contain a zero quaternion")
    q /= norms
    q[q[:, :, 3] < 0.0] *= -1.0
    q[0] = np.array([0.0, 0.0, 0.0, 1.0])
    return q.reshape(-1, 36).astype(np.float32)


def dqs_weights(weights: np.ndarray) -> np.ndarray:
    """Turn nine signed modal fields into paired deformation/rest fields."""
    w = np.asarray(weights[:, 1:10], dtype=np.float64)
    w = w - w.min(axis=0, keepdims=True)
    return np.ascontiguousarray(np.hstack((w, 1.0 - w)))


def dqs_pose(vertices: np.ndarray, weights: np.ndarray, q36: np.ndarray) -> np.ndarray:
    q = np.zeros((18, 4), dtype=np.float64)
    q[:, 3] = 1.0
    q[:9] = q36.reshape(9, 4)
    translations = np.zeros((18, 3), dtype=np.float64)
    return igl.dqs(
        np.ascontiguousarray(vertices, dtype=np.float64), weights,
        np.ascontiguousarray(q), np.ascontiguousarray(translations),
    )


def generate(source: Path, q_path: Path, p_path: Path, count: int = 2000,
             checkpoint_every: int = 50) -> None:
    raw = np.load(source)[:count]
    if len(raw) != count:
        raise ValueError(f"requested {count} samples but {source} has {len(raw)}")
    quaternions = canonical_quaternions(raw)
    q_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(q_path, quaternions)

    vertices, tets = import_tobj(str(ROOT / "assets" / "squishy" / "squishy.tobj"))
    slab = SlabMesh(str(ROOT / "assets" / "squishy" / "ma" / "squishy.ma"))
    fitter = Fitter(vertices, tets, slab)
    weights = dqs_weights(np.load(ROOT / "data" / "W_squishy.npy"))
    partial_path = p_path.with_suffix(".partial.npy")
    if partial_path.exists():
        targets = np.load(partial_path)
        completed = int(np.count_nonzero(np.isfinite(targets[:, 0, 0])))
        print(f"resuming at sample {completed}", flush=True)
    else:
        targets = np.full((count, slab.nv, 4), np.nan, dtype=np.float32)
        completed = 0

    start = time.perf_counter()
    for i in range(completed, count):
        centers, radii = fitter.V2p(dqs_pose(vertices, weights, quaternions[i]))
        fitted = np.column_stack((centers, radii))
        if not np.isfinite(fitted).all() or np.any(radii <= 0.0):
            raise RuntimeError(f"invalid SQEM fit at sample {i}")
        targets[i] = fitted
        if (i + 1) % checkpoint_every == 0 or i + 1 == count:
            np.save(partial_path, targets)
            print(f"SQEM {i + 1}/{count} ({time.perf_counter() - start:.1f}s)", flush=True)

    np.save(p_path, targets)
    partial_path.unlink(missing_ok=True)
    metadata = {
        "model": "squishy", "samples": count,
        "input_shape": list(quaternions.shape), "target_shape": list(targets.shape),
        "parameterization": "9 canonical unit quaternions (xyzw)", "source": str(source),
    }
    p_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--q-output", type=Path, default=DEFAULT_Q)
    parser.add_argument("--p-output", type=Path, default=DEFAULT_P)
    parser.add_argument("--count", type=int, default=2000)
    args = parser.parse_args()
    generate(args.source, args.q_output, args.p_output, args.count)


if __name__ == "__main__":
    main()
