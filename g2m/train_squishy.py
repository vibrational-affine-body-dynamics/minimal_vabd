"""Train and qualify the squishy neural MAT before runtime use."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from g2m.medial import SlabMesh
from g2m.export_squishy_warp import export as export_warp_checkpoint
from g2m.neural_medial_torch import NeuralMedialMLP
from g2m.squishy_data import dqs_weights


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_Q = ROOT / "data" / "pqsample" / "squishy_q_36d_2000.npy"
DEFAULT_P = ROOT / "data" / "pqsample" / "p_squishy_q_36d_2000.npy"
DEFAULT_CHECKPOINT = ROOT / "data" / "squishy_neural_medial.pth"


def physical_metrics(pred: torch.Tensor, target: torch.Tensor, bbox_diag: float) -> dict:
    center_error = torch.linalg.vector_norm(pred[..., :3] - target[..., :3], dim=-1)
    radius_error = torch.abs(pred[..., 3] - target[..., 3])
    radius_scale = target[..., 3].abs().median().clamp_min(1.0e-6)
    return {
        "center_rmse": float(torch.sqrt(torch.mean(center_error.square())).item()),
        "center_rmse_over_bbox": float((torch.sqrt(torch.mean(center_error.square())) / bbox_diag).item()),
        "center_p95": float(torch.quantile(center_error, 0.95).item()),
        "radius_rmse": float(torch.sqrt(torch.mean(radius_error.square())).item()),
        "radius_rmse_relative": float((torch.sqrt(torch.mean(radius_error.square())) / radius_scale).item()),
        "radius_p95": float(torch.quantile(radius_error, 0.95).item()),
    }


def derivative_check(model: nn.Module, samples: torch.Tensor, directions: int = 8,
                     epsilon: float = 2.0e-3) -> dict:
    errors = []
    model.eval()
    for x0 in samples[:min(4, len(samples))]:
        x = x0.detach().clone().requires_grad_(True)
        for _ in range(directions):
            direction = torch.randn_like(x)
            direction /= torch.linalg.vector_norm(direction)
            _, tangent = torch.autograd.functional.jvp(model, x, direction)
            with torch.no_grad():
                finite = (model(x + epsilon * direction) - model(x - epsilon * direction)) / (2 * epsilon)
            denominator = torch.linalg.vector_norm(finite).clamp_min(1.0e-8)
            errors.append(float((torch.linalg.vector_norm(tangent - finite) / denominator).item()))
    return {
        "jacobian_relative_mean": float(np.mean(errors)),
        "jacobian_relative_max": float(np.max(errors)),
    }


def train(q_path: Path, p_path: Path, checkpoint: Path, epochs: int = 6000,
          width: int = 64, seed: int = 7) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.from_numpy(np.load(q_path).astype(np.float32))
    p = torch.from_numpy(np.load(p_path).astype(np.float32))
    if q.shape != (2000, 36) or p.shape[0] != 2000 or p.shape[2] != 4:
        raise ValueError(f"unexpected dataset shapes: q={tuple(q.shape)}, p={tuple(p.shape)}")

    permutation = torch.randperm(len(q), generator=torch.Generator().manual_seed(seed))
    train_ids, test_ids = permutation[:1500], permutation[1500:]
    model = NeuralMedialMLP(p.shape[1], width).to(device)
    medial = SlabMesh(str(ROOT / "assets" / "squishy" / "ma" / "squishy.ma"))
    rest = torch.from_numpy(np.column_stack((medial.V, medial.R)).astype(np.float32)).to(device)
    medial_weights = torch.from_numpy(
        dqs_weights(np.load(ROOT / "data" / "W_medial_squishy.npy")).astype(np.float32)
    ).to(device)
    model.configure_dqs_baseline(medial_weights, rest)
    model.set_normalization(q[train_ids].to(device), p[train_ids].to(device))
    loader = DataLoader(
        TensorDataset(q[train_ids], p[train_ids]), batch_size=200, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    loss_fn = nn.MSELoss()
    best_loss, best_state, stale = float("inf"), None, 0
    test_q, test_p = q[test_ids].to(device), p[test_ids].to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        for qb, pb in loader:
            qb, pb = qb.to(device), pb.to(device)
            mean = model.output_mean.reshape(1, -1, 4)
            scale = model.output_scale.reshape(1, -1, 4)
            loss = loss_fn((model(qb) - mean) / scale, (pb - mean) / scale)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch % 25 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                mean = model.output_mean.reshape(1, -1, 4)
                scale = model.output_scale.reshape(1, -1, 4)
                validation = loss_fn((model(test_q) - mean) / scale, (test_p - mean) / scale).item()
            if validation < best_loss - 1.0e-6:
                best_loss = validation
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 25
            if epoch % 250 == 0:
                print(f"epoch {epoch}: normalized validation MSE={validation:.6g}", flush=True)
            if epoch >= 2000 and stale >= 750:
                print(f"early stop at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(test_q)
    extent = (p[..., :3].amax((0, 1)) - p[..., :3].amin((0, 1))).norm().item()
    metrics = physical_metrics(prediction, test_p, extent)
    metrics.update(derivative_check(model, test_q))
    metrics.update({
        "normalized_validation_mse": best_loss, "train_samples": 1500,
        "validation_samples": 500, "seed": seed,
    })
    approved = (
        metrics["center_rmse_over_bbox"] < 0.01
        and metrics["radius_rmse_relative"] < 0.05
        and metrics["jacobian_relative_max"] < 0.01
    )
    payload = {
        "state_dict": model.state_dict(), "n_medials": p.shape[1], "width": width,
        "metrics": metrics, "approved": approved, "input": "nine canonical xyzw quaternions",
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
    export_warp_checkpoint(
        checkpoint, checkpoint.with_name(checkpoint.stem + "_warp.npz")
    )
    checkpoint.with_suffix(".json").write_text(
        json.dumps({"approved": approved, **metrics}, indent=2) + "\n"
    )
    print(json.dumps({"approved": approved, **metrics}, indent=2), flush=True)
    if not approved:
        raise RuntimeError("checkpoint failed the runtime quality gate")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=Path, default=DEFAULT_Q)
    parser.add_argument("--p", type=Path, default=DEFAULT_P)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()
    train(args.q, args.p, args.checkpoint, args.epochs, args.width)


if __name__ == "__main__":
    main()
