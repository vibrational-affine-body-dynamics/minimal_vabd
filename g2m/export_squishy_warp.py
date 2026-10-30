"""Export a trained Torch checkpoint to the Torch-free Warp runtime format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def export(checkpoint: Path, output: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not payload.get("approved", False):
        raise RuntimeError(f"refusing unapproved checkpoint: {checkpoint}")
    state = payload["state_dict"]
    arrays = {
        "n_medials": np.array(payload["n_medials"], dtype=np.int32),
        "width": np.array(payload["width"], dtype=np.int32),
    }
    for name, tensor in state.items():
        arrays[name.replace(".", "_")] = tensor.detach().cpu().numpy().astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data" / "squishy_neural_medial.pth")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "squishy_neural_medial_warp.npz")
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
