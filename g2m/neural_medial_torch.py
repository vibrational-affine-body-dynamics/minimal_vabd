"""Torch model used only by offline squishy training and checkpoint export."""

from __future__ import annotations

import torch
from torch import nn


class NeuralMedialMLP(nn.Module):
    def __init__(self, n_medials: int, width: int = 64):
        super().__init__()
        self.n_medials = n_medials
        self.width = width
        self.register_buffer("input_mean", torch.zeros(36))
        self.register_buffer("input_scale", torch.ones(36))
        self.register_buffer("output_mean", torch.zeros(4 * n_medials))
        self.register_buffer("output_scale", torch.ones(4 * n_medials))
        self.register_buffer("dqs_weights", torch.empty(0))
        self.register_buffer("rest_medials", torch.empty(0))
        self.network = nn.Sequential(
            nn.Linear(36, width), nn.Sigmoid(),
            nn.Linear(width, width), nn.Sigmoid(),
            nn.Linear(width, width), nn.Sigmoid(),
            nn.Linear(width, 4 * n_medials),
        )

    def configure_dqs_baseline(self, weights: torch.Tensor, rest_medials: torch.Tensor) -> None:
        self.dqs_weights = weights.detach().clone().float()
        self.rest_medials = rest_medials.detach().clone().float()

    @staticmethod
    def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
        x, y, z, w = q.unbind(-1)
        return torch.stack((
            1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
            2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
            2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
        ), dim=-1).reshape(q.shape[:-1] + (3, 3))

    def baseline(self, q: torch.Tensor) -> torch.Tensor:
        if self.dqs_weights.numel() == 0:
            return torch.zeros(q.shape[:-1] + (self.n_medials, 4), dtype=q.dtype, device=q.device)
        batched = q.ndim == 2
        qb = q.reshape(-1, 9, 4)
        identity = torch.zeros((len(qb), 9, 4), dtype=q.dtype, device=q.device)
        identity[..., 3] = 1.0
        transforms = torch.cat((qb, identity), dim=1)
        blended = torch.einsum("vm,bmf->bvf", self.dqs_weights, transforms)
        blended = blended / torch.linalg.vector_norm(blended, dim=-1, keepdim=True).clamp_min(1.0e-8)
        rotations = self._quat_to_matrix(blended)
        centers = torch.matmul(rotations, self.rest_medials[:, :3, None]).squeeze(-1)
        radii = self.rest_medials[:, 3:].expand(len(qb), -1, -1)
        result = torch.cat((centers, radii), dim=-1)
        return result if batched else result[0]

    def set_normalization(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.input_mean.copy_(inputs.mean(0))
        self.input_scale.copy_(inputs.std(0).clamp_min(1.0e-4))
        residual = (targets - self.baseline(inputs)).reshape(len(targets), -1)
        self.output_mean.copy_(residual.mean(0))
        self.output_scale.copy_(residual.std(0).clamp_min(1.0e-5))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        x = (q - self.input_mean) / self.input_scale
        y = self.network(x) * self.output_scale + self.output_mean
        return self.baseline(q) + y.reshape(q.shape[:-1] + (self.n_medials, 4))
