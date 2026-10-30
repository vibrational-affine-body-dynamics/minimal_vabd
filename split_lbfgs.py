"""Modified L-BFGS used by the paper's Newton--quasi-Newton solver.

Only collision-gradient differences are stored. At every application the full
secant vectors are reconstructed with the current, inexpensive elastic Hessian:

    y_tilde_i = delta_g_collision_i + H_elastic s_i.
"""

from collections import deque

import numpy as np
from scipy.sparse.linalg import spsolve


class SplitLBFGSHistory:
    """Limited collision-curvature history and modified two-loop recursion."""

    def __init__(self, size=8, curvature_tolerance=1.0e-12):
        self.size = int(size)
        self.curvature_tolerance = float(curvature_tolerance)
        self._pairs = deque(maxlen=max(self.size, 1))

    def __len__(self):
        return 0 if self.size == 0 else len(self._pairs)

    def clear(self):
        self._pairs.clear()

    def append(self, step, collision_gradient_difference):
        """Store ``s`` and collision-only ``y``; degenerate steps are skipped."""
        if self.size == 0:
            return False
        step = np.asarray(step, dtype=float).reshape(-1).copy()
        collision_y = np.asarray(
            collision_gradient_difference, dtype=float
        ).reshape(-1).copy()
        if step.shape != collision_y.shape:
            raise ValueError("step and collision gradient difference must match")
        if not np.isfinite(step).all() or not np.isfinite(collision_y).all():
            return False
        if np.linalg.norm(step) <= np.finfo(float).eps:
            return False
        self._pairs.append((step, collision_y))
        return True

    def apply_inverse(
        self, gradient, elastic_hessian, initial_hessian, linear_solve=None
    ):
        """Apply the split L-BFGS inverse approximation to ``gradient``.

        ``initial_hessian`` is the current elastic Hessian plus the collision
        Hessian saved by the most recent exact Newton refresh.
        """
        q = np.asarray(gradient, dtype=float).reshape(-1).copy()
        usable = []

        # The newest pair is processed first in loop one.
        for step, collision_y in reversed(self._pairs if self.size else ()):
            y_tilde = collision_y + np.asarray(elastic_hessian @ step).reshape(-1)
            curvature = float(np.dot(y_tilde, step))
            scale = np.linalg.norm(y_tilde) * np.linalg.norm(step)
            if (not np.isfinite(curvature) or
                    curvature <= self.curvature_tolerance * max(scale, 1.0)):
                continue
            rho = 1.0 / curvature
            alpha = rho * float(np.dot(step, q))
            q -= alpha * y_tilde
            usable.append((step, y_tilde, rho, alpha))

        if linear_solve is None:
            z = np.asarray(spsolve(initial_hessian, q)).reshape(-1)
        else:
            z = np.asarray(linear_solve(initial_hessian, q)).reshape(-1)

        # Reverse the order used above: oldest pair first in loop two.
        for step, y_tilde, rho, alpha in reversed(usable):
            beta = rho * float(np.dot(y_tilde, z))
            z += step * (alpha - beta)
        return z
