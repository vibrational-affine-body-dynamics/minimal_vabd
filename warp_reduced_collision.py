"""Batched Warp assembly for the linear medial collision pullback.

The implementation uses the paper's world-frame vibration coordinates and
never materializes the global embedding Jacobian.  It directly assembles the
positive-semidefinite Gauss--Newton pullback in persistent CUDA buffers.
"""

import numpy as np
import warp as wp
from scipy.sparse import coo_matrix


@wp.kernel
def build_linear_embedding(
    rest: wp.array2d(dtype=wp.float64),
    interpolation: wp.array2d(dtype=wp.float64),
    body: wp.array(dtype=wp.int32),
    valid: wp.array(dtype=wp.int32),
    affine: wp.array3d(dtype=wp.float64),
    translation: wp.array2d(dtype=wp.float64),
    q_vibration: wp.array2d(dtype=wp.float64),
    n_features: int,
    positions: wp.array2d(dtype=wp.float64),
    jacobian: wp.array3d(dtype=wp.float64),
):
    medial = wp.tid()
    if valid[medial] == 0:
        return

    body_id = body[medial]
    rest_position = wp.vec3d(rest[medial, 0], rest[medial, 1], rest[medial, 2])

    for world_axis in range(3):
        value = translation[body_id, world_axis]
        for local_axis in range(3):
            value += (
                affine[body_id, world_axis, local_axis]
                * rest_position[local_axis]
            )
            jacobian[medial, world_axis, local_axis * 3 + world_axis] = (
                rest_position[local_axis]
            )
        jacobian[medial, world_axis, 9 + world_axis] = wp.float64(1.0)

        for feature in range(n_features):
            weight = interpolation[medial, feature]
            value += weight * q_vibration[body_id, feature * 3 + world_axis]
            jacobian[medial, world_axis, 12 + feature * 3 + world_axis] = weight
        positions[medial, world_axis] = value


@wp.kernel
def assemble_reduced_gradient(
    jacobian: wp.array3d(dtype=wp.float64),
    gradient: wp.array2d(dtype=wp.float64),
    body: wp.array(dtype=wp.int32),
    valid: wp.array(dtype=wp.int32),
    n_bodies: int,
    n_modes: int,
    tilde_width: int,
    term: wp.float64,
    reduced_gradient: wp.array(dtype=wp.float64),
):
    medial, local_dof = wp.tid()
    if valid[medial] == 0:
        return
    value = wp.float64(0.0)
    for world_axis in range(3):
        value += jacobian[medial, world_axis, local_dof] * gradient[medial, world_axis]
    body_id = body[medial]
    system_dof = body_id * 12 + local_dof
    if local_dof >= 12:
        system_dof = n_bodies * 12 + body_id * tilde_width + local_dof - 12
    wp.atomic_add(reduced_gradient, system_dof, term * value)


@wp.kernel
def assemble_gauss_newton_blocks(
    jacobian: wp.array3d(dtype=wp.float64),
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    hessian_blocks: wp.array3d(dtype=wp.float64),
    pair_slots: wp.array(dtype=wp.int32),
    n_modes: int,
    term: wp.float64,
    reduced_blocks: wp.array3d(dtype=wp.float64),
):
    block_id, row_dof, col_dof = wp.tid()
    medial_row = rows[block_id]
    medial_col = cols[block_id]
    value = wp.float64(0.0)
    for world_row in range(3):
        left = jacobian[medial_row, world_row, row_dof]
        for world_col in range(3):
            value += (
                left
                * hessian_blocks[block_id, world_row, world_col]
                * jacobian[medial_col, world_col, col_dof]
            )
    wp.atomic_add(reduced_blocks, pair_slots[block_id], row_dof, col_dof, term * value)


class WarpLinearCollisionPullback:
    """Persistent CUDA buffers for a batched linear-medial pullback."""

    def __init__(
        self,
        rest_positions,
        interpolation,
        body_ids,
        valid,
        n_modes,
        device="cuda",
    ):
        self.device = device
        self.rest_np = np.ascontiguousarray(rest_positions, dtype=np.float64)
        self.interpolation_np = np.ascontiguousarray(interpolation, dtype=np.float64)
        self.body_np = np.ascontiguousarray(body_ids, dtype=np.int32)
        self.valid_np = np.ascontiguousarray(valid, dtype=np.int32)
        self.n_medials = self.rest_np.shape[0]
        self.n_bodies = int(self.body_np.max()) + 1
        self.n_modes = int(n_modes)
        self.tilde_width = self.n_modes - 12
        self.n_features = self.tilde_width // 3
        self.n_reduced = self.n_bodies * self.n_modes

        if self.interpolation_np.shape != (self.n_medials, self.n_features):
            raise ValueError("linear medial interpolation has an unexpected shape")

        self.rest = wp.array(self.rest_np, dtype=wp.float64, device=device)
        self.interpolation = wp.array(
            self.interpolation_np, dtype=wp.float64, device=device
        )
        self.body = wp.array(self.body_np, dtype=wp.int32, device=device)
        self.valid = wp.array(self.valid_np, dtype=wp.int32, device=device)
        self.affine = wp.zeros(
            (self.n_bodies, 3, 3), dtype=wp.float64, device=device
        )
        self.translation = wp.zeros(
            (self.n_bodies, 3), dtype=wp.float64, device=device
        )
        self.q_vibration = wp.zeros(
            (self.n_bodies, self.tilde_width), dtype=wp.float64, device=device
        )
        self.positions = wp.zeros(
            (self.n_medials, 3), dtype=wp.float64, device=device
        )
        self.jacobian = wp.zeros(
            (self.n_medials, 3, self.n_modes), dtype=wp.float64, device=device
        )
        self.medial_gradient = wp.zeros(
            (self.n_medials, 3), dtype=wp.float64, device=device
        )
        self.reduced_gradient = wp.zeros(
            self.n_reduced, dtype=wp.float64, device=device
        )
        self.block_capacity = 0
        self.pair_capacity = 0
        self.rows = None
        self.cols = None
        self.hessian_blocks = None
        self.pair_slots = None
        self.reduced_blocks = None
        self._rows_stage = None
        self._cols_stage = None
        self._hessian_stage = None
        self._slot_stage = None
        self._sparse_pattern_cache = {}
        self._embedding_current = False

    def update_embedding(self, affine, translation, q_vibration):
        affine = np.ascontiguousarray(affine, dtype=np.float64)
        translation = np.ascontiguousarray(translation, dtype=np.float64)
        q_vibration = np.ascontiguousarray(
            np.asarray(q_vibration).reshape(self.n_bodies, self.tilde_width),
            dtype=np.float64,
        )
        self.affine.assign(affine)
        self.translation.assign(translation)
        self.q_vibration.assign(q_vibration)
        self.jacobian.zero_()
        self.positions.zero_()
        wp.launch(
            build_linear_embedding,
            dim=self.n_medials,
            inputs=[
                self.rest,
                self.interpolation,
                self.body,
                self.valid,
                self.affine,
                self.translation,
                self.q_vibration,
                self.n_features,
                self.positions,
                self.jacobian,
            ],
            device=self.device,
        )
        self._embedding_current = True
        return self.positions.numpy()

    @staticmethod
    def _grown_capacity(required):
        capacity = 1
        while capacity < required:
            capacity *= 2
        return capacity

    def _ensure_block_capacity(self, required):
        if required <= self.block_capacity:
            return
        self.block_capacity = self._grown_capacity(required)
        self.rows = wp.zeros(
            self.block_capacity, dtype=wp.int32, device=self.device
        )
        self.cols = wp.zeros(
            self.block_capacity, dtype=wp.int32, device=self.device
        )
        self.hessian_blocks = wp.zeros(
            (self.block_capacity, 3, 3), dtype=wp.float64, device=self.device
        )
        self.pair_slots = wp.zeros(
            self.block_capacity, dtype=wp.int32, device=self.device
        )
        self._rows_stage = np.zeros(self.block_capacity, dtype=np.int32)
        self._cols_stage = np.zeros(self.block_capacity, dtype=np.int32)
        self._hessian_stage = np.zeros(
            (self.block_capacity, 3, 3), dtype=np.float64
        )
        self._slot_stage = np.zeros(self.block_capacity, dtype=np.int32)

    def _ensure_pair_capacity(self, required):
        if required <= self.pair_capacity:
            return
        self.pair_capacity = self._grown_capacity(required)
        self.reduced_blocks = wp.zeros(
            (self.pair_capacity, self.n_modes, self.n_modes),
            dtype=wp.float64,
            device=self.device,
        )

    def _system_dofs(self, body_id):
        dofs = np.empty(self.n_modes, dtype=np.int32)
        dofs[:12] = body_id * 12 + np.arange(12, dtype=np.int32)
        dofs[12:] = (
            self.n_bodies * 12
            + body_id * self.tilde_width
            + np.arange(self.tilde_width, dtype=np.int32)
        )
        return dofs

    def _sparse_pattern(self, pairs):
        key = tuple(pairs)
        cached = self._sparse_pattern_cache.get(key)
        if cached is not None:
            return cached
        rows = []
        cols = []
        for body_row, body_col in pairs:
            row_dofs = self._system_dofs(body_row)
            col_dofs = self._system_dofs(body_col)
            rows.append(np.repeat(row_dofs, self.n_modes))
            cols.append(np.tile(col_dofs, self.n_modes))
        pattern = (np.concatenate(rows), np.concatenate(cols))
        self._sparse_pattern_cache[key] = pattern
        return pattern

    def assemble(self, medial_gradient, rows, cols, blocks, term, with_hessian):
        if not self._embedding_current:
            raise RuntimeError("update_embedding() must precede collision assembly")

        gradient_np = np.ascontiguousarray(
            np.asarray(medial_gradient).reshape(self.n_medials, 3),
            dtype=np.float64,
        )
        self.medial_gradient.assign(gradient_np)
        self.reduced_gradient.zero_()
        wp.launch(
            assemble_reduced_gradient,
            dim=(self.n_medials, self.n_modes),
            inputs=[
                self.jacobian,
                self.medial_gradient,
                self.body,
                self.valid,
                self.n_bodies,
                self.n_modes,
                self.tilde_width,
                np.float64(term),
                self.reduced_gradient,
            ],
            device=self.device,
        )

        reduced_gradient = self.reduced_gradient.numpy()
        if not with_hessian:
            return reduced_gradient, None

        rows_np = np.ascontiguousarray(rows, dtype=np.int32)
        cols_np = np.ascontiguousarray(cols, dtype=np.int32)
        blocks_np = np.ascontiguousarray(blocks, dtype=np.float64).reshape(-1, 3, 3)
        n_blocks = len(rows_np)
        body_rows = self.body_np[rows_np] if n_blocks else np.empty(0, dtype=np.int32)
        body_cols = self.body_np[cols_np] if n_blocks else np.empty(0, dtype=np.int32)
        pairs = sorted(
            set(zip(body_rows.tolist(), body_cols.tolist()))
            | {(body_id, body_id) for body_id in range(self.n_bodies)}
        )
        pair_lookup = {pair: slot for slot, pair in enumerate(pairs)}
        pair_slots_np = np.array(
            [pair_lookup[(int(r), int(c))] for r, c in zip(body_rows, body_cols)],
            dtype=np.int32,
        )
        self._ensure_pair_capacity(len(pairs))
        self.reduced_blocks.zero_()
        if n_blocks:
            self._ensure_block_capacity(n_blocks)
            self._rows_stage[:n_blocks] = rows_np
            self._cols_stage[:n_blocks] = cols_np
            self._hessian_stage[:n_blocks] = blocks_np
            self._slot_stage[:n_blocks] = pair_slots_np
            self.rows.assign(self._rows_stage)
            self.cols.assign(self._cols_stage)
            self.hessian_blocks.assign(self._hessian_stage)
            self.pair_slots.assign(self._slot_stage)
            wp.launch(
                assemble_gauss_newton_blocks,
                dim=(n_blocks, self.n_modes, self.n_modes),
                inputs=[
                    self.jacobian,
                    self.rows,
                    self.cols,
                    self.hessian_blocks,
                    self.pair_slots,
                    self.n_modes,
                    np.float64(term),
                    self.reduced_blocks,
                ],
                device=self.device,
            )

        block_values = self.reduced_blocks.numpy()[: len(pairs)]
        sparse_rows, sparse_cols = self._sparse_pattern(pairs)
        hessian = coo_matrix(
            (block_values.reshape(-1), (sparse_rows, sparse_cols)),
            shape=(self.n_reduced, self.n_reduced),
        ).tocsc()
        hessian.eliminate_zeros()
        return reduced_gradient, hessian
