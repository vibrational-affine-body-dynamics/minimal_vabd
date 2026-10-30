import warp as wp
import numpy as np
import polyscope as ps 
from geometry.collision_cell import collision_eps
from stretch import h
from fem.params import gravity_np
import medial_reduced as medial_reduced_module
from medial_reduced import MedialRodComplex, vec
from scipy.sparse import block_diag, csc_matrix, hstack
from ortho import OrthogonalEnergy
from g2m.viewer import MedialViewer
from geometry.static_scene import StaticScene
import os
from reduced_collision import body_embedding
from warp_reduced_collision import WarpLinearCollisionPullback
from split_lbfgs import SplitLBFGSHistory
from dxslv_adapter import DxslvSolver
from coordinate_transform import local_vibration_and_jacobian
from mtk import medial_ccd as mtk_medial_ccd
import argparse
import contextlib

split_lbfgs = True
lbfgs_history_size = 8
lbfgs_newton_interval = 5
medial_ccd_safety = 0.9
# CCD's conservative advancement has its own 100-iteration budget.  One outer
medial_ccd_max_reintegrations = 1
medial_ccd_activation_distance = 2.0e-3
medial_ccd_min_distance = 1.0e-8
medial_ccd_safe_distance = 1.0e-6
ccd_only_affine = False
ccd_only_vabd = False
stability_checkpoint_interval = 0
medial_collision_stiffness = 1e7
damp_alpha = 0
damp_beta = 0
no_gravity_region = 15
class MedialVABD(MedialRodComplex):
    def __init__(self, h, meshes=[], transforms=[], static_meshes:StaticScene = None):
        super().__init__(h, meshes, transforms, static_meshes)
        self.body_centers = np.zeros((self.n_meshes, 3))
        for i in range(self.n_meshes): 
            self.body_centers[i] = self.transforms[i][:3, 3]
        self.abd_only = self.n_modes == 12 
        self.split_U0_U_tilide()
        self.define_warp_linear_pullback()
        self.define_mm()
        self.prefactor_once()
        self.define_A0_b0_btilde()
        
        self.ortho = OrthogonalEnergy(self.n_meshes)
        self.sum_weights()

        self.gen_F_idx()
        g = np.zeros(12)
        g[9:12] = gravity_np
        self.gravity = np.concatenate([g for _ in range(self.n_meshes)])
        self.process_collision = self.process_collision_medial_based
        self.compute_collision_energy = self.compute_collision_energy_medial_based

        V, R = self.get_VR()
        self.collider_medial.V = V
        self.collider_medial.R = R
        self.collision_frames = set()
        self.collision_evaluations = 0
        self.collision_constraints_total = 0
        self.max_collision_constraints = 0
        self.max_collision_energy = 0.0
        self.collect_stability_stats = False

        self.lbfgs_history = SplitLBFGSHistory(lbfgs_history_size)
        self.dxslv_solver = DxslvSolver()
        self.collision_hessian_seed = csc_matrix(
            (self.n_reduced, self.n_reduced)
        )
        self.previous_system_coordinates = None
        self.previous_collision_gradient = None
        self.nqn_newton_solves = 0
        self.nqn_lbfgs_solves = 0
        self.nqn_pairs = 0
        self.n_iter = 0
        # Pure CCD modes keep the last penetration-free reduced state.
        self.ccd_safe_affine = self.extract_z0(self.z).copy()
        self.ccd_safe_system = self.system_coordinates().copy()
        self.ccd_velocity_origin = None
        self.ccd_vabd_velocity_origin = None
        self.ccd_reintegrations_this_step = 0
        self.ccd_invocations = 0
        self.ccd_impacts = 0
        self.ccd_reintegrations = 0
        self.ccd_fallback_clamps = 0
        self.ccd_candidates = 0
        self.ccd_min_toi = 1.0
        self.ccd_checkpointed_frames = set()
        self.first_ipc_checkpoint_saved = False
        self.line_search_calls = 0
        self.line_search_alpha_one = 0
        self.line_search_reduced = 0
        self.line_search_failures = 0
        self.first_iteration_line_search_failures = 0
        self.nonlinear_converged = 0
        self.nonlinear_iteration_limits = 0

    def define_warp_linear_pullback(self):
        """Create persistent batched buffers for the linear medial proxy."""
        self.warp_linear_pullback = None
        if self.neural_medial is not None or self.abd_only:
            return

        n_features = (self.n_modes - 12) // 3
        interpolation = np.zeros((self.n_medial, n_features), dtype=np.float64)
        medial_start = 0
        for body_id, model in enumerate(self.models):
            count = self.slabmeshes[model].nv
            interpolation[medial_start:medial_start + count] = self.diags_tilde_lhs[body_id]
            medial_start += self.v_cnt_max if medial_reduced_module.padding else count

        self.warp_linear_pullback = WarpLinearCollisionPullback(
            self.V_medial_rest,
            interpolation,
            self.body_medial,
            self.R_rest >= 0.0,
            self.n_modes,
        )

    def use_newton(self):
        """Refresh the collision Hessian every five frames (paper default)."""
        return (not split_lbfgs or
                (self.n_iter == 0 and
                 self.frame % lbfgs_newton_interval == 0))

    def use_ipc_collision(self):
        return ccd_only_affine or ccd_only_vabd

    def use_full_ipc_collision(self):
        return ccd_only_vabd

    def homogeneous_medial_positions(self, affine):
        """Evaluate the affine MAT used by CCD (constant dp/dq_A)."""
        return np.asarray(self.Um0 @ affine).reshape((-1, 3))

    def set_affine_coordinates(self, affine):
        for i in range(self.n_meshes):
            self.z[i * self.n_modes:i * self.n_modes + 12] = \
                affine[i * 12:(i + 1) * 12]

    def set_system_coordinates(self, coordinates):
        """Assign the affine-first coordinate ordering used by the solver."""
        affine_dim = self.n_meshes * 12
        affine = np.asarray(coordinates[:affine_dim])
        self.set_affine_coordinates(affine)
        self.set_q_vibration(coordinates[affine_dim:])
        self.sync_local_coordinates()

    def linear_medial_positions(self, coordinates):
        """Evaluate ``p=A*p_rest+t+L*q_V`` without mutating the state."""
        coordinates = np.asarray(coordinates)
        affine_dim = self.n_meshes * 12
        affine = coordinates[:affine_dim]
        vibration = coordinates[affine_dim:]
        positions = np.copy(self.V_medial_rest)
        medial_start = 0
        vibration_width = self.n_modes - 12
        for body, model in enumerate(self.models):
            count = self.slabmeshes[model].nv
            body_affine = affine[body * 12:(body + 1) * 12]
            matrix = body_affine[:9].reshape((3, 3)).T
            translation = body_affine[9:12]
            body_vibration = vibration[
                body * vibration_width:(body + 1) * vibration_width
            ]
            rest = self.V_medial_rest[medial_start:medial_start + count]
            value = (
                rest @ matrix.T
                + translation
                + self.diags_tilde_lhs[body]
                @ body_vibration.reshape((-1, 3))
            )
            positions[medial_start:medial_start + count] = value
            medial_start += (
                self.v_cnt_max if medial_reduced_module.padding else count
            )
        return positions

    def query_medial_position_ccd(self, start_positions, end_positions):
        """Run medial and static-surface CCD on supplied center endpoints."""
        start_positions = np.asarray(start_positions, dtype=np.float64)
        end_positions = np.asarray(end_positions, dtype=np.float64)
        static = self.static_meshes
        has_static = static is not None and getattr(static, "has_medials", False)
        static_positions = (np.asarray(static.V_medial, dtype=np.float64)
                            if has_static else np.empty((0, 3), dtype=np.float64))
        static_radii = (np.asarray(static.R, dtype=np.float64)
                        if has_static else np.empty(0, dtype=np.float64))
        static_edges = (np.asarray(static.E_medial, dtype=np.int32)
                        if has_static else np.empty((0, 2), dtype=np.int32))
        static_faces = (np.asarray(static.F_medial, dtype=np.int32)
                        if has_static else np.empty((0, 3), dtype=np.int32))
        medial_toi, medial_candidates = mtk_medial_ccd(
            start_positions, end_positions,
            np.asarray(self.R_rest, dtype=np.float64),
            np.asarray(self.E_medial, dtype=np.int32),
            np.asarray(self.F_medial, dtype=np.int32),
            np.asarray(self.body_medial, dtype=np.int32),
            static_positions, static_radii, static_edges, static_faces,
        )
        surface_toi, surface_candidates = self.collider_medial.surface_ccd(
            start_positions, end_positions, self.R_rest
        )
        return (
            min(float(medial_toi), float(surface_toi)),
            int(medial_candidates) + int(surface_candidates),
        )

    def query_medial_ccd(self, affine_start, affine_end):
        start_positions = np.asarray(
            self.homogeneous_medial_positions(affine_start), dtype=np.float64
        )
        end_positions = np.asarray(
            self.homogeneous_medial_positions(affine_end), dtype=np.float64
        )
        return self.query_medial_position_ccd(start_positions, end_positions)

    def prepare_ccd_reintegration(self):
        """Trim a pure-CCD step to its first medial TOI and reintegrate."""
        if ccd_only_vabd:
            return self.prepare_full_ccd_reintegration()
        if not ccd_only_affine:
            self.ccd_reintegrations_this_step = 0
            return False

        candidate = self.extract_z0(self.z)
        toi, candidates = self.query_medial_ccd(self.ccd_safe_affine, candidate)
        toi = float(toi)
        self.ccd_invocations += 1
        self.ccd_candidates += int(candidates)
        candidate_min_distance = self.collider_medial.minimum_ipc_distance(
            self.homogeneous_medial_positions(candidate),
            self.R_rest,
            medial_ccd_activation_distance,
        )
        candidate_is_feasible = (
            candidate_min_distance > medial_ccd_safe_distance
        )
        if toi >= 1.0 - 1.0e-9 and candidate_is_feasible:
            self.ccd_safe_affine = candidate.copy()
            self.ccd_reintegrations_this_step = 0
            return False

        # Conservative advancement can occasionally exhaust its iterations
        # without reporting an impact.  The IPC feasibility query is the final
        # authority: never promote a penetrated endpoint to the safe origin.
        if toi >= 1.0 - 1.0e-9:
            toi = 1.0

        self.ccd_impacts += 1
        self.ccd_min_toi = min(self.ccd_min_toi, toi)
        if (ccd_only_affine and
                self.frame not in self.ccd_checkpointed_frames):
            self.save_preimpact_state(self.ccd_safe_affine)
            self.ccd_checkpointed_frames.add(self.frame)
        if self.ccd_velocity_origin is None:
            self.ccd_velocity_origin = self.ccd_safe_affine.copy()
        safe_toi = medial_ccd_safety * max(0.0, min(1.0, toi))
        safe_affine = (self.ccd_safe_affine +
                       safe_toi * (candidate - self.ccd_safe_affine))
        safe_affine = self.filter_ipc_feasible_affine(
            self.ccd_safe_affine, safe_affine
        )
        self.set_affine_coordinates(safe_affine)
        self.lbfgs_history.clear()
        self.previous_system_coordinates = None
        self.previous_collision_gradient = None

        if self.ccd_reintegrations_this_step < medial_ccd_max_reintegrations:
            # The prefix ending at safe_affine has already been certified and
            # handled.  Continue CCD from there so the next solve only checks
            # the unresolved path; otherwise every retry redetects the same
            # first impact from the original step endpoint.  The original
            # The original step start remains in ccd_velocity_origin.
            self.ccd_safe_affine = safe_affine.copy()
            self.ccd_reintegrations_this_step += 1
            self.ccd_reintegrations += 1
            return True

        # A quadratic penalty need not settle strictly outside contact.  Keep
        # the last CCD-safe affine state after bounded reintegration attempts.
        self.ccd_fallback_clamps += 1
        self.ccd_safe_affine = safe_affine.copy()
        self.ccd_reintegrations_this_step = 0
        return False

    def prepare_full_ccd_reintegration(self):
        """CCD correction for affine plus reduced vibration coordinates."""
        candidate = self.system_coordinates()
        start_positions = self.linear_medial_positions(self.ccd_safe_system)
        end_positions = self.linear_medial_positions(candidate)
        toi, candidates = self.query_medial_position_ccd(
            start_positions, end_positions
        )
        toi = float(toi)
        self.ccd_invocations += 1
        self.ccd_candidates += int(candidates)
        candidate_distance = self.collider_medial.minimum_ipc_distance(
            end_positions, self.R_rest, medial_ccd_activation_distance
        )
        if (toi >= 1.0 - 1.0e-9 and
                candidate_distance > medial_ccd_safe_distance):
            self.ccd_safe_system = candidate.copy()
            self.ccd_safe_affine = candidate[:self.n_meshes * 12].copy()
            self.ccd_reintegrations_this_step = 0
            return False

        self.ccd_impacts += 1
        self.ccd_min_toi = min(self.ccd_min_toi, toi)
        if self.ccd_vabd_velocity_origin is None:
            self.ccd_vabd_velocity_origin = self.ccd_safe_system.copy()
        safe_toi = medial_ccd_safety * max(0.0, min(1.0, toi))
        safe = self.ccd_safe_system + safe_toi * (
            candidate - self.ccd_safe_system
        )
        safe = self.filter_full_ipc_feasible(
            self.ccd_safe_system, safe
        )
        self.set_system_coordinates(safe)
        self.lbfgs_history.clear()
        self.previous_system_coordinates = None
        self.previous_collision_gradient = None

        if self.ccd_reintegrations_this_step < medial_ccd_max_reintegrations:
            self.ccd_safe_system = safe.copy()
            self.ccd_safe_affine = safe[:self.n_meshes * 12].copy()
            self.ccd_reintegrations_this_step += 1
            self.ccd_reintegrations += 1
            return True

        self.ccd_fallback_clamps += 1
        self.ccd_safe_system = safe.copy()
        self.ccd_safe_affine = safe[:self.n_meshes * 12].copy()
        self.ccd_reintegrations_this_step = 0
        return False

    def filter_full_ipc_feasible(self, origin, candidate):
        """Keep a full reduced CCD state inside the IPC domain."""
        def minimum_distance(coordinates):
            positions = self.linear_medial_positions(coordinates)
            return self.collider_medial.minimum_ipc_distance(
                positions, self.R_rest, medial_ccd_activation_distance
            )

        candidate_distance = minimum_distance(candidate)
        if candidate_distance > medial_ccd_safe_distance:
            return candidate
        origin_distance = minimum_distance(origin)
        if origin_distance <= medial_ccd_min_distance:
            raise FloatingPointError(
                "the stored full CCD origin is not IPC feasible "
                f"(frame={self.frame}, origin_distance="
                f"{origin_distance:.9g}, candidate_distance="
                f"{candidate_distance:.9g})"
            )
        if origin_distance <= medial_ccd_safe_distance:
            return origin.copy()
        lower, upper = 0.0, 1.0
        for _ in range(40):
            alpha = 0.5 * (lower + upper)
            trial = origin + alpha * (candidate - origin)
            if minimum_distance(trial) > medial_ccd_safe_distance:
                lower = alpha
            else:
                upper = alpha
        return origin + (lower * medial_ccd_safety) * (candidate - origin)

    def filter_ipc_feasible_affine(self, origin, candidate):
        """Backtrack a CCD guess until the logarithmic barrier is defined."""
        def minimum_distance(affine):
            return self.collider_medial.minimum_ipc_distance(
                self.homogeneous_medial_positions(affine),
                self.R_rest,
                medial_ccd_activation_distance,
            )

        candidate_distance = minimum_distance(candidate)
        if candidate_distance > medial_ccd_safe_distance:
            return candidate
        origin_distance = minimum_distance(origin)
        if origin_distance <= medial_ccd_min_distance:
            raise FloatingPointError(
                "the stored CCD origin is not IPC feasible "
                f"(frame={self.frame}, reintegration="
                f"{self.ccd_reintegrations_this_step}, "
                f"origin_distance={origin_distance:.9g}, "
                f"candidate_distance={candidate_distance:.9g})"
            )
        if origin_distance <= medial_ccd_safe_distance:
            # Keep the already feasible origin.  Advancing from a point this
            # close to the boundary only creates zero-TOI roundoff chatter.
            return origin.copy()
        lower, upper = 0.0, 1.0
        for _ in range(40):
            alpha = 0.5 * (lower + upper)
            trial = origin + alpha * (candidate - origin)
            if minimum_distance(trial) > medial_ccd_safe_distance:
                lower = alpha
            else:
                upper = alpha
        # Leave a small relative margin inside the feasible interval.
        alpha = max(0.0, lower * medial_ccd_safety)
        return origin + alpha * (candidate - origin)

    def system_coordinates(self):
        """Return the paper's world coordinates ``[q_A, q_V]``."""
        return np.concatenate((self.extract_z0(self.z), self.extract_q_vibration()))

    def define_collider(self):
        self.define_medials()

    def gen_F_idx(self):
        '''
        optioal, only used in optimizing fetching the deformation gradient 
        '''
        idx = [np.arange(i * self.n_modes, i * self.n_modes + 9) for i in range(self.n_meshes)]
        self.F_idx = np.array(idx)

    def sum_weights(self):
        ww  = self.W.numpy()
        self.sum_W = np.zeros((self.n_meshes), float)
        for i in range(self.n_meshes):
            start = self.tet_start[i]
            nt = self.tet_start[i + 1] - self.tet_start[i]
            self.sum_W[i] = np.sum(ww[start: start + nt])
    
    def define_A0_b0_btilde(self):
        self.A0 = np.zeros((self.n_meshes * 12, self.n_meshes * 12))
        self.b0 = np.zeros((self.n_meshes * 12,))
        self.b_tilde = np.zeros((self.n_meshes * (self.n_modes - 12),))


    def define_U(self):
        # self.U = np.zeros((self.n_nodes * 3, self.n_reduced))
        # nodes_per_mesh = self.n_nodes // self.n_meshes
        x0 = self.xcs.numpy()
        diags = []
        
        start = 0 
        for i in range(self.n_meshes): 
            model = self.models[i]
            Q = self.Q[model]
            nv = Q.shape[0]
            xi = x0[start: start + nv]
            Ui = self.lbs_matrix(xi, Q)
            diags.append(Ui)
            start += nv
        # self.U = block_diag(diags)

        # self.wp_define_U()        
        # Uwp = self.to_scipy_bsr(self.Uwp)
        # print(f"U norm = {norm(self.U)}, diff norm = {norm(self.U - Uwp)}")

    def split_U0_U_tilide(self):
        # self.U0 = np.zeros((self.n_nodes * 3, self.n_meshes * 12))
        # self.U_tilde = np.zeros((self.n_nodes * 3, (self.n_modes - 12) * self.n_meshes))
        # n_nodes_per_mesh = self.n_nodes // self.n_meshes
        x0np = self.xcs.numpy()
        diags0 = []
        diags_tilde = []
        
        start = 0
        for i in range(self.n_meshes):
            model = self.models[i]
            Q = self.Q[model]
            nv = Q.shape[0]

            
            V = x0np[start: start + nv]
            jac = self.lbs_matrix(V, Q)
            
            diags0.append(jac[:, :12])
            if not self.abd_only:
                diags_tilde.append(jac[:, 12:])
            start += nv
            # self.U0[start: end, 12 * i: 12 * (i + 1)] = rhs


            # self.U_tilde[start:end, i * (self.n_modes - 12): (i + 1) * (self.n_modes - 12)] = self.lbs_matrix(V, self.Q[:, 1:])

        self.U0 = block_diag(diags0)
        self.U_tilde = block_diag(diags_tilde) if not self.abd_only else None

        # fill Um_tilde 
        
        # self.Um_tilde = np.zeros((self.n_medial * 3, self.z_tilde.shape[0]))
        # self.Um0 = np.zeros((self.n_medial * 3, 12 * self.n_meshes))
        diags0 = []
        diags_tilde = []
        self.diags_tilde_lhs = []
        
        start = 0
        for i in range(self.n_meshes):
            # start = i * self.n_modes
            # end = (i + 1) * self.n_modes

            # self.Um_tilde[:, i * (self.n_modes - 12): (i + 1) * (self.n_modes - 12)] = self.Um[:, start + 12: end]
            # self.Um0[:, i * 12: (i + 1) * 12] = self.Um[:, start: start + 12]
            model = self.models[i]
            nv = self.slabmeshes[model].nv
            Vmi = self.V_medial_rest[start: start + nv]
            W_medial = self.W_medial[model]

            ji = self.lbs_matrix(Vmi, W_medial)
            j0i = ji[:,: 12]
            jti = ji[:, 12:]
            diags0.append(j0i)
            diags_tilde.append(jti)

            if not self.abd_only: 
                di = self.lbs_no_kron(Vmi, W_medial[:, 1:])
                self.diags_tilde_lhs.append(di)     
            start += nv
        self.Um0 = block_diag(diags0, "csc")
        self.Um_tilde = block_diag(diags_tilde, "csc")

    def define_mm(self):
        self.mm = self.U0.T @ self.to_scipy_bsr(self.M_sparse) @ self.U0
    
    def lbs_no_kron(self, V, W):
        nvm = V.shape[0]
        v1 = np.ones((nvm, 4))
        v1[:, :3] = V
        lhs = np.hstack([W[:, j: j + 1] * v1 for j in range(W.shape[1])])
        return lhs

    def get_F(self, i):
        '''
        columns of F is formed by segments of z 
        vec(F) = z
        '''
        z = self.z[i * self.n_modes: i * self.n_modes + 12]
        F = z.reshape((-1, 3)).T
        return F[:, :3]

    def get_F_batch(self):
        '''
        columns of F is formed by segments of z 
        vec(F) = z
        '''
        f = self.z[self.F_idx].reshape((self.n_meshes, 3, 3)) # n_meshes x 3 x3
        return np.transpose(f, axes = (0, 2, 1))

    def save_sparse(self, name, sparse_matrix):
        np.save(f"data/{name}_data.npy", sparse_matrix.data)
        np.save(f"data/{name}_indices.npy", sparse_matrix.indices)
        np.save(f"data/{name}_indptr.npy", sparse_matrix.indptr)

    def prefactor_once(self):
        if self.abd_only:
            return
        h = self.h
        self.compute_K()
        self.K0 = (
            self.U_tilde.T @ self.to_scipy_bsr() @ self.U_tilde * (h * h)
        ).tocsc()
        self.M_tilde = (
            self.U_tilde.T @ self.to_scipy_bsr(self.M_sparse) @ self.U_tilde
        ).tocsc()
        self.A_tilde = self.K0 + self.M_tilde
        self.C = self.K0 * damp_beta / (h * h) + self.M_tilde * damp_alpha
        self.save_sparse("K0", self.K0)
        self.save_sparse("M_tilde", self.M_tilde)
        print("saved K0 and M_tilde to data/")


    def define_z(self, transforms):
        self.n_modes = self.Q[self.models[0]].shape[1] * 12 
        self.n_meshes = len(transforms)
        self.n_reduced = self.n_modes * self.n_meshes
        
        self.z = np.zeros(self.n_reduced)
        for i in range(self.n_meshes):
            self.z[i * self.n_modes: i * self.n_modes + 9] = vec(np.identity(3))

        self.z0 = np.zeros(self.n_meshes * 12)
        self.z_dot = np.zeros_like(self.z0)

        self.dz = np.zeros_like(self.z)

        vibration_size = self.n_reduced - 12 * self.n_meshes
        # Object-frame vibration is derived from the world-frame state in z.
        self.z_tilde = np.zeros(self.n_reduced - 12 * self.n_meshes)
        self.q_v0 = np.zeros(vibration_size)
        self.q_v_dot = np.zeros(vibration_size)
        self.dq_v = np.zeros(vibration_size)
        self.z_fields = [self.z, self.z0, self.z_dot, self.q_v0, self.q_v_dot]
        self.fields_alias = ["z", "z0", "z_dot", "q_v0", "q_v_dot"]

    def extract_z0(self, z):
        z0 = np.zeros(self.n_meshes * 12)
        for i in range(self.n_meshes):
            start = i * self.n_modes
            end = start + 12
            z0[i * 12: (i + 1) * 12] = z[start: end]
        return z0

    def extract_q_vibration(self, z=None):
        source = self.z if z is None else np.asarray(z)
        width = self.n_modes - 12
        q_vibration = np.empty(self.n_meshes * width)
        for i in range(self.n_meshes):
            q_vibration[i * width:(i + 1) * width] = source[
                i * self.n_modes + 12:(i + 1) * self.n_modes
            ]
        return q_vibration

    def set_q_vibration(self, q_vibration):
        width = self.n_modes - 12
        for i in range(self.n_meshes):
            self.z[i * self.n_modes + 12:(i + 1) * self.n_modes] = \
                q_vibration[i * width:(i + 1) * width]

    def sync_local_coordinates(self, with_jacobian=False):
        q_affine = self.extract_z0(self.z)
        q_vibration = self.extract_q_vibration()
        local, jacobian = local_vibration_and_jacobian(q_affine, q_vibration)
        self.z_tilde[:] = local
        return jacobian if with_jacobian else None

    def update_x0_xdot(self):
        # super().update_x0_xdot()

        z = self.extract_z0(self.z)
        q_vibration = self.extract_q_vibration()
        self.sync_local_coordinates()

        # Keep the surface buffer used by the viewer synchronized with the
        # reduced state.  Leaving this disabled made the displayed mesh
        # disagree with the medial geometry used by collision detection.
        surface_tilde = np.concatenate([
            self.z[i * self.n_modes + 12:(i + 1) * self.n_modes]
            for i in range(self.n_meshes)
        ])
        surface = self.U0 @ z
        if not self.abd_only:
            surface = surface + self.U_tilde @ surface_tilde
        self.states.x.assign(np.asarray(surface).reshape((-1, 3)))

        full_velocity_corrected = self.ccd_vabd_velocity_origin is not None
        if full_velocity_corrected:
            affine_dim = self.n_meshes * 12
            origin = self.ccd_vabd_velocity_origin
            self.z_dot[:] = (
                z - origin[:affine_dim]
            ) / self.h
            self.q_v_dot[:] = (
                q_vibration - origin[affine_dim:]
            ) / self.h
            self.ccd_vabd_velocity_origin = None
        elif self.ccd_velocity_origin is None:
            self.z_dot[:] = (z - self.z0) / self.h
        else:
            self.z_dot[:] = (z - self.ccd_velocity_origin) / self.h
            self.ccd_velocity_origin = None
        self.z0[:] = z

        if not full_velocity_corrected:
            self.q_v_dot[:] = (q_vibration - self.q_v0) / self.h
        self.q_v0[:] = q_vibration

    # super.solve:
    # def solve(self):
    #     dz = solve(self.A_reduced, self.b_reduced, assume_a="sym")
    #     self.dz[:] = dz
    #     self.states.dx.assign((self.U @ dz).reshape(-1, 3))

    def solve(self):
        self.A0[:] = 0.
        self.b0[:] = 0.

        h2 = self.h * self.h
        
        # set A0, b0
        with wp.ScopedTimer("compute A0"):
            Fs = [self.get_F(i) for i in range(self.n_meshes)]
            gg, HH = self.ortho.analyze(np.array(Fs))
            for i in range(self.n_meshes):
                # fi = self.get_F(i)

                # g, H = self.ortho.analyze(fi)
                g = gg[i]
                H = HH[i]

                mmi = self.mm[i * 12: (i + 1) * 12, i * 12: (i + 1) * 12]
                aai = H * h2 * self.sum_W[i] + mmi

                ai = self.z0[i * 12: i * 12 + 9].reshape(3, 3).T
                bi = self.body_centers[i]
                y = (ai @ bi)[1] + self.z0[i * 12 + 10]
                
                ci = H * damp_beta * self.sum_W[i]
                if y <= no_gravity_region:
                    ci += mmi * damp_alpha
                aai += ci
                self.A0[i * 12: (i + 1) * 12, i * 12: (i + 1) * 12] = aai
                damp_term = ci @ (self.z[i * self.n_modes: i * self.n_modes + 12] - self.z0[i * 12: i * 12 + 12]) * self.h
                self.b0[i * 12: (i + 1) * 12] = g * h2 * self.sum_W[i] + mmi @ (self.z[i * self.n_modes: i * self.n_modes + 12] - self.z_hat(i)) + damp_term

        q_vibration = self.extract_q_vibration()
        self.b_tilde = (
            self.K0 @ q_vibration
            + self.M_tilde @ (q_vibration - self.q_v_hat())
            + self.C @ (q_vibration - self.q_v0) * self.h
            + self.compute_excitement()
        ) if not self.abd_only else 0.0

        blocks = [csc_matrix(self.A0)]
        if not self.abd_only:
            blocks.append(self.A_tilde)
        A_sys = block_diag(blocks, format="csc")
            
        b_sys = np.zeros(self.n_reduced)
        b_sys[:self.n_meshes * 12] = self.b0# + self.b0_col
        b_sys[self.n_meshes * 12:] = self.b_tilde# + self.b_col_tilde

        self.b_sys = b_sys
        with wp.ScopedTimer("linalg system"):
            gradient = b_sys + self.b_sys_col
            coordinates = self.system_coordinates()
            if self.use_newton():
                dz_sys = self.dxslv_solver.solve(
                    A_sys + self.collision_hessian_seed, gradient
                )
                self.lbfgs_history.clear()
                self.nqn_newton_solves += 1
            else:
                if (self.n_iter > 0 and
                        self.previous_system_coordinates is not None):
                    accepted = self.lbfgs_history.append(
                        coordinates - self.previous_system_coordinates,
                        self.b_sys_col - self.previous_collision_gradient,
                    )
                    self.nqn_pairs += int(accepted)
                dz_sys = self.lbfgs_history.apply_inverse(
                    gradient,
                    A_sys,
                    A_sys + self.collision_hessian_seed,
                    linear_solve=self.dxslv_solver.solve,
                )
                self.nqn_lbfgs_solves += 1
            self.previous_system_coordinates = coordinates.copy()
            self.previous_collision_gradient = self.b_sys_col.copy()
            
        cos_dz_b = np.dot(dz_sys, b_sys + self.b_sys_col) / np.linalg.norm(dz_sys) / np.linalg.norm(b_sys + self.b_sys_col)
        print(f"dz dot gradient cos = {cos_dz_b}")
        if cos_dz_b < 0.0:
            print("warning: dz is in the opposite direction of gradient")
        dz0 = dz_sys[:self.n_meshes * 12]
        self.dq_v[:] = dz_sys[self.n_meshes * 12:]

        dz = np.zeros_like(self.z)
        for i in range(self.n_meshes):
            start = i * self.n_modes
            end = start + self.n_modes
            dz[start: start + 12] = dz0[i * 12: (i + 1) * 12]
            dz[start + 12: end] = self.dq_v[
                i * (self.n_modes - 12):(i + 1) * (self.n_modes - 12)
            ]

        self.dz[:] = dz

    def converged(self):
        norm_dz = np.max(np.linalg.norm(self.dz.reshape(self.n_meshes, self.n_modes), axis = 1))
        print(f"dz norm = {norm_dz}")
        return norm_dz < 2e-3
        
    def line_search(self):
        z_tmp = np.copy(self.z)

        alpha = 1.0
        e00 = self.compute_psi() + self.compute_inertia()
        e0c = self.compute_collision_energy()
        E0 = e00 + e0c 

        # e01 = self.compute_inertia2()
        # print(f"diff energy = {e01 - e00}, e = {e01}")

        while True:
            self.z[:] = z_tmp - alpha * self.dz
            self.sync_local_coordinates()

            e10 = self.compute_psi() + self.compute_inertia()
            # e11 = self.compute_inertia2()
            # print(f"diff energy = {e11 - e10}, e = {e11}")

            e1c = self.compute_collision_energy()
            E1 = e10 + e1c
            if E1 < E0:
                break
            if alpha < 1e-2:
                self.z[:] = z_tmp
                self.sync_local_coordinates()
                alpha = 0.0
                break
            alpha *= 0.5
        print(f"line search alpha = {alpha}")
        self.line_search_calls += 1
        if alpha == 1.0:
            self.line_search_alpha_one += 1
        elif alpha == 0.0:
            self.line_search_failures += 1
            self.first_iteration_line_search_failures += int(self.n_iter == 0)
        else:
            self.line_search_reduced += 1
        return alpha
    
    def compute_psi(self):
        with wp.ScopedTimer("Psi"):
            h2 = self.h ** 2
            ff = self.get_F_batch()
            e = self.ortho.energy_batch(ff)
            ret = np.sum(e * self.sum_W) * h2

        if not self.abd_only:
            q_vibration = self.extract_q_vibration()
            ret += q_vibration @ self.K0 @ q_vibration * 0.5
        return ret

    def compute_inertia(self):
        for i in range(self.n_meshes):
            ai = self.z0[i * 12: i * 12 + 9].reshape(3, 3).T
            bi = self.body_centers[i]
            y = (ai @ bi)[1] + self.z0[i * 12 + 10]
            if y > no_gravity_region:
                self.gravity[i * 12 + 9: i * 12 + 12] = 0.0
            else:
                self.gravity[i * 12 + 9: i * 12 + 12] = gravity_np

        with wp.ScopedTimer("inertia"):

            zh = self.z0 + self.h * self.z_dot + self.gravity * self.h * self.h
            z0 = self.extract_z0(self.z)
            dz0 = z0 - zh

            ret = (dz0 @ self.mm @ dz0) * 0.5
            if not self.abd_only:
                dq_v = self.extract_q_vibration() - self.q_v_hat()
                ret += dq_v @ self.M_tilde @ dq_v * 0.5
        return ret
    
    def q_v_hat(self):
        return self.h * self.q_v_dot + self.q_v0

    def z_hat(self, i):
        zh = self.z0 + self.h * self.z_dot
        ret = zh[i * 12: (i + 1) * 12]

        ai = self.z0[i * 12: i * 12 + 9].reshape(3, 3).T
        bi = self.body_centers[i]
        y = (ai @ bi)[1] + self.z0[i * 12 + 10]

        if y > no_gravity_region:
            pass
        else: 
            ret[9: 12] += gravity_np * self.h * self.h
        return ret

    def compute_excitement(self):
        return np.zeros_like(self.q_v0)

    def reset_z(self, from_frame = 0):
        self.z[:] = 0.0
        self.dz[:] = 0.0

        
        for i in range(self.n_meshes):
            self.z[i * self.n_modes: i * self.n_modes + 9] = vec(np.identity(3))
        self.z0[:] = self.extract_z0(self.z)
        self.z_dot[:] = 0.0
        
        for i in range(self.n_meshes):
            self.z_dot[i * 12 + 9:i * 12 + 12] = np.array(
                [0.0, -4.0, 0.0]
            )

        self.z_tilde[:] = 0.0
        self.q_v0[:] = 0.0
        self.q_v_dot[:] = 0.0
        self.dq_v[:] = 0.0

        self.frame = 0
        # Base construction calls this override before the NQN members exist.
        if hasattr(self, "lbfgs_history"):
            self.lbfgs_history.clear()
            self.collision_hessian_seed = csc_matrix(
                (self.n_reduced, self.n_reduced)
            )
            self.previous_system_coordinates = None
            self.previous_collision_gradient = None
            self.nqn_newton_solves = 0
            self.nqn_lbfgs_solves = 0
            self.nqn_pairs = 0
        
        if from_frame > 0:
            self.frame = from_frame
            states = np.load(f"output/states/z_{self.frame}.npz")
            for (alias, field) in zip(self.fields_alias, self.z_fields):
                field[:] = states[alias]
            self.sync_local_coordinates()

        if hasattr(self, "ccd_safe_affine"):
            self.ccd_safe_affine = self.extract_z0(self.z).copy()
            self.ccd_safe_system = self.system_coordinates().copy()
            self.ccd_velocity_origin = None
            self.ccd_vabd_velocity_origin = None
            self.ccd_reintegrations_this_step = 0
            self.ccd_invocations = 0
            self.ccd_impacts = 0
            self.ccd_reintegrations = 0
            self.ccd_fallback_clamps = 0
            self.ccd_candidates = 0
            self.ccd_min_toi = 1.0
            self.ccd_checkpointed_frames.clear()
            self.first_ipc_checkpoint_saved = False
            self.line_search_calls = 0
            self.line_search_alpha_one = 0
            self.line_search_reduced = 0
            self.line_search_failures = 0
            self.first_iteration_line_search_failures = 0
            self.nonlinear_converged = 0
            self.nonlinear_iteration_limits = 0

            # self.states.x.assign((self.U @ self.z).reshape((-1, 3)))

    def save_states(self):
        os.makedirs("output/states", exist_ok=True)
        np.savez_compressed(f"output/states/z_{self.frame}.npz", **dict(zip(self.fields_alias, self.z_fields)))
        # # for alias, field in zip(self.fields_alias, self.z_fields):
        # #     np.save(f"output/states/{alias}_{self.frame}.npy", field)

    def save_preimpact_state(self, affine_origin):
        """Save the committed state immediately before a detected CCD impact."""
        os.makedirs("output/states", exist_ok=True)
        z_preimpact = self.z.copy()
        for body in range(self.n_meshes):
            start = body * self.n_modes
            z_preimpact[start:start + 12] = affine_origin[
                body * 12:(body + 1) * 12
            ]
        fields = dict(zip(self.fields_alias, self.z_fields))
        fields["z"] = z_preimpact
        np.savez_compressed(
            f"output/states/z_{self.frame - 1}.npz",
            **fields,
        )
            
    def collision_embedding_derivatives(self):
        """Build the constant linear-medial Jacobian in world coordinates."""
        affine_jacobians = []
        tilde_jacobians = []
        medial_start = 0
        tilde_width = self.n_modes - 12
        q_vibration = self.extract_q_vibration()
        for i in range(self.n_meshes):
            model = self.models[i]
            count = self.slabmeshes[model].nv
            medial_end = medial_start + count
            tilde_start = i * tilde_width
            tilde_end = tilde_start + tilde_width
            _, j_affine, j_tilde = body_embedding(
                self.V_medial_rest[medial_start:medial_end],
                self.diags_tilde_lhs[i],
                q_vibration[tilde_start:tilde_end],
                self.get_F(i),
            )
            affine_jacobians.append(j_affine)
            tilde_jacobians.append(j_tilde)
            medial_start = medial_end

        j_affine = block_diag(affine_jacobians, format="csc")
        j_tilde = block_diag(tilde_jacobians, format="csc")
        jacobian = hstack((j_affine, j_tilde), format="csc")
        return jacobian

    def cd(self): 
        if self.use_ipc_collision():
            if self.use_full_ipc_collision():
                V = self.linear_medial_positions(self.system_coordinates())
            else:
                V = self.homogeneous_medial_positions(self.extract_z0(self.z))
            self.collider_medial.collision_set(
                V,
                self.R_rest,
                activation_distance=medial_ccd_activation_distance,
            )
        else:
            V, R = self.get_VR()
            self.collider_medial.collision_set(V, R)

    def process_collision_medial_based(self):
        if self.use_ipc_collision():
            g, H, barrier_energy, constraint_count = self.collider_medial.analyze_ipc(
                medial_ccd_activation_distance
            )
            if (ccd_only_affine and barrier_energy > 0.0 and
                    not self.first_ipc_checkpoint_saved):
                self.save_preimpact_state(self.ccd_safe_affine)
                self.first_ipc_checkpoint_saved = True
            term = self.h * self.h * medial_collision_stiffness
            if self.use_full_ipc_collision():
                jacobian, _ = self.collision_embedding_derivatives(g)
                self.b_sys_col = np.asarray(
                    jacobian.T @ (g * term)
                ).reshape(-1)
                self.collision_hessian_seed = csc_matrix(
                    jacobian.T @ (H.tocsc() * term) @ jacobian
                )
                if self.collect_stability_stats:
                    self.collision_evaluations += 1
                    self.collision_constraints_total += constraint_count
                    self.max_collision_constraints = max(
                        self.max_collision_constraints, constraint_count
                    )
                    self.max_collision_energy = max(
                        self.max_collision_energy, barrier_energy
                    )
                    if constraint_count:
                        self.collision_frames.add(self.frame)
                return
            affine_gradient = np.asarray(self.Um0.T @ (g * term)).reshape(-1)
            affine_hessian = self.Um0.T @ (H.tocsc() * term) @ self.Um0
            affine_dim = self.n_meshes * 12
            self.b_sys_col = np.zeros(self.n_reduced)
            self.b_sys_col[:affine_dim] = affine_gradient
            self.collision_hessian_seed = block_diag(
                (
                    affine_hessian,
                    csc_matrix((self.n_reduced - affine_dim,
                                self.n_reduced - affine_dim)),
                ),
                format="csc",
            )
            if self.collect_stability_stats:
                self.collision_evaluations += 1
                self.collision_constraints_total += constraint_count
                self.max_collision_constraints = max(
                    self.max_collision_constraints, constraint_count
                )
                self.max_collision_energy = max(
                    self.max_collision_energy, barrier_energy
                )
                if constraint_count:
                    self.collision_frames.add(self.frame)
            return

        using_neural = self.neural_medial is not None
        with wp.ScopedTimer("analyze"):
            g, H, idx, rows, cols, values = self.collider_medial.analyze()

        collision_sets = [
            "pt_set", "ee_set", "g_set", "p_static_set",
            "p_static_set_cone", "ee_static_set",
        ]
        constraint_count = len(rows)
        if self.collect_stability_stats:
            constraint_count = sum(
                int(getattr(self.collider_medial, name).cnt.numpy()[0])
                for name in collision_sets if hasattr(self.collider_medial, name)
            )
            collision_energy = sum(
                float(getattr(self.collider_medial, name).E.numpy()[0])
                for name in collision_sets if hasattr(self.collider_medial, name)
            )
            self.collision_evaluations += 1
            self.collision_constraints_total += constraint_count
            self.max_collision_constraints = max(self.max_collision_constraints, constraint_count)
            self.max_collision_energy = max(self.max_collision_energy, collision_energy)
            if constraint_count:
                self.collision_frames.add(self.frame)
            
        term = self.h * self.h * medial_collision_stiffness

        if using_neural:
            with wp.ScopedTimer("neural reduced collision pullback"):
                if constraint_count == 0:
                    self.b_sys_col = np.zeros(self.n_reduced)
                    if self.use_newton():
                        self.collision_hessian_seed = csc_matrix(
                            (self.n_reduced, self.n_reduced)
                        )
                    return
                # Detection in cd() already used neural centers/radii. Only
                # evaluate the more expensive Jacobian when contacts exist.
                self.get_VR(with_jacobian=True)
                # Detection uses neural centers and radii.  The solve uses the
                # center Jacobian and a PSD Gauss-Newton pullback only; radius
                # derivatives and d2p/dq2 are deliberately omitted.
                Jc = self.neural_medial_center_jacobian
                self.b_sys_col = np.asarray(Jc.T @ (g * term)).reshape(-1)
                if self.use_newton():
                    self.collision_hessian_seed = csc_matrix(
                        Jc.T @ (H.tocsc() * term) @ Jc
                    )

        if not using_neural:
            with wp.ScopedTimer("batched warp collision pullback"):
                if self.warp_linear_pullback is not None:
                    self.b_sys_col, hessian = self.warp_linear_pullback.assemble(
                        g,
                        rows,
                        cols,
                        values,
                        term,
                        with_hessian=self.use_newton(),
                    )
                    if hessian is not None:
                        self.collision_hessian_seed = hessian
                else:
                    jacobian = self.collision_embedding_derivatives()
                    self.b_sys_col = jacobian.T @ (g * term)
                    if self.use_newton():
                        self.collision_hessian_seed = (
                            jacobian.T @ (H.tocsc() * term) @ jacobian
                        ).tocsc()
        
    def compute_collision_energy_medial_based(self):
        with wp.ScopedTimer("get V, R"):
            if self.use_ipc_collision():
                if self.use_full_ipc_collision():
                    V = self.linear_medial_positions(self.system_coordinates())
                else:
                    V = self.homogeneous_medial_positions(self.extract_z0(self.z))
                R = self.R_rest
            else:
                V, R = self.get_VR()
        h = self.h
        with wp.ScopedTimer("energy"):
            if self.use_ipc_collision():
                collision_energy = self.collider_medial.ipc_energy(
                    V, R, medial_ccd_activation_distance
                )
            else:
                collision_energy = self.collider_medial.energy(V, R)
            ret = collision_energy * medial_collision_stiffness * h * h
        return ret

    def compute_A(self):
        pass

    def compute_rhs(self):
        pass

def pyramid(from_frame=0, n_meshes=10, frames=None, quiet=False):
    model = "squishy"
    if not 1 <= n_meshes <= 356:
        raise ValueError("pyramid n_meshes must be between 1 and 356")
    meshes = [f"assets/{model}/{model}.tobj"] * n_meshes
    transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]

    positions = np.load("data/init_pos_356.npy")
    for i in range(n_meshes):
        transforms[i][:3, 3] = positions[i]
    
    # stacked bowls
    static_meshes_file = ["assets/bowl stack.obj"]
    scale = np.identity(4)
    static_bars = StaticScene(static_meshes_file, np.array([scale]))
    rods = MedialVABD(h, meshes, transforms, static_bars)

    if frames is not None:
        rods.collect_stability_stats = True
        rods.reset_z(from_frame)
        completed_frames = 0
        stability_status = "completed"
        with open(os.devnull, "w") as sink:
            for _ in range(frames):
                try:
                    if (stability_checkpoint_interval > 0 and
                            rods.frame % stability_checkpoint_interval == 0):
                        rods.save_states()
                    if quiet:
                        with contextlib.redirect_stdout(sink):
                            rods.step()
                    else:
                        rods.step()
                except SystemExit as exc:
                    stability_status = "failed_first_iteration_line_search"
                    break
                state_fields = (rods.z, rods.z_tilde, rods.z_dot, rods.q_v_dot)
                if not all(np.isfinite(field).all() for field in state_fields):
                    raise FloatingPointError(f"non-finite reduced state at frame {rods.frame}")
                completed_frames += 1
        state_fields = (rods.z, rods.z_tilde, rods.z_dot, rods.q_v_dot)
        max_abs_state = max(
            (float(np.max(np.abs(field))) for field in state_fields if field.size),
            default=0.0,
        )
        if ccd_only_affine or ccd_only_vabd:
            if ccd_only_vabd:
                final_coordinates = rods.system_coordinates()
                final_ccd_toi, final_ccd_candidates = \
                    rods.query_medial_position_ccd(
                        rods.linear_medial_positions(rods.ccd_safe_system),
                        rods.linear_medial_positions(final_coordinates),
                    )
            else:
                final_ccd_toi, final_ccd_candidates = rods.query_medial_ccd(
                    rods.ccd_safe_affine, rods.extract_z0(rods.z)
                )
        else:
            final_ccd_toi, final_ccd_candidates = 1.0, 0
        interpolation = "neural" if rods.neural_medial is not None else "linear"
        print(
            "PYRAMID_STABILITY "
            f"mode={'ccd_only_affine' if ccd_only_affine else ('ccd_only_vabd' if ccd_only_vabd else 'vabd')} "
            f"medial={interpolation} "
            f"split_lbfgs={str(split_lbfgs).lower()} "
            f"collision_mode={'ccd' if (ccd_only_affine or ccd_only_vabd) else 'dcd'} "
            f"status={stability_status} requested_frames={frames} "
            f"completed_frames={completed_frames} bodies={n_meshes} "
            f"final_frame={rods.frame} "
            f"contact_frames={len(rods.collision_frames)} "
            f"first_contact={min(rods.collision_frames) if rods.collision_frames else 'none'} "
            f"last_contact={max(rods.collision_frames) if rods.collision_frames else 'none'} "
            f"max_constraints={rods.max_collision_constraints} "
            f"collision_evaluations={rods.collision_evaluations} "
            f"max_collision_energy={rods.max_collision_energy:.9g} "
            f"newton_solves={rods.nqn_newton_solves} "
            f"lbfgs_solves={rods.nqn_lbfgs_solves} "
            f"lbfgs_pairs={rods.nqn_pairs} "
            f"dxslv_gpu_solves={rods.dxslv_solver.gpu_solves} "
            f"dxslv_refinements={rods.dxslv_solver.refinements} "
            f"dxslv_fallbacks={rods.dxslv_solver.fallbacks} "
            f"ccd_invocations={rods.ccd_invocations} "
            f"ccd_impacts={rods.ccd_impacts} "
            f"ccd_reintegrations={rods.ccd_reintegrations} "
            f"ccd_fallback_clamps={rods.ccd_fallback_clamps} "
            f"ccd_candidates={rods.ccd_candidates} "
            f"ccd_min_toi={rods.ccd_min_toi:.9g} "
            f"ccd_final_toi={float(final_ccd_toi):.9g} "
            f"ccd_final_candidates={int(final_ccd_candidates)} "
            f"nonlinear_converged={rods.nonlinear_converged} "
            f"iteration_limits={rods.nonlinear_iteration_limits} "
            f"line_search_calls={rods.line_search_calls} "
            f"alpha_one={rods.line_search_alpha_one} "
            f"alpha_reduced={rods.line_search_reduced} "
            f"line_search_failures={rods.line_search_failures} "
            f"first_iter_line_search_failures={rods.first_iteration_line_search_failures} "
            f"max_abs_state={max_abs_state:.9g} finite=true"
        )
        return rods

    if from_frame > 0:
        rods.reset_z(from_frame)
    viewer = MedialViewer(rods, static_bars)
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal vibrational affine-body dynamics demo")
    parser.add_argument("--bodies", type=int, default=10, help="prefix of the 356 pyramid bodies")
    parser.add_argument("--frames", type=int, help="headless frame count; omit for the viewer")
    parser.add_argument("--from-frame", type=int, default=0)
    parser.add_argument("--quiet", action="store_true", help="only print the headless stability summary")
    parser.add_argument("--linear-medial", action="store_true", help="disable the neural medial proxy")
    parser.add_argument("--no-split-lbfgs", action="store_true", help="use an exact Newton solve every iteration")
    parser.add_argument(
        "--ccd-only-affine", action="store_true",
        help="use affine DOFs with homogeneous medial CCD/IPC every timestep and no DCD",
    )
    parser.add_argument(
        "--ccd-only-vabd", action="store_true",
        help="use affine plus vibration DOFs with linear medial CCD/IPC every timestep and no DCD",
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=0,
        help="save restart states every N frames (defaults to 25 in pure CCD modes)",
    )
    args = parser.parse_args()
    if args.ccd_only_affine and args.ccd_only_vabd:
        parser.error("choose only one CCD-only mode")
    ccd_only_affine = args.ccd_only_affine
    ccd_only_vabd = args.ccd_only_vabd
    medial_reduced_module.affine_only_enabled = ccd_only_affine
    medial_reduced_module.neural_medial_enabled = (
        not args.linear_medial and not ccd_only_affine and not ccd_only_vabd
    )
    split_lbfgs = (
        not args.no_split_lbfgs and not ccd_only_affine and not ccd_only_vabd
    )
    stability_checkpoint_interval = args.checkpoint_interval
    if (ccd_only_affine or ccd_only_vabd) and stability_checkpoint_interval == 0:
        stability_checkpoint_interval = 25
    ps.init()
    # ps.set_ground_plane_mode("none")
    ps.set_ground_plane_height(-collision_eps)
    wp.config.max_unroll = 0
    wp.init()
    ps.look_at((0, 6, 15), (0, 6, 0))
    pyramid(args.from_frame, args.bodies, args.frames, args.quiet)
