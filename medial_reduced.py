import warp as wp
import numpy as np
import os
from stretch import Triplets
from mesh_complex import RodComplexBC
from warp.sparse import bsr_set_from_triplets, bsr_zeros, bsr_transposed
from scipy.sparse import block_diag, hstack
from g2m.collision_medial import MedialCollisionDetector
from g2m.medial import SlabMesh
from g2m.bary_centric import TetBaryCentricCompute
from g2m.neural_medial import SquishyNeuralMedial
neural_medial_enabled = True
affine_only_enabled = False
padding = True
def vec(t):
    return (t.T).reshape(-1)

@wp.kernel
def fill_U_triplets(mesh_id: int, offset: int, xcs: wp.array(dtype = wp.vec3), W: wp.array2d(dtype = float), triplets: Triplets):
    '''
    offset = mesh_id * xx for repeating meshes
    '''
    i, j, k = wp.tid()
    xx = W.shape[0]
    yy = W.shape[1]
    # block_nnz = 4 * xx * yy
    

    xid = i + offset
    idx = (xid * yy + j) * 4 + k #block_nnz * mesh_id
    triplets.rows[idx] = xid
    triplets.cols[idx] = j * 4 + k + mesh_id * yy * 4
    c = float(1.0)
    if k < 3:
        c = xcs[xid][k]
    triplets.vals[idx] = wp.diag(wp.vec3(W[i, j] * c))


class MedialRodComplex(RodComplexBC):
    def __init__(self, h, meshes=[], transforms=[], static_meshes = None):
        self.models = [mesh.split("/")[1].split(".")[0] for mesh in meshes]
        self.model_set = set(self.models)
        # model = meshes[0].split("/")[1].split(".")[0]
        d = []
        for model in self.model_set:
            Q = self.load_Q(model)
            d.append((model, Q))
        self.Q = dict(d)
            
        self.define_z(transforms)
        super().__init__(h, meshes, transforms, static_meshes)
        self.define_encoder()

        self.define_U()
        self.compute_Um()

    def lbs_matrix(self, V, W):
        nvm = V.shape[0]
        v1 = np.ones((nvm, 4))
        v1[:, :3] = V
        lhs = np.hstack([W[:, j: j + 1] * v1 for j in range(W.shape[1])])
        return np.kron(lhs, np.identity(3))

    def define_U(self):
        self.U = np.zeros((self.n_nodes * 3, self.n_reduced))
        x0 = self.xcs.numpy()
        
        start = 0
        for i in range(self.n_meshes):
            Q = self.Q[self.models[i]]
            mesh_nodes = Q.shape[0]
            start_nxt = start + mesh_nodes
            xi = x0[start: start_nxt]
            Ui = self.lbs_matrix(xi, Q)
            self.U[start * 3: start_nxt * 3, i * self.n_modes: (i + 1) * self.n_modes] = Ui
            start = start_nxt
        
        self.wp_define_U()
        self.UwpT = bsr_transposed(self.Uwp)

    def wp_define_U(self):
        self.Uwp = bsr_zeros(self.n_nodes, self.n_modes // 3 * self.n_meshes, wp.mat33)
        triplets = Triplets()
        nnz = self.n_nodes * 4 * self.n_modes
        triplets.cols = wp.zeros((nnz, ), int)
        triplets.rows = wp.zeros((nnz, ), int)
        triplets.vals = wp.zeros((nnz,), wp.mat33)

        start = 0
        for i in range(self.n_meshes):
            Q = self.Q[self.models[i]]
            mesh_nodes = Q.shape[0]
            q = wp.array(Q, dtype = float)
            wp.launch(fill_U_triplets, (q.shape[0], q.shape[1], 4), inputs = [i, start, self.geo.xcs, q, triplets])
            start += mesh_nodes
        bsr_set_from_triplets(self.Uwp, triplets.rows, triplets.cols, triplets.vals, )

    def load_Q(self, model):
        Q = np.load(f"data/W_{model}.npy")
        Q = Q[:, :1 if affine_only_enabled else 10]
        Q[:, 0] = 1.0
        return Q
        
    def define_z(self, transforms):
        Q = self.Q[self.models[0]]
        self.n_modes = Q.shape[1] * 12 
        self.n_meshes = len(transforms)

        self.n_reduced = self.n_modes * self.n_meshes
        self.z = np.zeros(self.n_reduced)
        for i in range(self.n_meshes):
            self.z[i * self.n_modes: i * self.n_modes + 9] = vec(np.identity(3))
        
        self.dz = np.zeros_like(self.z)


    def define_collider(self):
        self.define_medials()
    
    def define_medials(self):        
        self.slabmeshes = dict([(model, SlabMesh(f"assets/{model}/ma/{model}.ma")) for model in self.model_set])
        
        
        R = np.zeros(0, float)
        E = np.zeros((0, 2), int)
        F = np.zeros((0, 3), int)
        V = np.zeros((0, 3))
        body = []

        cnt = 0
        if padding: 
            v_cnt = [self.slabmeshes[model].nv for model in self.model_set]
            self.v_cnt_max = np.max(np.array(v_cnt))
            print(f"\npadding : max v cnt = {self.v_cnt_max}\n") 

        for i in range(self.n_meshes):
            slabmesh = self.slabmeshes[self.models[i]]
            V0 = np.copy(slabmesh.V)
            v4 = np.ones((V0.shape[0], 4))
            v4[:, :3] = V0
            R0 = slabmesh.R
            E0 = slabmesh.E
            F0 = slabmesh.F

            Vi = (v4 @ self.transforms[i].T)[:, : 3]
            J3 = np.abs(np.linalg.det(self.transforms[i][:3, :3]))
            J = J3 ** (1./3.)
            Ri = np.copy(R0) * J
            if padding:
                Vii = np.zeros((self.v_cnt_max, 3), float)
                Vii[:Vi.shape[0], :] = Vi
                Vi = Vii 

                Rii = np.ones((self.v_cnt_max, ), float) * -1
                Rii[:Ri.shape[0]] = Ri
                Ri = Rii 
                
            V = np.vstack([V, Vi])
            # R = np.concatenate([R, np.copy(R0) * J])
            R = np.concatenate([R, Ri])
            E = np.vstack((E, E0 + cnt))
            F = np.vstack((F, F0 + cnt))

            if not padding:
                body += [i] * slabmesh.nv
                cnt += slabmesh.nv
            else:
                body += [i] * self.v_cnt_max
                cnt += self.v_cnt_max
        

        self.F_medial = F
        self.E_medial = E
        self.V_medial_rest = np.copy(V)
        self.V_medial = np.zeros_like(V)
        self.R_rest = np.copy(R)
        self.R = np.zeros_like(self.R_rest)
        self.body_medial = np.array(body, int)

        self.V_medial[:] = self.V_medial_rest
        self.R[:] = self.R_rest

        self.collider_medial = MedialCollisionDetector(
            self.V_medial, self.R_rest, self.E_medial, self.F_medial, self.body_medial, dense = False, ground = 0.0, static_objects = self.static_meshes)

        self.n_medial = self.V_medial.shape[0]

    def reset(self):
        if hasattr(self, "V_medial"):
            self.V_medial[:] = self.V_medial_rest[:]
            self.R[:] = self.R_rest
        super().reset()
        self.reset_z()

    def reset_z(self):
        t = self.transforms[0]
        self.z[:] = 0.0
        self.dz[:] = 0.0
        for i in range(self.n_meshes):
            self.z[self.n_modes * i: self.n_modes * i + 9] = vec(np.identity(3))

    def compute_Um(self):

        # jac = self.encoder.jacobian(x)
        diags = []
        start = 0
        for i in range(self.n_meshes):
            model = self.models[i]
            nv = self.slabmeshes[model].nv

            Vmi = self.V_medial_rest[start: start + nv]
            jaci = self.lbs_matrix(Vmi, self.W_medial[model])
            if padding: 
                start += self.v_cnt_max
                jacii = np.zeros((self.v_cnt_max * 3, self.n_modes), float)
                jacii[:nv * 3, :] = jaci
                jaci = jacii 
            else:
                start += nv
            diags.append(jaci)
        self.Um = block_diag(diags, "csr")


    def define_encoder(self):
        W_medial_list = []
        for model in self.model_set:
            Q = self.Q[model]
            if os.path.exists(f"data/W_medial_{model}.npy") and Q.shape[0] > 300:
                nc = Q.shape[1]
                weight = np.load(f"data/W_medial_{model}.npy")[:, :nc]
            else: 
                with wp.ScopedTimer(f"define_encoder_{model}"):
                    intp = TetBaryCentricCompute(model)
                with wp.ScopedTimer(f"compute_weight_{model}"):
                    weight = intp.compute_weight(Q)
            W_medial_list.append((model, weight))
        self.W_medial = dict(W_medial_list)
        checkpoint = "data/squishy_neural_medial.pth"
        self.neural_medial = None
        if neural_medial_enabled and self.model_set == {"squishy"} and os.path.exists(checkpoint):
            self.neural_medial = SquishyNeuralMedial(checkpoint)
            expected = self.slabmeshes["squishy"].nv
            if self.neural_medial.n_medials != expected:
                raise RuntimeError(
                    f"neural medial checkpoint has {self.neural_medial.n_medials} spheres; expected {expected}"
                )
            print(f"using approved squishy neural MAT: {self.neural_medial.metrics}")

    def get_VR(self, with_jacobian=False):
        if self.neural_medial is not None:
            local_to_global = self.sync_local_coordinates(
                with_jacobian=with_jacobian
            )
            V = np.copy(self.V_medial_rest)
            R = np.copy(self.R_rest)
            affine_jacobians = []
            tilde_jacobians = []
            vibration_dim = self.n_modes - 12
            q_affine = np.vstack([
                self.z[i * self.n_modes:i * self.n_modes + 12]
                for i in range(self.n_meshes)
            ])
            z_tilde = self.z_tilde.reshape(self.n_meshes, vibration_dim)
            if with_jacobian:
                predictions, body_jacobians = self.neural_medial.evaluate(
                    q_affine, z_tilde, self.transforms, with_jacobian=True
                )
            else:
                predictions = self.neural_medial.evaluate(q_affine, z_tilde, self.transforms)
            medial_start = 0
            for i, model in enumerate(self.models):
                count = self.slabmeshes[model].nv
                value = predictions[i]
                V[medial_start:medial_start + count] = value[:, :3]
                R[medial_start:medial_start + count] = value[:, 3]
                if with_jacobian:
                    body_jacobian = body_jacobians[i]
                    if padding and count < self.v_cnt_max:
                        padded = np.zeros((self.v_cnt_max * 3, self.n_modes))
                        padded[:count * 3] = body_jacobian
                        body_jacobian = padded
                    affine_jacobians.append(body_jacobian[:, :12])
                    tilde_jacobians.append(body_jacobian[:, 12:])
                medial_start += self.v_cnt_max if padding else count
            if with_jacobian:
                local_jacobian = hstack((
                    block_diag(affine_jacobians, format="csc"),
                    block_diag(tilde_jacobians, format="csc"),
                ), format="csc")
                self.neural_medial_center_jacobian = (
                    local_jacobian @ local_to_global
                ).tocsc()
            return V, R
        warp_pullback = getattr(self, "warp_linear_pullback", None)
        if warp_pullback is not None:
            translations = np.vstack([
                self.z[i * self.n_modes + 9:i * self.n_modes + 12]
                for i in range(self.n_meshes)
            ])
            V = warp_pullback.update_embedding(
                self.get_F_batch(), translations, self.extract_q_vibration()
            )
            return V, self.R_rest
        V = (self.Um @ self.z).reshape((-1, 3))# + self.V_medial_rest
        # V = self.V_medial_rest
        R = self.R_rest
        return V, R
