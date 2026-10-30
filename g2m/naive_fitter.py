"""SQEM fitting of a fixed medial topology to a deformed surface."""

import igl
import numpy as np
from sqem import Sqem


class Fitter:
    def __init__(self, vertices, tets, slabmesh):
        self.surface_F = igl.boundary_facets(tets)
        self.nv_surface = np.max(self.surface_F) + 1
        self.V = vertices
        self.slabmesh = slabmesh
        self.surface_V = self.V[:self.nv_surface]
        self.nv_surface = self.surface_V.shape[0]
        self.V_deform = np.copy(vertices)
        self.VF, self.NI = igl.vertex_triangle_adjacency(
            self.surface_F, self.surface_V.shape[0]
        )
        self.nearest = np.zeros(self.nv_surface, dtype=np.int32)
        self.fitted_V = slabmesh.V.copy()
        self.fitted_R = slabmesh.R.copy()
        self.assign_nearest()

    def assign_nearest(self):
        for i in range(self.nv_surface):
            vertex = self.V_deform[i]
            distance = np.abs(
                np.linalg.norm(self.slabmesh.V - vertex, axis=1) - self.slabmesh.R
            )
            self.nearest[i] = np.argmin(distance)

    def fit_spheres(self):
        threshold = 5
        self.N = -igl.per_face_normals(
            self.V_deform, self.surface_F, np.array([0.0, 0.0, 1.0])
        )
        for i in range(self.slabmesh.V.shape[0]):
            selected = self.nearest == i
            if np.count_nonzero(selected) < threshold:
                self.fitted_V[i] = self.slabmesh.V[i]
                self.fitted_R[i] = self.slabmesh.R[i]
            else:
                center, radius = self.sphere_from_patch(selected)
                self.fitted_V[i] = center
                self.fitted_R[i] = radius

    def sphere_from_patch(self, selected):
        accumulator = Sqem()
        accumulator.set_zero()
        for face in self.face_set_from_points(selected):
            point = self.V_deform[self.surface_F[face, 0]]
            accumulator = accumulator + Sqem(point, self.N[face])
        return accumulator.minimize(np.full(3, -10.0), np.full(3, 10.0))

    def V2p(self, vertices):
        self.V_deform[:] = vertices
        self.fit_spheres()
        return self.fitted_V, self.fitted_R

    def face_set_from_points(self, selected):
        faces = np.zeros(0, dtype=np.int32)
        for vertex in np.arange(self.nv_surface)[selected]:
            faces = np.concatenate((faces, self.VF[self.NI[vertex]:self.NI[vertex + 1]]))
        return faces
