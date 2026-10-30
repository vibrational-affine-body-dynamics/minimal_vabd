import warp as wp
from stretch import RodBCBase
from fem.interface import RodComplex
import numpy as np
from geometry.static_scene import StaticScene

omega = 3.0

class Timeit:
    def __init__(self):
        self.dict = {
            "compute A": [],
            "collision": [],              
            "assemble": [],
            "solve": [],
            "line search": [],
            "detection": [],
        }
        
    def reset(self):
        for k in self.dict.keys():
            self.dict[k] = 0.0


class RodComplexBC(RodBCBase, RodComplex):
    def __init__(self, h, meshes = [], transforms = [], static_meshes:StaticScene = None):
        self.meshes_filename = meshes 
        self.transforms = transforms
        super().__init__(h)
        self.static_meshes = static_meshes
        self.define_collider()
        self.n_pt = 0
        self.n_ee = 0
        self.n_ground = 0

        self.timeit = Timeit()
        self.tot_iters = []
        

    def reset(self):
        self.theta = 0.0
        self.frame = 0
        wp.copy(self.states.x, self.xcs)
        wp.copy(self.states.x0, self.xcs)
        

    def set_bc_fixed_hessian(self):
        pass

    def set_bc_fixed_grad(self):
        pass

    def step(self):
        self.theta += omega * self.h
        self.frame += 1
        with wp.ScopedTimer("step"):
            reintegrate = True
            step_iterations = 0
            while reintegrate:
                newton_iter = True
                self.n_iter = 0
                max_iter = 10
                while newton_iter:
                    with wp.ScopedTimer(f"newton #{self.n_iter}"):
                        with wp.ScopedTimer("compute A", dict = self.timeit.dict):
                            self.compute_A()

                        with wp.ScopedTimer("detection", dict = self.timeit.dict):
                            self.cd()
                        with wp.ScopedTimer("collision", dict = self.timeit.dict):
                            self.process_collision()

                        with wp.ScopedTimer("assemble", dict = self.timeit.dict):
                            self.compute_rhs()

                        with wp.ScopedTimer("solve", dict = self.timeit.dict):
                            self.solve()

                        did_converge = self.converged()
                        hit_iteration_limit = self.n_iter >= max_iter
                        newton_iter = not (did_converge or hit_iteration_limit)
                        if not newton_iter:
                            if hasattr(self, "nonlinear_converged"):
                                self.nonlinear_converged += int(did_converge)
                                self.nonlinear_iteration_limits += int(
                                    hit_iteration_limit and not did_converge
                                )
                            break
                        with wp.ScopedTimer("line search", dict = self.timeit.dict):
                            alpha = self.line_search()
                        if alpha == 0.0:
                            print("\nline search failed")
                            if self.n_iter == 0:
                                print("quitting: line search fails on the first iteration")
                                quit()
                            break

                        print(f"iter = {self.n_iter}, alpha = {alpha}")
                        self.n_iter += 1
                step_iterations += self.n_iter + 1
                reintegrate = self.prepare_ccd_reintegration()
            self.tot_iters.append(step_iterations)
            self.update_x0_xdot()
            
    def save_meta(self):
        tot_iters = np.array(self.tot_iters)
        tot = np.sum(tot_iters)
        print(f"total iterations = {tot}, total frames = {self.frame}")
        np.save("plot/metadata.npy", {"total_iters": tot_iters, "total_frames": self.frame})
        # np.save("timeit.npy", self.timeit.dict)
        np.savez("plot/timeit.npz", **self.timeit.dict)

    def converged(self):
        dxnp = self.states.dx.numpy()
        norm_dx = np.linalg.norm(dxnp)
        print(f"norm dx = {np.linalg.norm(dxnp)}")
        return norm_dx < 1e-3
