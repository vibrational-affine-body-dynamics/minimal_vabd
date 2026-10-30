"""World/object-frame reduced-coordinate transformations from Appendix A."""

import numpy as np
from scipy.sparse import csc_matrix


def local_vibration_and_jacobian(q_affine, q_vibration):
    """Return ``tilde(q)_V`` and ``d[q_A, tilde(q)_V]/d[q_A, q_V]``.

    Coordinates use the simulator ordering: all 12-DOF affine blocks first,
    followed by all vibrational blocks.  Every consecutive three vibration
    entries form one column transformed by the homogeneous affine matrix.
    This is Appendix A's local-to-global coordinate derivative, expressed in
    the conventional column-Jacobian orientation.
    """
    q_affine = np.asarray(q_affine)
    q_vibration = np.asarray(q_vibration)
    if q_affine.ndim != 1 or q_affine.size % 12:
        raise ValueError("q_affine must contain one 12-DOF block per body")
    n_bodies = q_affine.size // 12
    if q_vibration.ndim != 1 or q_vibration.size % n_bodies:
        raise ValueError("q_vibration must contain equally sized body blocks")
    vibration_width = q_vibration.size // n_bodies
    if vibration_width % 3:
        raise ValueError("each vibration block must contain 3-vectors")

    affine_dim = q_affine.size
    total_dim = affine_dim + q_vibration.size
    local = np.empty_like(q_vibration)
    rows = list(range(affine_dim))
    cols = list(range(affine_dim))
    values = [1.0] * affine_dim

    for body in range(n_bodies):
        affine_start = body * 12
        vibration_start = body * vibration_width
        matrix = q_affine[affine_start:affine_start + 9].reshape(3, 3).T
        inverse = np.linalg.inv(matrix)
        global_block = q_vibration[
            vibration_start:vibration_start + vibration_width
        ].reshape((-1, 3))
        local_block = global_block @ inverse.T
        local[vibration_start:vibration_start + vibration_width] = \
            local_block.reshape(-1)

        for vector, local_vector in enumerate(local_block):
            row_start = affine_dim + vibration_start + vector * 3
            column_start = affine_dim + vibration_start + vector * 3

            # d(A^-1 q_V)/dq_V = A^-1.
            for local_axis in range(3):
                for world_axis in range(3):
                    rows.append(row_start + local_axis)
                    cols.append(column_start + world_axis)
                    values.append(inverse[local_axis, world_axis])

            # d(A^-1 q_V)/dA_ab = -A^-1 e_a e_b^T tilde(q)_V.
            for world_axis in range(3):
                for local_axis in range(3):
                    affine_dof = affine_start + local_axis * 3 + world_axis
                    derivative = -inverse[:, world_axis] * local_vector[local_axis]
                    for output_axis in range(3):
                        rows.append(row_start + output_axis)
                        cols.append(affine_dof)
                        values.append(derivative[output_axis])

    jacobian = csc_matrix((values, (rows, cols)), shape=(total_dim, total_dim))
    return local, jacobian
