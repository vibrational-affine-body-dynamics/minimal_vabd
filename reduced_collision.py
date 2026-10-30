"""Linear medial embedding in VABD world-frame reduced coordinates."""

import numpy as np
from scipy.sparse import csc_matrix


def body_embedding(
    rest_positions, interpolation, q_vibration, affine, translation=None
):
    """Return medial centers and ``dp/d[q_A, q_V]`` for one body.

    This is the algebraically simplified result of composing the object-frame
    embedding with Appendix A's ``d[q_A, q_tilde]/d[q_A, q_V]``.  The
    affine/vibration coupling cancels from the linear interpolation, leaving
    ``p = A p_rest + t + L q_V``.
    """
    rest_positions = np.asarray(rest_positions)
    interpolation = np.asarray(interpolation)
    q_vibration = np.asarray(q_vibration)
    affine = np.asarray(affine)
    if translation is None:
        translation = np.zeros(3)
    positions = (
        rest_positions @ affine.T
        + np.asarray(translation)
        + interpolation @ q_vibration.reshape((-1, 3))
    )
    rest_h = np.column_stack((rest_positions, np.ones(rest_positions.shape[0])))
    j_affine = np.kron(rest_h, np.eye(3))
    j_vibration = np.kron(interpolation, np.eye(3))
    return positions, csc_matrix(j_affine), csc_matrix(j_vibration)
