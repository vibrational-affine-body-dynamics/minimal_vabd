"""Torch-free Warp inference and analytic center Jacobians for neural MAT."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import warp as wp


INPUT_DIM = 36
TILDE_DIM = 108
LOCAL_DIM = 120


@wp.func
def quat_rotate(q: wp.vec4, p: wp.vec3):
    x, y, z, w = q[0], q[1], q[2], q[3]
    return wp.vec3(
        (1.0 - 2.0 * (y*y + z*z)) * p[0] + 2.0 * (x*y - w*z) * p[1] + 2.0 * (x*z + w*y) * p[2],
        2.0 * (x*y + w*z) * p[0] + (1.0 - 2.0 * (x*x + z*z)) * p[1] + 2.0 * (y*z - w*x) * p[2],
        2.0 * (x*z - w*y) * p[0] + 2.0 * (y*z + w*x) * p[1] + (1.0 - 2.0 * (x*x + y*y)) * p[2],
    )


@wp.kernel
def modes_forward(z_tilde: wp.array(dtype=float), quaternions: wp.array(dtype=float)):
    body, mode = wp.tid()
    start = body * TILDE_DIM + mode * 12
    s = wp.max(4.0 + z_tilde[start] + z_tilde[start + 4] + z_tilde[start + 8], 1.0e-8)
    w = 0.5 * wp.sqrt(s)
    denominator = wp.max(4.0 * w, 1.0e-6)
    raw = wp.vec4(
        (z_tilde[start + 5] - z_tilde[start + 7]) / denominator,
        (z_tilde[start + 6] - z_tilde[start + 2]) / denominator,
        (z_tilde[start + 1] - z_tilde[start + 3]) / denominator,
        w,
    )
    q = wp.normalize(raw)
    qstart = body * INPUT_DIM + mode * 4
    for component in range(4):
        quaternions[qstart + component] = q[component]


@wp.kernel
def modes_jacobian(z_tilde: wp.array(dtype=float), quaternions: wp.array(dtype=float), quaternion_jacobian: wp.array(dtype=float)):
    body, mode, local_input = wp.tid()
    start = body * TILDE_DIM + mode * 12
    s = wp.max(4.0 + z_tilde[start] + z_tilde[start + 4] + z_tilde[start + 8], 1.0e-8)
    w = 0.5 * wp.sqrt(s)
    denominator = wp.max(4.0 * w, 1.0e-6)
    nx = z_tilde[start + 5] - z_tilde[start + 7]
    ny = z_tilde[start + 6] - z_tilde[start + 2]
    nz = z_tilde[start + 1] - z_tilde[start + 3]
    raw = wp.vec4(nx / denominator, ny / denominator, nz / denominator, w)
    raw_norm = wp.max(wp.length(raw), 1.0e-8)
    qstart = body * INPUT_DIM + mode * 4
    q = wp.vec4(quaternions[qstart], quaternions[qstart + 1], quaternions[qstart + 2], quaternions[qstart + 3])
    dw = float(0.0)
    if local_input == 0 or local_input == 4 or local_input == 8:
        dw = 1.0 / wp.max(8.0 * w, 1.0e-8)
    dnx = float(0.0)
    dny = float(0.0)
    dnz = float(0.0)
    if local_input == 5:
        dnx = 1.0
    elif local_input == 7:
        dnx = -1.0
    elif local_input == 6:
        dny = 1.0
    elif local_input == 2:
        dny = -1.0
    elif local_input == 1:
        dnz = 1.0
    elif local_input == 3:
        dnz = -1.0
    ddenominator = 4.0 * dw
    denominator2 = denominator * denominator
    du = wp.vec4(
        (dnx * denominator - nx * ddenominator) / denominator2,
        (dny * denominator - ny * ddenominator) / denominator2,
        (dnz * denominator - nz * ddenominator) / denominator2,
        dw,
    )
    dq = (du - q * wp.dot(q, du)) / raw_norm
    global_input = mode * 12 + local_input
    for component in range(4):
        quaternion_jacobian[(qstart + component) * TILDE_DIM + global_input] = dq[component]


@wp.kernel
def normalize_input_forward(q: wp.array(dtype=float), mean: wp.array(dtype=float), scale: wp.array(dtype=float), x: wp.array(dtype=float)):
    body, component = wp.tid()
    index = body * INPUT_DIM + component
    x[index] = (q[index] - mean[component]) / scale[component]


@wp.kernel
def normalize_input(q: wp.array(dtype=float), q_jacobian: wp.array(dtype=float), mean: wp.array(dtype=float), scale: wp.array(dtype=float), x: wp.array(dtype=float), x_jacobian: wp.array(dtype=float)):
    body, component = wp.tid()
    index = body * INPUT_DIM + component
    x[index] = (q[index] - mean[component]) / scale[component]
    for k in range(TILDE_DIM):
        x_jacobian[index * TILDE_DIM + k] = q_jacobian[index * TILDE_DIM + k] / scale[component]


@wp.kernel
def dense_forward(values_in: wp.array(dtype=float), weight: wp.array(dtype=float), bias: wp.array(dtype=float), values_out: wp.array(dtype=float), in_dim: int, out_dim: int, sigmoid: bool):
    body, row = wp.tid()
    value = bias[row]
    for column in range(in_dim):
        value += weight[row * in_dim + column] * values_in[body * in_dim + column]
    if sigmoid:
        value = 1.0 / (1.0 + wp.exp(-value))
    values_out[body * out_dim + row] = value


@wp.kernel
def dense_jacobian(jacobian_in: wp.array(dtype=float), weight: wp.array(dtype=float), values_out: wp.array(dtype=float), jacobian_out: wp.array(dtype=float), in_dim: int, out_dim: int, sigmoid: bool):
    body, row, column = wp.tid()
    value = float(0.0)
    for inner in range(in_dim):
        value += weight[row * in_dim + inner] * jacobian_in[(body * in_dim + inner) * TILDE_DIM + column]
    if sigmoid:
        activation = values_out[body * out_dim + row]
        value *= activation * (1.0 - activation)
    jacobian_out[(body * out_dim + row) * TILDE_DIM + column] = value


@wp.kernel
def output_center_jacobian(hidden_jacobian: wp.array(dtype=float), weight: wp.array(dtype=float), output_scale: wp.array(dtype=float), center_jacobian: wp.array(dtype=float), width: int, n_medials: int):
    body, medial, axis, column = wp.tid()
    output = medial * 4 + axis
    value = float(0.0)
    for inner in range(width):
        value += weight[output * width + inner] * hidden_jacobian[(body * width + inner) * TILDE_DIM + column]
    center_jacobian[((body * n_medials + medial) * 3 + axis) * TILDE_DIM + column] = value * output_scale[output]


@wp.kernel
def local_forward(q: wp.array(dtype=float), network_output: wp.array(dtype=float), output_mean: wp.array(dtype=float), output_scale: wp.array(dtype=float), dqs_weights: wp.array(dtype=float), rest_medials: wp.array(dtype=float), local_medials: wp.array(dtype=float), n_medials: int):
    body, medial = wp.tid()
    blended = wp.vec4(0.0, 0.0, 0.0, 0.0)
    for mode in range(9):
        weight = dqs_weights[medial * 18 + mode]
        qstart = body * INPUT_DIM + mode * 4
        blended += weight * wp.vec4(q[qstart], q[qstart + 1], q[qstart + 2], q[qstart + 3])
        blended[3] += dqs_weights[medial * 18 + 9 + mode]
    blended = wp.normalize(blended)
    rest = wp.vec3(rest_medials[medial * 4], rest_medials[medial * 4 + 1], rest_medials[medial * 4 + 2])
    center = quat_rotate(blended, rest)
    out_start = body * n_medials * 4 + medial * 4
    for axis in range(3):
        output_index = medial * 4 + axis
        local_medials[out_start + axis] = center[axis] + network_output[body * n_medials * 4 + output_index] * output_scale[output_index] + output_mean[output_index]
    radius_index = medial * 4 + 3
    local_medials[out_start + 3] = rest_medials[radius_index] + network_output[body * n_medials * 4 + radius_index] * output_scale[radius_index] + output_mean[radius_index]


@wp.kernel
def local_center_jacobian(q: wp.array(dtype=float), q_jacobian: wp.array(dtype=float), network_center_jacobian: wp.array(dtype=float), dqs_weights: wp.array(dtype=float), rest_medials: wp.array(dtype=float), local_jacobian: wp.array(dtype=float), n_medials: int):
    body, medial, axis, column = wp.tid()
    blended_raw = wp.vec4(0.0, 0.0, 0.0, 0.0)
    for transform_mode in range(9):
        weight0 = dqs_weights[medial * 18 + transform_mode]
        qindex = body * INPUT_DIM + transform_mode * 4
        blended_raw += weight0 * wp.vec4(q[qindex], q[qindex + 1], q[qindex + 2], q[qindex + 3])
        blended_raw[3] += dqs_weights[medial * 18 + 9 + transform_mode]
    blend_norm = wp.max(wp.length(blended_raw), 1.0e-8)
    rotation = blended_raw / blend_norm
    mode = column // 12
    qstart = body * INPUT_DIM + mode * 4
    weight = dqs_weights[medial * 18 + mode]
    ds = weight * wp.vec4(q_jacobian[qstart * TILDE_DIM + column], q_jacobian[(qstart + 1) * TILDE_DIM + column], q_jacobian[(qstart + 2) * TILDE_DIM + column], q_jacobian[(qstart + 3) * TILDE_DIM + column])
    dr = (ds - rotation * wp.dot(rotation, ds)) / blend_norm
    x, y, z, w = rotation[0], rotation[1], rotation[2], rotation[3]
    px, py, pz = rest_medials[medial * 4], rest_medials[medial * 4 + 1], rest_medials[medial * 4 + 2]
    derivative_x = wp.vec3(2.0*y*py + 2.0*z*pz, -4.0*y*px + 2.0*x*py + 2.0*w*pz, -4.0*z*px - 2.0*w*py + 2.0*x*pz)
    derivative_y = wp.vec3(2.0*y*px - 4.0*x*py - 2.0*w*pz, 2.0*x*px + 2.0*z*pz, 2.0*w*px - 4.0*z*py + 2.0*y*pz)
    derivative_z = wp.vec3(2.0*z*px + 2.0*w*py - 4.0*x*pz, -2.0*w*px + 2.0*z*py - 4.0*y*pz, 2.0*x*px + 2.0*y*py)
    derivative_w = wp.vec3(-2.0*z*py + 2.0*y*pz, 2.0*z*px - 2.0*x*pz, -2.0*y*px + 2.0*x*py)
    baseline = float(0.0)
    if axis == 0:
        baseline = wp.dot(derivative_x, wp.vec3(dr[0], dr[1], dr[2])) + derivative_w[0] * dr[3]
    elif axis == 1:
        baseline = wp.dot(derivative_y, wp.vec3(dr[0], dr[1], dr[2])) + derivative_w[1] * dr[3]
    else:
        baseline = wp.dot(derivative_z, wp.vec3(dr[0], dr[1], dr[2])) + derivative_w[2] * dr[3]
    index = ((body * n_medials + medial) * 3 + axis) * TILDE_DIM + column
    local_jacobian[index] = baseline + network_center_jacobian[index]


@wp.func
def determinant3(transform: wp.array(dtype=float), start: int):
    return transform[start] * (transform[start + 5] * transform[start + 10] - transform[start + 6] * transform[start + 9]) - transform[start + 1] * (transform[start + 4] * transform[start + 10] - transform[start + 6] * transform[start + 8]) + transform[start + 2] * (transform[start + 4] * transform[start + 9] - transform[start + 5] * transform[start + 8])


@wp.kernel
def world_forward(q_affine: wp.array(dtype=float), transforms: wp.array(dtype=float), local_medials: wp.array(dtype=float), world_medials: wp.array(dtype=float), n_medials: int):
    body, medial = wp.tid()
    t = body * 16
    local_start = (body * n_medials + medial) * 4
    local = wp.vec3(local_medials[local_start], local_medials[local_start + 1], local_medials[local_start + 2])
    base = wp.vec3(
        transforms[t] * local[0] + transforms[t + 1] * local[1] + transforms[t + 2] * local[2] + transforms[t + 3],
        transforms[t + 4] * local[0] + transforms[t + 5] * local[1] + transforms[t + 6] * local[2] + transforms[t + 7],
        transforms[t + 8] * local[0] + transforms[t + 9] * local[1] + transforms[t + 10] * local[2] + transforms[t + 11],
    )
    a = body * 12
    for axis in range(3):
        world_medials[local_start + axis] = q_affine[a + axis] * base[0] + q_affine[a + 3 + axis] * base[1] + q_affine[a + 6 + axis] * base[2] + q_affine[a + 9 + axis]
    world_medials[local_start + 3] = local_medials[local_start + 3] * wp.pow(wp.abs(determinant3(transforms, t)), 1.0 / 3.0)


@wp.kernel
def world_center_jacobian(q_affine: wp.array(dtype=float), transforms: wp.array(dtype=float), local_medials: wp.array(dtype=float), local_jacobian: wp.array(dtype=float), world_jacobian: wp.array(dtype=float), n_medials: int):
    body, medial, axis, column = wp.tid()
    t = body * 16
    local_start = (body * n_medials + medial) * 4
    local = wp.vec3(local_medials[local_start], local_medials[local_start + 1], local_medials[local_start + 2])
    base = wp.vec3(
        transforms[t] * local[0] + transforms[t + 1] * local[1] + transforms[t + 2] * local[2] + transforms[t + 3],
        transforms[t + 4] * local[0] + transforms[t + 5] * local[1] + transforms[t + 6] * local[2] + transforms[t + 7],
        transforms[t + 8] * local[0] + transforms[t + 9] * local[1] + transforms[t + 10] * local[2] + transforms[t + 11],
    )
    value = float(0.0)
    if column < 12:
        if column == 9 + axis:
            value = 1.0
        else:
            local_axis = column // 3
            world_axis = column % 3
            if local_axis < 3 and world_axis == axis:
                value = base[local_axis]
    else:
        zcolumn = column - 12
        dl = wp.vec3(
            local_jacobian[((body * n_medials + medial) * 3) * TILDE_DIM + zcolumn],
            local_jacobian[((body * n_medials + medial) * 3 + 1) * TILDE_DIM + zcolumn],
            local_jacobian[((body * n_medials + medial) * 3 + 2) * TILDE_DIM + zcolumn],
        )
        db = wp.vec3(
            transforms[t] * dl[0] + transforms[t + 1] * dl[1] + transforms[t + 2] * dl[2],
            transforms[t + 4] * dl[0] + transforms[t + 5] * dl[1] + transforms[t + 6] * dl[2],
            transforms[t + 8] * dl[0] + transforms[t + 9] * dl[1] + transforms[t + 10] * dl[2],
        )
        a = body * 12
        value = q_affine[a + axis] * db[0] + q_affine[a + 3 + axis] * db[1] + q_affine[a + 6 + axis] * db[2]
    world_jacobian[((body * n_medials + medial) * 3 + axis) * LOCAL_DIM + column] = value


class SquishyNeuralMedial:
    """Batched Warp forward/Jacobian evaluator for an approved exported checkpoint."""

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        checkpoint = Path(checkpoint)
        data = np.load(checkpoint.with_name(checkpoint.stem + "_warp.npz"))
        metrics_payload = json.loads(checkpoint.with_suffix(".json").read_text())
        if not metrics_payload.get("approved", False):
            raise RuntimeError(f"refusing unapproved neural medial checkpoint: {checkpoint}")
        self.n_medials = int(data["n_medials"])
        self.width = int(data["width"])
        self.metrics = {k: v for k, v in metrics_payload.items() if k != "approved"}
        self.device = device

        def array(name):
            return wp.array(np.ascontiguousarray(data[name]).reshape(-1), dtype=float, device=device)

        self.input_mean, self.input_scale = array("input_mean"), array("input_scale")
        self.output_mean, self.output_scale = array("output_mean"), array("output_scale")
        self.dqs_weights, self.rest_medials = array("dqs_weights"), array("rest_medials")
        self.weights = [array(f"network_{index}_weight") for index in (0, 2, 4, 6)]
        self.biases = [array(f"network_{index}_bias") for index in (0, 2, 4, 6)]
        self.capacity = 0

    def _ensure_capacity(self, bodies: int) -> None:
        if bodies <= self.capacity:
            return
        self.capacity = bodies
        b, m, w = bodies, self.n_medials, self.width
        self.q_affine = wp.zeros(b * 12, dtype=float, device=self.device)
        self.z_tilde = wp.zeros(b * TILDE_DIM, dtype=float, device=self.device)
        self.transforms = wp.zeros(b * 16, dtype=float, device=self.device)
        self.q = wp.zeros(b * INPUT_DIM, dtype=float, device=self.device)
        self.q_jacobian = wp.zeros(b * INPUT_DIM * TILDE_DIM, dtype=float, device=self.device)
        self.x = wp.zeros(b * INPUT_DIM, dtype=float, device=self.device)
        self.x_jacobian = wp.zeros(b * INPUT_DIM * TILDE_DIM, dtype=float, device=self.device)
        self.hidden = [wp.zeros(b * w, dtype=float, device=self.device) for _ in range(3)]
        self.hidden_jacobian = [wp.zeros(b * w * TILDE_DIM, dtype=float, device=self.device) for _ in range(3)]
        self.network_output = wp.zeros(b * m * 4, dtype=float, device=self.device)
        self.network_center_jacobian = wp.zeros(b * m * 3 * TILDE_DIM, dtype=float, device=self.device)
        self.local_medials = wp.zeros(b * m * 4, dtype=float, device=self.device)
        self.local_jacobian = wp.zeros(b * m * 3 * TILDE_DIM, dtype=float, device=self.device)
        self.world_medials = wp.zeros(b * m * 4, dtype=float, device=self.device)
        self.world_jacobian = wp.zeros(b * m * 3 * LOCAL_DIM, dtype=float, device=self.device)

    def evaluate(self, q_affine, z_tilde, transforms, with_jacobian=False):
        q_affine = np.ascontiguousarray(q_affine, dtype=np.float32).reshape(-1, 12)
        z_tilde = np.ascontiguousarray(z_tilde, dtype=np.float32).reshape(-1, TILDE_DIM)
        transforms = np.ascontiguousarray(transforms, dtype=np.float32).reshape(-1, 4, 4)
        bodies = len(q_affine)
        self._ensure_capacity(bodies)
        self.q_affine.assign(q_affine.reshape(-1))
        self.z_tilde.assign(z_tilde.reshape(-1))
        self.transforms.assign(transforms.reshape(-1))
        wp.launch(modes_forward, (bodies, 9), [self.z_tilde, self.q], device=self.device)
        wp.launch(normalize_input_forward, (bodies, INPUT_DIM), [self.q, self.input_mean, self.input_scale, self.x], device=self.device)
        previous, in_dim = self.x, INPUT_DIM
        for layer in range(3):
            wp.launch(dense_forward, (bodies, self.width), [previous, self.weights[layer], self.biases[layer], self.hidden[layer], in_dim, self.width, True], device=self.device)
            previous, in_dim = self.hidden[layer], self.width
        wp.launch(dense_forward, (bodies, self.n_medials * 4), [previous, self.weights[3], self.biases[3], self.network_output, self.width, self.n_medials * 4, False], device=self.device)
        wp.launch(local_forward, (bodies, self.n_medials), [self.q, self.network_output, self.output_mean, self.output_scale, self.dqs_weights, self.rest_medials, self.local_medials, self.n_medials], device=self.device)
        wp.launch(world_forward, (bodies, self.n_medials), [self.q_affine, self.transforms, self.local_medials, self.world_medials, self.n_medials], device=self.device)
        values = self.world_medials.numpy()[:bodies * self.n_medials * 4].reshape(bodies, self.n_medials, 4)
        if not with_jacobian:
            return values

        self.q_jacobian.zero_()
        wp.launch(modes_jacobian, (bodies, 9, 12), [self.z_tilde, self.q, self.q_jacobian], device=self.device)
        wp.launch(normalize_input, (bodies, INPUT_DIM), [self.q, self.q_jacobian, self.input_mean, self.input_scale, self.x, self.x_jacobian], device=self.device)
        previous_jacobian, in_dim = self.x_jacobian, INPUT_DIM
        for layer in range(3):
            wp.launch(dense_jacobian, (bodies, self.width, TILDE_DIM), [previous_jacobian, self.weights[layer], self.hidden[layer], self.hidden_jacobian[layer], in_dim, self.width, True], device=self.device)
            previous_jacobian, in_dim = self.hidden_jacobian[layer], self.width
        wp.launch(output_center_jacobian, (bodies, self.n_medials, 3, TILDE_DIM), [previous_jacobian, self.weights[3], self.output_scale, self.network_center_jacobian, self.width, self.n_medials], device=self.device)
        wp.launch(local_center_jacobian, (bodies, self.n_medials, 3, TILDE_DIM), [self.q, self.q_jacobian, self.network_center_jacobian, self.dqs_weights, self.rest_medials, self.local_jacobian, self.n_medials], device=self.device)
        wp.launch(world_center_jacobian, (bodies, self.n_medials, 3, LOCAL_DIM), [self.q_affine, self.transforms, self.local_medials, self.local_jacobian, self.world_jacobian, self.n_medials], device=self.device)
        jacobian = self.world_jacobian.numpy()[:bodies * self.n_medials * 3 * LOCAL_DIM].reshape(bodies, self.n_medials * 3, LOCAL_DIM)
        return values, jacobian

    def value(self, q_a: np.ndarray, z_tilde: np.ndarray, rest_transform=None) -> np.ndarray:
        transform = np.eye(4) if rest_transform is None else rest_transform
        return self.evaluate(np.asarray(q_a)[None], np.asarray(z_tilde)[None], np.asarray(transform)[None])[0]

    def value_and_center_jacobian(self, q_a, z_tilde, rest_transform=None):
        transform = np.eye(4) if rest_transform is None else rest_transform
        value, jacobian = self.evaluate(np.asarray(q_a)[None], np.asarray(z_tilde)[None], np.asarray(transform)[None], True)
        return value[0], jacobian[0]
