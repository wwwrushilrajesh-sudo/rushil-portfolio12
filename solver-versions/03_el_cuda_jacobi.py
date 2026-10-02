"""CUDA version of the 2-D impulse/projection cavity-flow solver.

The field arrays remain in GPU global memory for the full run.  Numba CUDA
kernels perform the impulse RHS, all RK4 algebra, face interpolation, pressure
projection, velocity reconstruction, boundary conditions, and diagnostics.

The pressure Poisson equation is solved with ordinary double-buffered Jacobi.
Every pressure cell is updated in parallel from the previous Jacobi field.
The true Poisson residual controls convergence.
"""

from __future__ import annotations

import csv
from pathlib import Path
import time as wall_time

import matplotlib.pyplot as plt
import numpy as np
from numba import cuda


TPB_2D = (16, 16)
TPB_1D = 256


# ---------------------------------------------------------------------------
# PARAMETERS -- EDIT THESE VALUES EXACTLY LIKE IN YOUR ORIGINAL CODE
# ---------------------------------------------------------------------------
n = 256
vel = 1.0
delt = 0.001
visc = 0.001

total_time_steps = 150000
phi_tolerance = 1.0e-2
max_phi_iterations = 0       # 0 = no cap
phi_residual_check_interval = 2000
phi_iteration_sample_interval = 1000
save_interval = 1000

steady_state_tolerance = 1.0e-7
steady_state_required_consecutive = 1000

output_dir = Path("cuda_results")

# Values derived from the parameters above.
d = 1.0 / (n - 3)
k1c = 1.0 / (2.0 * d)
k2c = visc / (d * d)


@cuda.reduce
def sum_reduce(a, b):
    return a + b


@cuda.reduce
def max_reduce(a, b):
    return a if a > b else b


def grid_2d(shape):
    return (
        (shape[0] + TPB_2D[0] - 1) // TPB_2D[0],
        (shape[1] + TPB_2D[1] - 1) // TPB_2D[1],
    )


def grid_1d(length):
    return (length + TPB_1D - 1) // TPB_1D


@cuda.jit
def set_velocity_top(u, vel, n):
    j = cuda.grid(1)
    if 1 <= j <= n:
        u[1, j] = vel


@cuda.jit
def set_velocity_left_ghost(u, n):
    i = cuda.grid(1)
    if 1 <= i <= n:
        u[i, 0] = u[i, 1]


@cuda.jit
def set_velocity_corners(u, n):
    if cuda.grid(1) == 0:
        u[0, 0] = u[1, 1]
        u[0, n] = u[1, n - 1]
        u[n, 0] = u[n - 1, 1]
        u[n, n] = u[n - 1, n - 1]


@cuda.jit
def set_q_horizontal_boundaries(q1, q2, u, p, d, n):
    j = cuda.grid(1)
    if 3 <= j < n - 2:
        q1[1, j] = -q1[2, j] + 2.0 * u[1, j] + (p[1, j + 1] - p[1, j - 1]) / d
        q1[n - 1, j] = -q1[n - 2, j] + (p[n - 1, j + 1] - p[n - 1, j - 1]) / d
        q2[1, j] = -q2[2, j]
        q2[n - 1, j] = -q2[n - 2, j]


@cuda.jit
def set_q_vertical_boundaries(q1, q2, p, d, n):
    i = cuda.grid(1)
    if 3 <= i < n - 2:
        q2[i, 1] = -q2[i, 2] + (p[i + 1, 1] - p[i - 1, 1]) / d
        q2[i, n - 1] = -q2[i, n - 2] + (p[i + 1, n - 1] - p[i - 1, n - 1]) / d
        q1[i, 1] = -q1[i, 2]
        q1[i, n - 1] = -q1[i, n - 2]


@cuda.jit
def set_q_boundary_endpoints(q1, q2, u, p, d, n):
    if cuda.grid(1) == 0:
        q1[n - 1, 2] = (p[n - 1, 3] - p[n - 1, 2]) / d
        q1[n - 1, n - 2] = (p[n - 1, n - 2] - p[n - 1, n - 3]) / d
        q2[2, 1] = (p[3, 1] - p[2, 1]) / d
        q2[2, n - 1] = (p[3, n - 1] - p[2, n - 1]) / d
        q2[n - 2, 1] = (p[n - 2, 1] - p[n - 3, 1]) / d
        q2[n - 2, n - 1] = (p[n - 2, n - 1] - p[n - 3, n - 1]) / d
        q1[1, 2] = u[1, 2] + (p[1, 3] - p[1, 2]) / d
        q1[1, n - 2] = u[1, n - 2] + (p[1, n - 2] - p[1, n - 3]) / d


@cuda.jit
def impulse_rhs(out_q1, out_q2, q1, q2, u, v, k1c, k2c, n):
    i, j = cuda.grid(2)
    if i <= n and j <= n:
        if 2 <= i < n - 1 and 2 <= j < n - 1:
            du_dx = u[i, j + 1] - u[i, j - 1]
            du_dy = u[i + 1, j] - u[i - 1, j]
            dv_dx = v[i, j + 1] - v[i, j - 1]
            dv_dy = v[i + 1, j] - v[i - 1, j]
            lap_q1 = q1[i - 1, j] + q1[i + 1, j] + q1[i, j - 1] + q1[i, j + 1] - 4.0 * q1[i, j]
            lap_q2 = q2[i - 1, j] + q2[i + 1, j] + q2[i, j - 1] + q2[i, j + 1] - 4.0 * q2[i, j]
            out_q1[i, j] = -k1c * (u[i, j] * du_dx + v[i, j] * du_dy) + k2c * lap_q1
            out_q2[i, j] = -k1c * (u[i, j] * dv_dx + v[i, j] * dv_dy) + k2c * lap_q2
        else:
            out_q1[i, j] = 0.0
            out_q2[i, j] = 0.0


@cuda.jit
def rk_intermediate(q1a, q2a, q1, q2, k_q1, k_q2, factor, n):
    i, j = cuda.grid(2)
    if i <= n and j <= n:
        q1a[i, j] = q1[i, j] + factor * k_q1[i, j]
        q2a[i, j] = q2[i, j] + factor * k_q2[i, j]


@cuda.jit
def rk_final(q1, q2, k1_q1, k1_q2, k2_q1, k2_q2, k3_q1, k3_q2, k4_q1, k4_q2, delt, n):
    i, j = cuda.grid(2)
    if i <= n and j <= n:
        scale = delt / 6.0
        q1[i, j] += scale * (k1_q1[i, j] + 2.0 * k2_q1[i, j] + 2.0 * k3_q1[i, j] + k4_q1[i, j])
        q2[i, j] += scale * (k1_q2[i, j] + 2.0 * k2_q2[i, j] + 2.0 * k3_q2[i, j] + k4_q2[i, j])


@cuda.jit
def interpolate_faces(q1f, q2f, q1, q2, n):
    i, j = cuda.grid(2)
    if 1 <= i < n - 2 and 2 <= j < n - 2:
        q1f[i, j] = 0.5 * (q1[i + 1, j] + q1[i + 1, j + 1])
    if 2 <= i < n - 2 and 1 <= j < n - 2:
        q2f[i, j] = 0.5 * (q2[i, j + 1] + q2[i + 1, j + 1])


@cuda.jit
def build_pressure_rhs(rhs, q1f, q2f, d, n):
    ii, jj = cuda.grid(2)
    if ii < n - 3 and jj < n - 3:
        i = ii + 2
        j = jj + 2
        rhs[ii, jj] = (
            (q1f[i - 1, j] - q1f[i - 1, j - 1]) / d
            + (q2f[i, j - 1] - q2f[i - 1, j - 1]) / d
        )


@cuda.jit
def jacobi_pressure_update(p_new, p_old, rhs, change, d2, n):
    i, j = cuda.grid(2)
    if 2 <= i < n - 1 and 2 <= j < n - 1:
        ii = i - 2
        jj = j - 2
        new_value = 0.25 * (
            p_old[i + 1, j]
            + p_old[i - 1, j]
            + p_old[i, j + 1]
            + p_old[i, j - 1]
        ) - 0.25 * d2 * rhs[ii, jj]
        p_new[i, j] = new_value
        change[ii, jj] = abs(new_value - p_old[i, j])


@cuda.jit
def pressure_inner_faces(p, n):
    k = cuda.grid(1)
    if 2 <= k < n - 1:
        p[1, k] = p[2, k]
        p[n - 1, k] = p[n - 2, k]
        p[k, 1] = p[k, 2]
        p[k, n - 1] = p[k, n - 2]


@cuda.jit
def pressure_inner_corners(p, n):
    if cuda.grid(1) == 0:
        p[1, 1] = 0.5 * (p[1, 2] + p[2, 1])
        p[1, n - 1] = 0.5 * (p[2, n - 1] + p[1, n - 2])
        p[n - 1, 1] = 0.5 * (p[n - 1, 2] + p[n - 2, 1])
        p[n - 1, n - 1] = 0.5 * (p[n - 1, n - 2] + p[n - 2, n - 1])


@cuda.jit
def pressure_outer_faces(p, n):
    k = cuda.grid(1)
    if 1 <= k < n:
        p[0, k] = p[1, k]
        p[n, k] = p[n - 1, k]
        p[k, 0] = p[k, 1]
        p[k, n] = p[k, n - 1]


@cuda.jit
def pressure_outer_corners(p, n):
    if cuda.grid(1) == 0:
        p[0, 0] = p[1, 1]
        p[n, 0] = p[n - 1, 1]
        p[0, n] = p[1, n - 1]
        p[n, n] = p[n - 1, n - 1]


@cuda.jit
def poisson_residual(residual, p, rhs, d2, n):
    ii, jj = cuda.grid(2)
    if ii < n - 3 and jj < n - 3:
        i = ii + 2
        j = jj + 2
        lap_p = (p[i + 1, j] + p[i - 1, j] + p[i, j + 1] + p[i, j - 1] - 4.0 * p[i, j]) / d2
        residual[ii, jj] = abs(rhs[ii, jj] - lap_p)


@cuda.jit
def reconstruct_face_velocity(uf, vf, q1f, q2f, p, d, n):
    i, j = cuda.grid(2)
    if 1 <= i < n - 2 and 2 <= j < n - 2:
        uf[i, j] = q1f[i, j] - (p[i + 1, j + 1] - p[i + 1, j]) / d
    if 2 <= i < n - 2 and 1 <= j < n - 2:
        vf[i, j] = q2f[i, j] - (p[i + 1, j + 1] - p[i, j + 1]) / d


@cuda.jit
def reconstruct_cell_velocity(u, v, uf, vf, n):
    i, j = cuda.grid(2)
    if 2 <= i < n - 1 and 2 <= j < n - 1:
        u[i, j] = 0.5 * (uf[i - 1, j] + uf[i - 1, j - 1])
        v[i, j] = 0.5 * (vf[i - 1, j - 1] + vf[i, j - 1])


@cuda.jit
def steady_terms(diff_sq, sol_sq, max_change, u, v, u_prev, v_prev, n):
    i, j = cuda.grid(2)
    if 2 <= i < n - 1 and 2 <= j < n - 1:
        ii = i - 2
        jj = j - 2
        du = u[i, j] - u_prev[i, j]
        dv = v[i, j] - v_prev[i, j]
        diff_sq[ii, jj] = du * du + dv * dv
        sol_sq[ii, jj] = u[i, j] * u[i, j] + v[i, j] * v[i, j]
        adu = abs(du)
        adv = abs(dv)
        max_change[ii, jj] = adu if adu > adv else adv


@cuda.jit
def divergence_terms(face_abs, face_sq, cell_abs, cell_sq, u, v, uf, vf, d, n):
    i, j = cuda.grid(2)
    if 2 <= i < n - 1 and 2 <= j < n - 1:
        ii = i - 2
        jj = j - 2
        div_face = (uf[i - 1, j] - uf[i - 1, j - 1]) / d + (vf[i, j - 1] - vf[i - 1, j - 1]) / d
        div_cell = (u[i, j + 1] - u[i, j - 1]) / (2.0 * d) + (v[i + 1, j] - v[i - 1, j]) / (2.0 * d)
        af = abs(div_face)
        ac = abs(div_cell)
        face_abs[ii, jj] = af
        face_sq[ii, jj] = div_face * div_face
        cell_abs[ii, jj] = ac
        cell_sq[ii, jj] = div_cell * div_cell


@cuda.jit
def velocity_magnitude(magnitude, u, v, n):
    i, j = cuda.grid(2)
    if i <= n and j <= n:
        magnitude[i, j] = (u[i, j] * u[i, j] + v[i, j] * v[i, j]) ** 0.5



# ---------------------------------------------------------------------------
# GPU ARRAYS -- CREATED ONCE, THEN KEPT IN GPU GLOBAL MEMORY
# ---------------------------------------------------------------------------
def initialize_gpu_arrays():
    global d_u, d_v, d_q1, d_q2, d_p, d_p_new, d_u_prev, d_v_prev
    global d_q1f, d_q2f, d_uf, d_vf, d_rhs
    global d_k1_q1, d_k1_q2, d_k2_q1, d_k2_q2
    global d_k3_q1, d_k3_q2, d_k4_q1, d_k4_q2, d_q1a, d_q2a
    global d_residual, d_change, d_term1, d_term2, d_term3, d_term4
    global d_magnitude
    global blocks_field, blocks_interior, blocks_line

    field_shape = (n + 1, n + 1)
    interior_shape = (n - 3, n - 3)
    blocks_field = grid_2d(field_shape)
    blocks_interior = grid_2d(interior_shape)
    blocks_line = grid_1d(n + 1)

    d_u = cuda.to_device(np.zeros(field_shape, dtype=np.float64))
    d_v = cuda.to_device(np.zeros(field_shape, dtype=np.float64))
    d_q1 = cuda.to_device(np.zeros(field_shape, dtype=np.float64))
    d_q2 = cuda.to_device(np.zeros(field_shape, dtype=np.float64))
    d_p = cuda.to_device(np.zeros(field_shape, dtype=np.float64))
    d_p_new = cuda.to_device(np.zeros(field_shape, dtype=np.float64))
    d_u_prev = cuda.to_device(np.zeros(field_shape, dtype=np.float64))
    d_v_prev = cuda.to_device(np.zeros(field_shape, dtype=np.float64))

    d_q1f = cuda.to_device(np.zeros((n - 1, n), dtype=np.float64))
    d_q2f = cuda.to_device(np.zeros((n, n - 1), dtype=np.float64))
    d_uf = cuda.to_device(np.zeros((n - 1, n), dtype=np.float64))
    d_vf = cuda.to_device(np.zeros((n, n - 1), dtype=np.float64))
    d_rhs = cuda.to_device(np.zeros(interior_shape, dtype=np.float64))

    d_k1_q1 = cuda.to_device(np.zeros(field_shape, dtype=np.float64))
    d_k1_q2 = cuda.to_device(np.zeros(field_shape, dtype=np.float64))
    d_k2_q1 = cuda.to_device(np.zeros(field_shape, dtype=np.float64))
    d_k2_q2 = cuda.to_device(np.zeros(field_shape, dtype=np.float64))
    d_k3_q1 = cuda.to_device(np.zeros(field_shape, dtype=np.float64))
    d_k3_q2 = cuda.to_device(np.zeros(field_shape, dtype=np.float64))
    d_k4_q1 = cuda.to_device(np.zeros(field_shape, dtype=np.float64))
    d_k4_q2 = cuda.to_device(np.zeros(field_shape, dtype=np.float64))
    d_q1a = cuda.to_device(np.zeros(field_shape, dtype=np.float64))
    d_q2a = cuda.to_device(np.zeros(field_shape, dtype=np.float64))

    d_residual = cuda.to_device(np.zeros(interior_shape, dtype=np.float64))
    d_change = cuda.to_device(np.zeros(interior_shape, dtype=np.float64))
    d_term1 = cuda.to_device(np.zeros(interior_shape, dtype=np.float64))
    d_term2 = cuda.to_device(np.zeros(interior_shape, dtype=np.float64))
    d_term3 = cuda.to_device(np.zeros(interior_shape, dtype=np.float64))
    d_term4 = cuda.to_device(np.zeros(interior_shape, dtype=np.float64))
    d_magnitude = cuda.to_device(np.zeros(field_shape, dtype=np.float64))


def apply_boundaries():
    set_velocity_top[blocks_line, TPB_1D](d_u, vel, n)
    set_velocity_left_ghost[blocks_line, TPB_1D](d_u, n)
    set_velocity_corners[1, 1](d_u, n)
    set_q_horizontal_boundaries[blocks_line, TPB_1D](d_q1, d_q2, d_u, d_p, d, n)
    set_q_vertical_boundaries[blocks_line, TPB_1D](d_q1, d_q2, d_p, d, n)
    set_q_boundary_endpoints[1, 1](d_q1, d_q2, d_u, d_p, d, n)


def apply_pressure_boundaries(pressure_field):
    pressure_inner_faces[blocks_line, TPB_1D](pressure_field, n)
    pressure_inner_corners[1, 1](pressure_field, n)
    pressure_outer_faces[blocks_line, TPB_1D](pressure_field, n)
    pressure_outer_corners[1, 1](pressure_field, n)


def solve_pressure_and_velocity(stage_q1, stage_q2, record_history=False):
    global d_p, d_p_new

    interpolate_faces[blocks_field, TPB_2D](d_q1f, d_q2f, stage_q1, stage_q2, n)
    build_pressure_rhs[blocks_interior, TPB_2D](d_rhs, d_q1f, d_q2f, d, n)

    poisson_residual[blocks_interior, TPB_2D](d_residual, d_p, d_rhs, d * d, n)
    final_residual = float(max_reduce(d_residual.ravel()))

    iterations_used = 0
    residual_history = []
    change_history = []

    while final_residual > phi_tolerance:
        sweeps_this_batch = phi_residual_check_interval
        if max_phi_iterations > 0:
            sweeps_left = max_phi_iterations - iterations_used
            if sweeps_left < sweeps_this_batch:
                sweeps_this_batch = sweeps_left

        # Run a batch of Jacobi sweeps without returning data to the CPU.
        for _ in range(sweeps_this_batch):
            jacobi_pressure_update[blocks_field, TPB_2D](
                d_p_new, d_p, d_rhs, d_change, d * d, n
            )
            apply_pressure_boundaries(d_p_new)

            # Swap GPU-array references. No array is copied here.
            d_p, d_p_new = d_p_new, d_p
            iterations_used += 1

        # Only after the batch do we calculate and return one residual scalar.
        poisson_residual[blocks_interior, TPB_2D](
            d_residual, d_p, d_rhs, d * d, n
        )
        final_residual = float(max_reduce(d_residual.ravel()))

        if record_history:
            residual_history.append(final_residual)
            change_history.append(float(max_reduce(d_change.ravel())))

        if (
            max_phi_iterations > 0
            and iterations_used >= max_phi_iterations
            and final_residual > phi_tolerance
        ):
            raise RuntimeError(
                f"Pressure solve reached {max_phi_iterations} iterations "
                f"with residual {final_residual:.6e}"
            )

    reconstruct_face_velocity[blocks_field, TPB_2D](
        d_uf, d_vf, d_q1f, d_q2f, d_p, d, n
    )
    reconstruct_cell_velocity[blocks_field, TPB_2D](d_u, d_v, d_uf, d_vf, n)

    return iterations_used, final_residual, residual_history, change_history


def one_step():
    apply_boundaries()

    impulse_rhs[blocks_field, TPB_2D](
        d_k1_q1, d_k1_q2, d_q1, d_q2, d_u, d_v, k1c, k2c, n
    )
    rk_intermediate[blocks_field, TPB_2D](
        d_q1a, d_q2a, d_q1, d_q2, d_k1_q1, d_k1_q2, 0.5 * delt, n
    )
    stage1_iters, stage1_residual, _, _ = solve_pressure_and_velocity(d_q1a, d_q2a)

    impulse_rhs[blocks_field, TPB_2D](
        d_k2_q1, d_k2_q2, d_q1a, d_q2a, d_u, d_v, k1c, k2c, n
    )
    rk_intermediate[blocks_field, TPB_2D](
        d_q1a, d_q2a, d_q1, d_q2, d_k2_q1, d_k2_q2, 0.5 * delt, n
    )
    solve_pressure_and_velocity(d_q1a, d_q2a)

    impulse_rhs[blocks_field, TPB_2D](
        d_k3_q1, d_k3_q2, d_q1a, d_q2a, d_u, d_v, k1c, k2c, n
    )
    rk_intermediate[blocks_field, TPB_2D](
        d_q1a, d_q2a, d_q1, d_q2, d_k3_q1, d_k3_q2, delt, n
    )
    rk3_iters, rk3_residual, rk3_history, rk3_change = (
        solve_pressure_and_velocity(d_q1a, d_q2a, record_history=True)
    )

    impulse_rhs[blocks_field, TPB_2D](
        d_k4_q1, d_k4_q2, d_q1a, d_q2a, d_u, d_v, k1c, k2c, n
    )
    rk_final[blocks_field, TPB_2D](
        d_q1, d_q2,
        d_k1_q1, d_k1_q2,
        d_k2_q1, d_k2_q2,
        d_k3_q1, d_k3_q2,
        d_k4_q1, d_k4_q2,
        delt, n,
    )

    final_iters, final_residual, _, _ = solve_pressure_and_velocity(d_q1, d_q2)

    return (
        stage1_iters, stage1_residual,
        final_iters, final_residual,
        rk3_iters, rk3_residual,
        rk3_history, rk3_change,
    )


def compute_steady_state_metrics():
    steady_terms[blocks_field, TPB_2D](
        d_term1, d_term2, d_term3, d_u, d_v, d_u_prev, d_v_prev, n
    )
    diff_sq = float(sum_reduce(d_term1.ravel()))
    sol_sq = float(sum_reduce(d_term2.ravel()))
    linf_change = float(max_reduce(d_term3.ravel()))
    count = (n - 3) ** 2

    if sol_sq > 1.0e-300:
        relative_l2 = (diff_sq / sol_sq) ** 0.5
    else:
        relative_l2 = diff_sq ** 0.5

    rms_change = (diff_sq / count) ** 0.5
    return relative_l2, rms_change, linf_change


def compute_divergence_metrics():
    divergence_terms[blocks_field, TPB_2D](
        d_term1, d_term2, d_term3, d_term4, d_u, d_v, d_uf, d_vf, d, n
    )
    count = (n - 3) ** 2

    face_l1 = float(sum_reduce(d_term1.ravel())) / count
    face_l2 = (float(sum_reduce(d_term2.ravel())) / count) ** 0.5
    face_linf = float(max_reduce(d_term1.ravel()))

    cell_l1 = float(sum_reduce(d_term3.ravel())) / count
    cell_l2 = (float(sum_reduce(d_term4.ravel())) / count) ** 0.5
    cell_linf = float(max_reduce(d_term3.ravel()))

    return face_l1, face_l2, face_linf, cell_l1, cell_l2, cell_linf


def copy_current_velocity_to_previous():
    d_u_prev.copy_to_device(d_u)
    d_v_prev.copy_to_device(d_v)


def write_csv(path, header, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def save_stage1_history(time_history, iteration_history):
    write_csv(
        output_dir / "rk4_stage1_phi_iterations_sampled.csv",
        ["time_step", "rk4_stage1_phi_iterations"],
        zip(time_history, iteration_history),
    )

    fig, ax = plt.subplots()
    ax.plot(time_history, iteration_history, marker="o", markersize=3, linewidth=1.2)
    ax.set_xlabel("Time step")
    ax.set_ylabel("First RK4 projection Jacobi iterations")
    ax.set_title(
        rf"First RK4 projection iterations to "
        rf"$\|R_\phi\|_\infty \leq {phi_tolerance:.0e}$"
    )
    ax.grid(True)
    fig.savefig(
        output_dir / "rk4_stage1_phi_iterations_vs_timestep.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_results(divergence_history, steady_history, rk3_history, rk3_change):
    write_csv(
        output_dir / "divergence_history.csv",
        [
            "time_step", "face_div_L1", "face_div_L2", "face_div_Linf",
            "cell_div_L1", "cell_div_L2", "cell_div_Linf",
        ],
        divergence_history,
    )
    write_csv(
        output_dir / "steady_state_history.csv",
        [
            "time_step", "relative_velocity_L2_change", "rms_velocity_change",
            "linf_velocity_change", "steady_state_tolerance",
            "steady_reached_time",
        ],
        steady_history,
    )
    write_csv(
        output_dir / "rk3_phi_residual_latest_timestep.csv",
        ["local_phi_iteration", "poisson_residual", "phi_change"],
        (
            (
                (sample + 1) * phi_residual_check_interval,
                rk3_history[sample],
                rk3_change[sample],
            )
            for sample in range(len(rk3_history))
        ),
    )

    np.savez_compressed(
        output_dir / "latest_fields.npz",
        u=d_u.copy_to_host(),
        v=d_v.copy_to_host(),
        q1=d_q1.copy_to_host(),
        q2=d_q2.copy_to_host(),
        phi=d_p.copy_to_host(),
    )


def main():
    if n < 7:
        raise ValueError("n must be at least 7 for this boundary stencil")
    if phi_residual_check_interval < 1:
        raise ValueError("phi_residual_check_interval must be at least 1")
    if not cuda.is_available():
        raise RuntimeError(
            "No CUDA-capable NVIDIA GPU/driver/toolkit is available to Numba"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    device = cuda.current_context().device
    device_name = getattr(device, "name", "CUDA device")
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    print("CUDA device:", device_name)

    initialize_gpu_arrays()

    phi_time_history = []
    phi_iteration_history = []
    steady_history = []
    divergence_history = []

    steady_consecutive_count = 0
    steady_reached_time = -1
    latest_rk3_history = []
    latest_rk3_change = []

    start_time = wall_time.perf_counter()

    for time_step in range(total_time_steps):
        (
            stage1_phi_iters,
            stage1_phi_residual,
            final_phi_iters,
            final_phi_residual,
            rk3_phi_iters,
            rk3_phi_residual,
            latest_rk3_history,
            latest_rk3_change,
        ) = one_step()

        steady_rel_l2, steady_rms, steady_linf = compute_steady_state_metrics()

        if steady_rel_l2 <= steady_state_tolerance:
            steady_consecutive_count += 1
        else:
            steady_consecutive_count = 0

        if (
            steady_reached_time < 0
            and steady_consecutive_count >= steady_state_required_consecutive
        ):
            steady_reached_time = time_step - steady_state_required_consecutive + 1
            print("STEADY STATE DETECTED AT STEP:", steady_reached_time)

        divergence = compute_divergence_metrics()
        divergence_history.append((time_step, *divergence))

        if time_step % phi_iteration_sample_interval == 0:
            phi_time_history.append(time_step)
            phi_iteration_history.append(stage1_phi_iters)
            steady_history.append(
                (
                    time_step, steady_rel_l2, steady_rms, steady_linf,
                    steady_state_tolerance, steady_reached_time,
                )
            )

            elapsed = wall_time.perf_counter() - start_time
            print(
                "step=", time_step,
                "elapsed=", elapsed,
                "stage1_iters=", stage1_phi_iters,
                "stage1_R=", stage1_phi_residual,
                "rk3_R=", rk3_phi_residual,
                "final_R=", final_phi_residual,
                "face_div_L2=", divergence[1],
                "cell_div_L2=", divergence[4],
                "steady_rel_L2=", steady_rel_l2,
            )
            save_stage1_history(phi_time_history, phi_iteration_history)

        if time_step % save_interval == 0:
            save_results(
                divergence_history,
                steady_history,
                latest_rk3_history,
                latest_rk3_change,
            )

        copy_current_velocity_to_previous()

    cuda.synchronize()
    total_elapsed = wall_time.perf_counter() - start_time
    print("Completed", total_time_steps, "steps in", total_elapsed, "seconds")


if __name__ == "__main__":
    main()