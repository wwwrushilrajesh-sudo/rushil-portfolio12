"""
@author: wwwru
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import pyplot
from numba import njit, prange

# -------------------------
# Parameters
# -------------------------
n    = 256
vel  = 1.0
d    = 1.0/(n-3)
delt = 0.001
visc = 0.001
k1c  = 1.0/(2.0*d)
k2c  = visc/(d*d)
omega = 1.9   # SOR relaxation factor; use 1 < omega < 2

# Global run / phi convergence settings.
# EVERY phi projection at EVERY RK4 stage uses the true Poisson residual criterion
# max|RHS - lap(phi)| <= 1e-3 for the ENTIRE run. There is no later switch
# back to a fixed SOR iteration count.
total_time_steps = 150000
phi_tolerance = 1.0e-3
phi_iteration_sample_interval = 1000

# Steady-state tracker settings.
# A time step is considered locally steady when the relative L2 change of the
# velocity field between consecutive time steps is below this tolerance.
# Requiring many consecutive steps prevents a one-step fluctuation from being
# mislabeled as steady state. The run DOES NOT stop when steady state is detected.
steady_state_tolerance = 1.0e-7
steady_state_required_consecutive = 1000
steady_state_sample_interval = 1000

# Optional RK3 residual-history storage only. This is NOT an iteration cap;
# SOR may run past this value at any time during the full residual-controlled run.
max_phi_iters = 120000

# Console printing controls for phi residual history.
# True = print every local SOR iteration of the RK4 3rd-stage projection.
# Keep this False so the terminal does NOT print every SOR iteration.
print_phi_residual_each_iteration = False

# To avoid killing the terminal, only print this detailed residual table every N time steps.
# Set to 1 to print every time step. Set to 100 or 1000 for cleaner long runs.
phi_residual_print_interval = 1000


# -------------------------
# Arrays
# -------------------------
u  = np.zeros((n+1, n+1), dtype=np.float64)
v  = np.zeros((n+1, n+1), dtype=np.float64)
q1 = np.zeros((n+1, n+1), dtype=np.float64)
q2 = np.zeros((n+1, n+1), dtype=np.float64)
p  = np.zeros((n+1, n+1), dtype=np.float64)

# Previous-step velocity fields for steady-state tracking.
u_prev = np.zeros_like(u)
v_prev = np.zeros_like(v)

# face arrays
q1f = np.zeros((n-1, n),   dtype=np.float64)
q2f = np.zeros((n,   n-1), dtype=np.float64)
uf  = np.zeros((n-1, n),   dtype=np.float64)
vf  = np.zeros((n,   n-1), dtype=np.float64)

# scratch (prealloc so njit stays fast + no allocations)
q1s = np.zeros((n+1, n+1), dtype=np.float64)
q2s = np.zeros((n+1, n+1), dtype=np.float64)

p_old = np.zeros_like(p)
rhs   = np.zeros((n-3, n-3), dtype=np.float64)  # interior size (2:n-1,2:n-1)

# RK4 scratch (NO temporaries)
k1_q1 = np.zeros_like(q1); k1_q2 = np.zeros_like(q2)
k2_q1 = np.zeros_like(q1); k2_q2 = np.zeros_like(q2)
k3_q1 = np.zeros_like(q1); k3_q2 = np.zeros_like(q2)
k4_q1 = np.zeros_like(q1); k4_q2 = np.zeros_like(q2)
q1a   = np.zeros_like(q1)
q2a   = np.zeros_like(q2)

F = np.zeros_like(u)  # plotting buffer

# These arrays are overwritten every time step with the RK4 3rd-stage projection history.
rk3_phi_residual_iter = np.zeros(max_phi_iters, dtype=np.float64)
rk3_phi_change_iter   = np.zeros(max_phi_iters, dtype=np.float64)



# -------------------------
# Numba kernels (FOR-LOOP versions)
# -------------------------
@njit(fastmath=True,parallel=True)
def solve_impulse(out_q1, out_q2, q1, q2, u, v, k1, k2, q1s, q2s, n):
    # zero scratch
    for i in prange(n+1):
        for j in prange(n+1):
            q1s[i, j] = 0.0
            q2s[i, j] = 0.0

    # interior update
    for i in range(2, n-1):
        for j in range(2, n-1):
            du_dx = q1[i, j+1] - q1[i, j-1]
            du_dy = q1[i+1, j] - q1[i-1, j]
            lap_q1 = (q1[i-1, j] + q1[i+1, j] + q1[i, j-1] + q1[i, j+1] - 4.0*q1[i, j])

            dv_dx = q2[i, j+1] - q2[i, j-1]
            dv_dy = q2[i+1, j] - q2[i-1, j]
            lap_q2 = (q2[i-1, j] + q2[i+1, j] + q2[i, j-1] + q2[i, j+1] - 4.0*q2[i, j])

            q1s[i, j] = -k1 * (u[i, j]*du_dx + v[i, j]*du_dy) + k2 * lap_q1-k1*(q1[i,j]*(u[i,j+1]-u[i,j-1])+q2[i,j]*(v[i,j+1]-v[i,j-1]))  
            q2s[i, j] = -k1 * (u[i, j]*dv_dx + v[i, j]*dv_dy) + k2 * lap_q2-k1*(q1[i,j]*(u[i+1,j]-u[i-1,j])+q2[i,j]*(v[i+1,j]-v[i-1,j]))

    # copy to outputs
    for i in prange(n+1):
        for j in prange(n+1):
            out_q1[i, j] = q1s[i, j]
            out_q2[i, j] = q2s[i, j]



@njit(fastmath=True)
def compute_poisson_residual_max(p, rhs, d, n):
    """
    True Poisson residual for the pressure/potential solve:
        R_phi = RHS - Laplacian(phi)

    This is different from max(|phi_new - phi_old|). It checks whether
    the current phi field actually satisfies the discrete Poisson equation.
    """
    max_res = 0.0
    for i in range(2, n-1):
        ii = i - 2
        for j in range(2, n-1):
            jj = j - 2
            lap_p = (p[i+1, j] + p[i-1, j] + p[i, j+1] + p[i, j-1] - 4.0*p[i, j]) / (d*d)
            res = rhs[ii, jj] - lap_p
            ares = abs(res)
            if ares > max_res:
                max_res = ares
    return max_res

@njit(fastmath=True)
def solve_pressure_and_velocity(p, p_old, rhs, q1, q2, u, v,
                               q1f, q2f, uf, vf,
                               d, ter, ni, n, omega,
                               record_rk3_phi,
                               rk3_phi_residual_iter,
                               rk3_phi_change_iter,
                               use_phi_tolerance,
                               phi_tolerance):

    # -------------------------
    # face interpolation (FOR LOOPS)
    # q1f[1:n-2, 2:n-2] = 0.5*(q1[2:n-1,2:n-2] + q1[2:n-1,3:n-1])
    # -------------------------
    for i in range(1, n-2):          # 1..n-3
        for j in range(2, n-2):      # 2..n-3
            q1f[i, j] = 0.5*(q1[i+1, j] + q1[i+1, j+1])

    # q2f[2:n-2, 1:n-2] = 0.5*(q2[2:n-2,2:n-1] + q2[3:n-1,2:n-1])
    for i in range(2, n-2):          # 2..n-3
        for j in range(1, n-2):      # 1..n-3
            q2f[i, j] = 0.5*(q2[i, j+1] + q2[i+1, j+1])

    # -------------------------
    # rhs (compute ONCE) (FOR LOOPS)
    # rhs[i-2,j-2] corresponds to p cell (i,j) for i,j = 2..n-2
    # rhs = (q1f[1:n-2,2:n-1]-q1f[1:n-2,1:n-2])/d + (q2f[2:n-1,1:n-2]-q2f[1:n-2,1:n-2])/d
    # -------------------------
    for ii in range(0, n-3):         # 0..n-4  (maps to i = ii+2)
        i = ii + 2
        for jj in range(0, n-3):     # 0..n-4  (maps to j = jj+2)
            j = jj + 2
            term1 = (q1f[i-1, j]   - q1f[i-1, j-1]) / d
            term2 = (q2f[i,   j-1] - q2f[i-1, j-1]) / d
            rhs[ii, jj] = term1 + term2

    # Clear the RK3 diagnostic arrays only when this pressure solve is the
    # 3rd RK4 projection stage. This avoids mixing old iteration histories.
    if record_rk3_phi == 1:
        for it_clear in range(rk3_phi_residual_iter.shape[0]):
            rk3_phi_residual_iter[it_clear] = 0.0
            rk3_phi_change_iter[it_clear] = 0.0
    
    # -------------------------
    # SOR / Gauss-Seidel iterations for p
    # -------------------------
    # Whenever use_phi_tolerance == 1 there is deliberately NO maximum
    # SOR iteration count: iterate until the true Poisson residual is <= tol.
    # In this script use_phi_tolerance is kept ON for the entire run.
    iterations_used = 0
    final_poisson_residual = compute_poisson_residual_max(p, rhs, d, n)

    # If the warm-started phi field already satisfies the tolerance, zero SOR
    # sweeps are required for this projection.
    if not (use_phi_tolerance == 1 and final_poisson_residual <= phi_tolerance):
        it = 0
        while True:
            phi_max_change = 0.0

            # interior update: p[2:n-1,2:n-1]
            for i in range(2, n-1):          # 2..n-2
                ii = i - 2
                for j in range(2, n-1):      # 2..n-2
                    jj = j - 2
                    p_new = 0.25*(p[i+1, j] + p[i-1, j] + p[i, j+1] + p[i, j-1]) \
                            - 0.25*(d*d)*rhs[ii, jj]

                    p_before = p[i, j]
                    p_after = (1.0 - omega)*p_before + omega*p_new
                    change = abs(p_after - p_before)
                    if change > phi_max_change:
                        phi_max_change = change
                    p[i, j] = p_after

            # boundaries (same logic as original solver)
            for j in range(2, n-1):
                p[1, j]   = p[2, j]
                p[n-1, j] = p[n-2, j]
            for i in range(2, n-1):
                p[i, 1]   = p[i, 2]
                p[i, n-1] = p[i, n-2]

            p[1, 1]     = 0.5*(p[1, 2]     + p[2, 1])
            p[1, n-1]   = 0.5*(p[2, n-1]   + p[1, n-2])
            p[n-1, 1]   = 0.5*(p[n-1, 2]   + p[n-2, 1])
            p[n-1, n-1] = 0.5*(p[n-1, n-2] + p[n-2, n-1])

            for j in range(1, n):
                p[0, j] = p[1, j]
                p[n, j] = p[n-1, j]
            for i in range(1, n):
                p[i, 0] = p[i, 1]
                p[i, n] = p[i, n-1]

            p[0, 0] = p[1, 1]
            p[n, 0] = p[n-1, 1]
            p[0, n] = p[1, n-1]
            p[n, n] = p[n-1, n-1]

            final_poisson_residual = compute_poisson_residual_max(p, rhs, d, n)
            iterations_used = it + 1

            # Diagnostic history remains storage-capped only; this does NOT
            # cap the number of SOR iterations performed.
            if record_rk3_phi == 1 and it < rk3_phi_residual_iter.shape[0]:
                rk3_phi_change_iter[it] = phi_max_change
                rk3_phi_residual_iter[it] = final_poisson_residual

            if use_phi_tolerance == 1:
                if final_poisson_residual <= phi_tolerance:
                    break
            else:
                if iterations_used >= ter:
                    break

            it += 1

    # -------------------------
    # face velocities uf/vf (FOR LOOPS)
    # uf[1:n-2,2:n-2] = q1f[...] - (p[2:n-1,3:n-1]-p[2:n-1,2:n-2])/d
    # vf[2:n-2,1:n-2] = q2f[...] - (p[3:n-1,2:n-1]-p[2:n-2,2:n-1])/d
    # -------------------------
    for i in range(1, n-2):          # 1..n-3
        for j in range(2, n-2):      # 2..n-3
            uf[i, j] = q1f[i, j] - (p[i+1, j+1] - p[i+1, j]) / d

    for i in range(2, n-2):          # 2..n-3
        for j in range(1, n-2):      # 1..n-3
            vf[i, j] = q2f[i, j] - (p[i+1, j+1] - p[i, j+1]) / d

    # -------------------------
    # cell-centered u,v (FOR LOOPS)
    # u[2:n-1,2:n-1] = 0.5*(uf[1:n-2,2:n-1] + uf[1:n-2,1:n-2])
    # v[2:n-1,2:n-1] = 0.5*(vf[1:n-2,1:n-2] + vf[2:n-1,1:n-2])
    # -------------------------
    for i in range(2, n-1):          # 2..n-2
        for j in range(2, n-1):      # 2..n-2
            u[i, j] = 0.5*(uf[i-1, j]   + uf[i-1, j-1])
            v[i, j] = 0.5*(vf[i-1, j-1] + vf[i,   j-1])

    if ni == 1:
        for i in range(2, n-1):
            for j in range(2, n-1):
                q1[i, j] = 0.5*(q1f[i-1, j]   + q1f[i-1, j-1])
                q2[i, j] = 0.5*(q2f[i,   j-1] + q2f[i-1, j-1])

    return iterations_used, final_poisson_residual


@njit(fastmath=True,parallel=True)
def one_step(u, v, q1, q2, p,
             q1f, q2f, uf, vf,
             q1s, q2s,
             p_old, rhs,
             k1_q1, k1_q2, k2_q1, k2_q2, k3_q1, k3_q2, k4_q1, k4_q2,
             q1a, q2a,
             rk3_phi_residual_iter, rk3_phi_change_iter,
             n, d, delt, k1c, k2c, vel,
             ter_stage, omega,
             use_phi_tolerance, phi_tolerance):

    # -------------------------
    # BCs (FOR LOOPS)
    # -------------------------
    for j in range(1, n+1):
        u[1, j] = vel
    u[1, 1] = vel

    for i in range(1, n+1):
        u[i, 0] = u[i, 1]

    u[0, 0] = u[1, 1]
    u[0, n] = u[1, n-1]
    u[n, 0] = u[n-1, 1]
    u[n, n] = u[n-1, n-1]

    # -------------------------
    # q boundary conditions (FOR LOOPS versions of your slices)
    # -------------------------
    for j in range(3, n-2):  # 3..n-3
        q1[1,   j] = -q1[2,j]+2*u[1, j] + (p[1, j+1] - p[1, j-1])/(1*d)
        q1[n-1, j] =   -q1[n-2,j]+       (p[n-1, j+1] - p[n-1, j-1])/(1*d)
        q2[1,   j] = -q2[2,j]
        q2[n-1, j] = -q2[n-2,j]

    for i in range(3, n-2):  # 3..n-3
        q2[i, 1]   = -q2[i,2]+(p[i+1, 1] - p[i-1, 1])/(1*d)
        q2[i, n-1] = -q2[i,n-2]+(p[i+1, n-1] - p[i-1, n-1])/(1*d)
        q1[i, 1]   = -q1[i,2]
        q1[i, n-1] = -q1[i,n-2]

    q1[n-1, 2]   = (p[n-1, 3]   - p[n-1, 2]) / d
    q1[n-1, n-2] = (p[n-1, n-2] - p[n-1, n-3]) / d
    q2[2,   1]   = (p[3,   1]   - p[2,   1]) / d
    q2[2,   n-1] = (p[3,   n-1] - p[2,   n-1]) / d

    # NOTE: keeping your exact line (you omitted "/d" on the last one in your code)
    q2[n-2, 1]   = (p[n-2, 1]   - p[n-3, 1]) /( d)
    q2[n-2, n-1] = (p[n-2, n-1] - p[n-3, n-1]) / d   # fixed: missing /d

    q1[1, 2]   = u[1, 2]   + (p[1, 3]   - p[1, 2]) / d
    q1[1, n-2] = u[1, n-2] + (p[1, n-2] - p[1, n-3]) / d

   

    # -------------------------
    # RK4 (same logic)
    # -------------------------
    solve_impulse(k1_q1, k1_q2, q1, q2, u, v, k1c, k2c, q1s, q2s, n)

    for i in prange(n+1):
        for j in prange(n+1):
            q1a[i, j] = q1[i, j] + 0.5*delt*k1_q1[i, j]
            q2a[i, j] = q2[i, j] + 0.5*delt*k1_q2[i, j]
    stage1_phi_iters, stage1_phi_residual = solve_pressure_and_velocity(p, p_old, rhs, q1a, q2a, u, v, q1f, q2f, uf, vf, d, ter_stage, 0, n, omega, 0, rk3_phi_residual_iter, rk3_phi_change_iter, use_phi_tolerance, phi_tolerance)

    solve_impulse(k2_q1, k2_q2, q1a, q2a, u, v, k1c, k2c, q1s, q2s, n)

    for i in prange(n+1):
        for j in prange(n+1):
            q1a[i, j] = q1[i, j] + 0.5*delt*k2_q1[i, j]
            q2a[i, j] = q2[i, j] + 0.5*delt*k2_q2[i, j]
    solve_pressure_and_velocity(p, p_old, rhs, q1a, q2a, u, v, q1f, q2f, uf, vf, d, ter_stage, 0, n, omega, 0, rk3_phi_residual_iter, rk3_phi_change_iter, use_phi_tolerance, phi_tolerance)

    solve_impulse(k3_q1, k3_q2, q1a, q2a, u, v, k1c, k2c, q1s, q2s, n)

    for i in prange(n+1):
        for j in prange(n+1):
            q1a[i, j] = q1[i, j] + delt*k3_q1[i, j]
            q2a[i, j] = q2[i, j] + delt*k3_q2[i, j]
    rk3_phi_iters, rk3_phi_final_residual = solve_pressure_and_velocity(p, p_old, rhs, q1a, q2a, u, v, q1f, q2f, uf, vf, d, ter_stage, 0, n, omega, 1, rk3_phi_residual_iter, rk3_phi_change_iter, use_phi_tolerance, phi_tolerance)

    solve_impulse(k4_q1, k4_q2, q1a, q2a, u, v, k1c, k2c, q1s, q2s, n)

    for i in prange(n+1):
        for j in prange(n+1):
            q1[i, j] = q1[i, j] + (delt/6.0)*(k1_q1[i, j] + 2.0*k2_q1[i, j] + 2.0*k3_q1[i, j] + k4_q1[i, j])
            q2[i, j] = q2[i, j] + (delt/6.0)*(k1_q2[i, j] + 2.0*k2_q2[i, j] + 2.0*k3_q2[i, j] + k4_q2[i, j])

    # final projection uses SAME stage count
    final_phi_iters, final_phi_residual = solve_pressure_and_velocity(p, p_old, rhs, q1, q2, u, v, q1f, q2f, uf, vf, d, ter_stage, 0, n, omega, 0, rk3_phi_residual_iter, rk3_phi_change_iter, use_phi_tolerance, phi_tolerance)

    return stage1_phi_iters, stage1_phi_residual, final_phi_iters, final_phi_residual, rk3_phi_iters, rk3_phi_final_residual


@njit(fastmath=True)
def compute_steady_state_metrics(u, v, u_prev, v_prev, n):
    """
    Compare the current and previous velocity fields on the physical interior.

    Returns
    -------
    rel_l2 : ||U^n-U^(n-1)||_2 / ||U^n||_2
    rms_change : RMS absolute velocity-vector change per cell
    linf_change : maximum component-wise |delta u| or |delta v|
    """
    diff_sq = 0.0
    sol_sq = 0.0
    max_change = 0.0
    count = 0

    for i in range(2, n-1):
        for j in range(2, n-1):
            du = u[i, j] - u_prev[i, j]
            dv = v[i, j] - v_prev[i, j]

            diff_sq += du*du + dv*dv
            sol_sq += u[i, j]*u[i, j] + v[i, j]*v[i, j]

            adu = abs(du)
            adv = abs(dv)
            if adu > max_change:
                max_change = adu
            if adv > max_change:
                max_change = adv

            count += 1

    if sol_sq > 1.0e-300:
        rel_l2 = (diff_sq / sol_sq) ** 0.5
    else:
        rel_l2 = diff_sq ** 0.5

    if count > 0:
        rms_change = (diff_sq / count) ** 0.5
    else:
        rms_change = 0.0

    return rel_l2, rms_change, max_change


@njit(fastmath=True)
def compute_F(F, u, v, n):
    for i in prange(n+1):
        for j in prange(n+1):
            F[i, j] = (u[i, j]*u[i, j] + v[i, j]*v[i, j])**0.5


@njit(fastmath=True)
def compute_divergence_metrics(u, v, uf, vf, d, n):
    """
    Computes divergence after projection.

    face divergence matches your projection/Rhie-Chow face-flux layout.
    cell divergence uses central differences on cell-centered u and v.
    """
    face_sum_sq = 0.0
    face_sum_abs = 0.0
    face_max = 0.0

    cell_sum_sq = 0.0
    cell_sum_abs = 0.0
    cell_max = 0.0

    count = 0

    for i in range(2, n-1):
        for j in range(2, n-1):
            div_face = ((uf[i-1, j] - uf[i-1, j-1]) / d
                        + (vf[i, j-1] - vf[i-1, j-1]) / d)

            div_cell = ((u[i, j+1] - u[i, j-1]) / (2.0*d)
                        + (v[i+1, j] - v[i-1, j]) / (2.0*d))

            abs_face = abs(div_face)
            abs_cell = abs(div_cell)

            face_sum_sq += div_face * div_face
            face_sum_abs += abs_face
            if abs_face > face_max:
                face_max = abs_face

            cell_sum_sq += div_cell * div_cell
            cell_sum_abs += abs_cell
            if abs_cell > cell_max:
                cell_max = abs_cell

            count += 1

    if count > 0:
        face_l2 = (face_sum_sq / count) ** 0.5
        face_l1 = face_sum_abs / count
        cell_l2 = (cell_sum_sq / count) ** 0.5
        cell_l1 = cell_sum_abs / count
    else:
        face_l2 = 0.0
        face_l1 = 0.0
        cell_l2 = 0.0
        cell_l1 = 0.0

    return face_l1, face_l2, face_max, cell_l1, cell_l2, cell_max


# -------------------------
# Main loop
# -------------------------
pyplot.ion()

# -------------------------
# Divergence diagnostics
# -------------------------
div_check_interval = 1000      # plot/print every 100 steps
div_save_interval  = 1000      # save CSV every 1000 steps

div_time_hist = []
div_face_l1_hist = []
div_face_l2_hist = []
div_face_linf_hist = []
div_cell_l1_hist = []
div_cell_l2_hist = []
div_cell_linf_hist = []

# RK4 stage-3 phi residual history is NOT stored for every time step.
# Only the current/latest residual arrays are kept, so Spyder Variable Explorer
# does not try to save tens of thousands of residual-history objects.
latest_rk3_phi_residual = np.zeros(max_phi_iters, dtype=np.float64)
latest_rk3_phi_change   = np.zeros(max_phi_iters, dtype=np.float64)

# First RK4 projection iteration history, sampled throughout the full run.
phi_stage1_iter_time_hist = []
phi_stage1_iter_count_hist = []

# Steady-state history.
steady_time_hist = []
steady_rel_l2_hist = []
steady_rms_change_hist = []
steady_linf_change_hist = []
steady_consecutive_count = 0
steady_reached_time = -1

# Initial reference field for the first steady-state comparison.
u_prev[:, :] = u
v_prev[:, :] = v

for time in range(total_time_steps):

    # Retained only because one_step/solve_pressure_and_velocity still accepts
    # ter_stage. It is NOT used while use_phi_tolerance == 1.
    ter_stage = 300

    # Residual-based stopping is ON for every projection for the entire run.
    use_phi_tolerance = 1

    stage1_phi_iters, stage1_phi_residual, final_phi_iters, final_phi_residual, rk3_phi_iters, rk3_phi_final_residual = one_step(u, v, q1, q2, p,
             q1f, q2f, uf, vf,
             q1s, q2s,
             p_old, rhs,
             k1_q1, k1_q2, k2_q1, k2_q2, k3_q1, k3_q2, k4_q1, k4_q2,
             q1a, q2a,
             rk3_phi_residual_iter, rk3_phi_change_iter,
             n, d, delt, k1c, k2c, vel,
             ter_stage, omega,
             use_phi_tolerance, phi_tolerance)

    # ------------------------------------------------------------
    # First RK4 projection phi-iteration convergence study.
    # Sample across the ENTIRE run.
    # ------------------------------------------------------------
    if time % phi_iteration_sample_interval == 0:
        phi_stage1_iter_time_hist.append(time)
        phi_stage1_iter_count_hist.append(stage1_phi_iters)

        # Live thesis-style plot: iterations required vs time step.
        pyplot.figure(5)
        pyplot.plot(phi_stage1_iter_time_hist, phi_stage1_iter_count_hist, marker='o', markersize=3, linewidth=1.2)
        pyplot.xlabel("Time step")
        pyplot.ylabel("First RK4 projection SOR iterations")
        pyplot.title(r"First RK4 projection iterations to $\|R_\phi\|_\infty \leq 10^{-3}$")
        pyplot.grid(True)
        pyplot.pause(0.00001)
        pyplot.cla()

        # Save the sampled data.
        with open("rk4_stage1_phi_iterations_sampled.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_step", "rk4_stage1_phi_iterations"])
            for idx in range(len(phi_stage1_iter_time_hist)):
                writer.writerow([phi_stage1_iter_time_hist[idx], phi_stage1_iter_count_hist[idx]])

        # Save/overwrite a high-resolution figure containing all samples so far.
        pyplot.figure(32)
        pyplot.plot(phi_stage1_iter_time_hist, phi_stage1_iter_count_hist, marker='o', markersize=3, linewidth=1.2)
        pyplot.xlabel("Time step")
        pyplot.ylabel("First RK4 projection SOR iterations")
        pyplot.title(r"First RK4 projection iterations to $\|R_\phi\|_\infty \leq 10^{-3}$")
        pyplot.grid(True)
        pyplot.savefig("rk4_stage1_phi_iterations_vs_timestep.png", dpi=300, bbox_inches="tight")
        pyplot.clf()

    # ------------------------------------------------------------
    # Steady-state tracker after EVERY time step.
    # ------------------------------------------------------------
    steady_rel_l2, steady_rms_change, steady_linf_change = compute_steady_state_metrics(
        u, v, u_prev, v_prev, n
    )

    if steady_rel_l2 <= steady_state_tolerance:
        steady_consecutive_count += 1
    else:
        steady_consecutive_count = 0

    if (steady_reached_time < 0 and
            steady_consecutive_count >= steady_state_required_consecutive):
        steady_reached_time = time - steady_state_required_consecutive + 1
        print(
            "STEADY STATE DETECTED: first sustained step =", steady_reached_time,
            "current step =", time,
            "relative velocity L2 change =", steady_rel_l2
        )

    if time % steady_state_sample_interval == 0:
        steady_time_hist.append(time)
        steady_rel_l2_hist.append(steady_rel_l2)
        steady_rms_change_hist.append(steady_rms_change)
        steady_linf_change_hist.append(steady_linf_change)

    # Update reference fields only AFTER computing the current-step change.
    u_prev[:, :] = u
    v_prev[:, :] = v

    # ------------------------------------------------------------
    # Divergence checker after EVERY timestep
    # face divergence is the important one because it matches your
    # pressure-projection/Rhie-Chow face flux formulation.
    # ------------------------------------------------------------
    div_face_l1, div_face_l2, div_face_linf, div_cell_l1, div_cell_l2, div_cell_linf = compute_divergence_metrics(
        u, v, uf, vf, d, n
    )

    div_time_hist.append(time)
    div_face_l1_hist.append(div_face_l1)
    div_face_l2_hist.append(div_face_l2)
    div_face_linf_hist.append(div_face_linf)
    div_cell_l1_hist.append(div_cell_l1)
    div_cell_l2_hist.append(div_cell_l2)
    div_cell_linf_hist.append(div_cell_linf)

    # Keep ONLY the latest RK4 3rd-stage phi residual in memory.
    # This preserves the latest residual plot without filling Spyder with 60,000+ objects.
    latest_rk3_phi_residual[:] = 0.0
    latest_rk3_phi_change[:] = 0.0
    rk3_phi_stored_iters = min(rk3_phi_iters, max_phi_iters)
    latest_rk3_phi_residual[:rk3_phi_stored_iters] = rk3_phi_residual_iter[:rk3_phi_stored_iters]
    latest_rk3_phi_change[:rk3_phi_stored_iters]   = rk3_phi_change_iter[:rk3_phi_stored_iters]

    # Console print in ANSYS-style format, but with BOTH counters:
    # - local_phi_iteration = iteration inside the current time step
    # - global_phi_iteration = continuous total SOR iteration count over the whole run
    if print_phi_residual_each_iteration and (time % phi_residual_print_interval == 0):
        print("\nRK4 3rd-stage phi residual table for time step", time)
        print("time_step, local_phi_iteration, printed_iteration, poisson_residual, phi_change")
        for local_it in range(rk3_phi_stored_iters):
            print(time, local_it + 1, local_it + 1, rk3_phi_residual_iter[local_it], rk3_phi_change_iter[local_it])

    if time % div_check_interval == 0:
        compute_F(F, u, v, n)

        print(
            "Time:", time,
            "ter_stage:", ter_stage,
            "omega:", omega,
            "face_div_L2:", div_face_l2,
            "face_div_Linf:", div_face_linf,
            "cell_div_L2:", div_cell_l2,
            "cell_div_Linf:", div_cell_linf,
            "stage1_phi_iters:", stage1_phi_iters,
            "stage1_phi_residual:", stage1_phi_residual,
            "rk3_phi_final_residual:", rk3_phi_final_residual,
            "final_phi_residual:", final_phi_residual,
            "steady_rel_L2:", steady_rel_l2,
            "steady_Linf_change:", steady_linf_change,
            "steady_counter:", steady_consecutive_count,
            "steady_reached_time:", steady_reached_time
        )

        pyplot.figure(1)
        pyplot.imshow(F, cmap='jet')
        pyplot.title("Velocity magnitude")
        pyplot.pause(0.00001)
        pyplot.cla()

        pyplot.figure(2)
        pyplot.semilogy(div_time_hist, div_face_l2_hist, label="Face divergence L2")
        pyplot.semilogy(div_time_hist, div_cell_l2_hist, label="Cell divergence L2")
        pyplot.xlabel("Time step")
        pyplot.ylabel("Divergence L2")
        pyplot.title("Divergence after projection")
        pyplot.legend()
        pyplot.pause(0.00001)
        pyplot.cla()

        # Steady-state convergence history.
        pyplot.figure(6)
        pyplot.semilogy(steady_time_hist, steady_rel_l2_hist, linewidth=1.5, label="Relative velocity L2 change")
        pyplot.axhline(steady_state_tolerance, linestyle="--", label="Steady-state threshold")
        pyplot.xlabel("Time step")
        pyplot.ylabel(r"$||U^n-U^{n-1}||_2 / ||U^n||_2$")
        pyplot.title("Steady-state convergence tracker")
        pyplot.legend()
        pyplot.grid(True)
        pyplot.pause(0.00001)
        pyplot.cla()

        # ANSYS-style residual graph removed to avoid the dense blue-wall plot.
        # Full residual data is still saved to CSV below.

        # Local residual plot for the current time step only.
        # x-axis = SOR iteration inside this time step.
        pyplot.figure(4)
        pyplot.semilogy(np.arange(1, rk3_phi_stored_iters + 1), rk3_phi_residual_iter[:rk3_phi_stored_iters], linewidth=1.5, label="current time step")
        pyplot.xlabel("Local phi/SOR iteration inside current time step")
        pyplot.ylabel("max |RHS - lap(phi)|")
        pyplot.title("RK4 3rd-stage phi residual inside current time step")
        pyplot.legend()
        pyplot.grid(True)
        pyplot.pause(0.00001)
        pyplot.cla()

    if time % div_save_interval == 0:
        with open("divergence_history.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time_step",
                "face_div_L1",
                "face_div_L2",
                "face_div_Linf",
                "cell_div_L1",
                "cell_div_L2",
                "cell_div_Linf"
            ])
            for idx in range(len(div_time_hist)):
                writer.writerow([
                    div_time_hist[idx],
                    div_face_l1_hist[idx],
                    div_face_l2_hist[idx],
                    div_face_linf_hist[idx],
                    div_cell_l1_hist[idx],
                    div_cell_l2_hist[idx],
                    div_cell_linf_hist[idx]
                ])

        # Save steady-state history.
        with open("steady_state_history.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time_step",
                "relative_velocity_L2_change",
                "rms_velocity_change",
                "linf_velocity_change",
                "steady_state_tolerance",
                "steady_reached_time"
            ])
            for idx in range(len(steady_time_hist)):
                writer.writerow([
                    steady_time_hist[idx],
                    steady_rel_l2_hist[idx],
                    steady_rms_change_hist[idx],
                    steady_linf_change_hist[idx],
                    steady_state_tolerance,
                    steady_reached_time
                ])

        pyplot.figure(33)
        pyplot.semilogy(steady_time_hist, steady_rel_l2_hist, linewidth=1.5, label="Relative velocity L2 change")
        pyplot.axhline(steady_state_tolerance, linestyle="--", label="Steady-state threshold")
        pyplot.xlabel("Time step")
        pyplot.ylabel(r"$||U^n-U^{n-1}||_2 / ||U^n||_2$")
        pyplot.title("Steady-state convergence tracker")
        pyplot.legend()
        pyplot.grid(True)
        pyplot.savefig("steady_state_history.png", dpi=300, bbox_inches="tight")
        pyplot.clf()

        # Full RK3 phi residual history CSVs removed on purpose.
        # Keeping only divergence_history.csv plus the latest local residual PNG.

        # Save the latest time step local residual graph.
        pyplot.figure(31)
        pyplot.semilogy(np.arange(1, rk3_phi_stored_iters + 1), rk3_phi_residual_iter[:rk3_phi_stored_iters], linewidth=1.5, label="latest time step")
        pyplot.xlabel("Local phi/SOR iteration inside time step")
        pyplot.ylabel("max |RHS - lap(phi)|")
        plt.title(f"RK4 3rd-stage phi residual inside latest time step: step = {time}")
        pyplot.legend()
        pyplot.grid(True)
        pyplot.savefig("rk3_phi_poisson_residual_latest_timestep_local.png", dpi=300, bbox_inches="tight")
        pyplot.clf()