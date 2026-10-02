# -*- coding: utf-8 -*-
"""
CUDA 3-D extension of the user's 2-D impulse + Smagorinsky cavity solver.

DESIGN RULE FOR THIS FILE
-------------------------
Keep the numerical structure of the 2-D solver and add only the third axis:

2-D:
    q1,q2 -> cell-centered advective RHS -> RK4 stage
           -> q face interpolation -> Poisson(phi)
           -> face velocity -> cell velocity

3-D here:
    q1,q2,q3 -> SAME cell-centered advective RHS + z terms -> RK4 stage
             -> q1f,q2f,q3f face interpolation -> 3-D Poisson(phi)
             -> uf,vf,wf face velocity -> u,v,w cell velocity

The flow solver keeps the 2-D numerical structure, but the LES source uses
the requested 3-D velocity-based symmetric Smagorinsky stress tensor:
    div{nu_t [grad(u) + grad(u)^T]}
with face-averaged nu_t.

The code intentionally DOES NOT use:
    * direct cell recovery u=q-grad(phi)
    * conservative face-flux convection
    * van-Driest damping
    * DCT Poisson solve
    * low-storage RK4

Those are all different from the reference 2-D algorithm.

Index convention:
    field[i,j,k] = field[y,x,z]
    top lid: i=1, moving +x with u=vel
    physical walls: indices 1 and n-1
    outer ghosts: indices 0 and n
    active cell centers: 2..n-2

A small divergence-free perturbation is added ONCE at t=0 to break exact
spanwise symmetry. It is not a source term and is not continuously forced.

PERFORMANCE-ONLY CHANGES IN THIS VERSION (V3):
    * 128-thread (32,4,1) blocks: full k-warp with lower register pressure
    * nu_t kernel launches only over its active 1..n-1 cube (no ghost-thread waste)
    * packed red/black SOR launch: no half-warp checkerboard waste
    * one compact pressure-boundary kernel instead of three full 3-D launches
    * algebraically collapse face interpolation + face-velocity reconstruction
      into direct cell-centered projection kernels (same centered stencil)
    * remove q1f/q2f/q3f and uf/vf/wf device arrays entirely
    * stream classical RK4 using one slope triplet + one weighted-sum triplet
      instead of storing all four slope triplets
    * pressure boundary shell is written only after the final RK projection;
      intermediate projections use the identical Neumann substitution directly
    * force hot-kernel constants to float32 so Numba does not promote arithmetic
      to slow float64 on consumer RTX GPUs
    * precompute 1/d, 1/(2d), 1/(4d), d^2 and Smagorinsky coefficient once,
      removing millions of identical per-thread divisions each RK stage
No coefficients, RK4 stage locations, SOR iteration count, BC formulas,
grid size, timestep, viscosity, Cs, or diagnostics were changed.
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda


# ============================================================
# PARAMETERS
# ============================================================
n = 236
vel = 1.0
d = 1.0 / (n - 3)
delt = 0.001
visc = 0.00001           # Re = 30000 for U=L=1
Cs = 0.12

k1c = 1.0 / (2.0 * d)
k2c = visc / (d * d)

omega = 1.90
ter_stage = 75               # keep current value; raise if desired

total_steps = 100000
print_interval = 1000
plot_interval = 1000
ENABLE_PLOTTING = True

PERTURB_AMPLITUDE = 0

DTYPE = np.float32

# IMPORTANT: keep arithmetic inside CUDA kernels in FP32.  Plain Python literals
# such as 0.5, 1.0, 2.0, 6.0 are float64 and can promote Numba expressions.
# On an RTX 3060 that is extremely expensive.
F0 = np.float32(0.0)
F025 = np.float32(0.25)
F05 = np.float32(0.5)
F1 = np.float32(1.0)
F2 = np.float32(2.0)
F3 = np.float32(3.0)
F4 = np.float32(4.0)
F6 = np.float32(6.0)

# Precompute invariants once on the CPU; never divide by grid spacing per CUDA thread.
VEL32 = np.float32(vel)
DT32 = np.float32(delt)
HALF_DT32 = np.float32(0.5*delt)
INV_D32 = np.float32(1.0/d)
INV_2D32 = np.float32(0.5/d)
INV_4D32 = np.float32(0.25/d)
D2_32 = np.float32(d*d)
OMEGA32 = np.float32(omega)
K1C32 = np.float32(k1c)
K2C32 = np.float32(k2c)
SMAG_COEFF32 = np.float32((Cs*d)*(Cs*d))

# CUDA x-index -> k (contiguous NumPy axis), then j, then i.
TPB = (32, 4, 1)             # 128 threads/block; full k-warp, wider j coverage; same arithmetic
TPB_SURF = (32, 8)            # compact 2-D wall/surface kernels
TPB_EDGE = 128                # compact 1-D edge kernels
TPB_CORNER = 8                # 8 cube corners


# ============================================================
# LAUNCH HELPERS
# ============================================================
def blocks_full(n):
    N = n + 1
    return (
        (N + TPB[0] - 1) // TPB[0],
        (N + TPB[1] - 1) // TPB[1],
        (N + TPB[2] - 1) // TPB[2],
    )


def blocks_active(n):
    m = n - 1
    return (
        (m + TPB[0] - 1) // TPB[0],
        (m + TPB[1] - 1) // TPB[1],
        (m + TPB[2] - 1) // TPB[2],
    )


def blocks_interior(n):
    m = n - 3
    return (
        (m + TPB[0] - 1) // TPB[0],
        (m + TPB[1] - 1) // TPB[1],
        (m + TPB[2] - 1) // TPB[2],
    )


def blocks_sor(n):
    """Launch only the checkerboard color being updated (same RB-SOR math)."""
    m = n - 3
    mh = (m + 1) // 2
    return (
        (mh + TPB[0] - 1) // TPB[0],
        (m  + TPB[1] - 1) // TPB[1],
        (m  + TPB[2] - 1) // TPB[2],
    )


def blocks_surface(n):
    m = n - 3
    return (
        (m + TPB_SURF[0] - 1) // TPB_SURF[0],
        (m + TPB_SURF[1] - 1) // TPB_SURF[1],
    )


def blocks_edge(n):
    m = n - 3
    return (m + TPB_EDGE - 1) // TPB_EDGE


def pressure_boundary_blocks(n):
    m = n - 3
    count = 6*m*m + 12*m + 8
    return (count + 255) // 256


def physical_boundary_blocks(n):
    # Six physical faces, including duplicate edge/corner visits.
    # Duplicate writers store exactly the same final value.
    count = 6*(n-1)*(n-1)
    return (count + 255) // 256


def outer_ghost_blocks(n):
    # Six outer faces, including duplicate edge/corner visits.
    count = 6*(n+1)*(n+1)
    return (count + 255) // 256


def q_face_boundary_blocks(n):
    m = n - 3
    return (6*m*m + 255) // 256


def q_edge_boundary_blocks(n):
    m = n - 3
    return (12*m + 255) // 256


def q_shell_copy_blocks(n):
    # Six physical faces; duplicates at edges/corners are identical copies.
    count = 6*(n-1)*(n-1)
    return (count + 255) // 256


# ============================================================
# INITIAL 3-D PERTURBATION
# ============================================================
@cuda.jit(fastmath=True)
def initialize_3d_perturbation(u, v, w, q1, q2, q3, amp, d, n):
    kk, jj, ii = cuda.grid(3)
    i = ii + 2
    j = jj + 2
    k = kk + 2

    if i < n - 1 and j < n - 1 and k < n - 1:
        # Active-cell coordinates.  Smooth and zero at all physical walls.
        x = (j - 1.5) * d
        y = (i - 1.5) * d
        z = (k - 1.5) * d

        sx = math.sin(math.pi * x)
        sy = math.sin(math.pi * y)
        sz = math.sin(math.pi * z)

        up = amp * sx*sx * sy*sy * math.sin(2.0 * math.pi * z)
        wp = -amp * math.sin(2.0 * math.pi * x) * sy*sy * sz*sz

        # d(up)/dx + d(wp)/dz = 0 analytically.
        u[i,j,k] = up
        v[i,j,k] = 0.0
        w[i,j,k] = wp

        # phi=0 initially -> q=u is consistent.
        q1[i,j,k] = up
        q2[i,j,k] = 0.0
        q3[i,j,k] = wp


# ============================================================
# VELOCITY BOUNDARIES
# Exact 3-D extension of the 2-D boundary-node convention:
# physical walls are stored at indices 1 and n-1; top lid wins at its edges.
# ============================================================
@cuda.jit
def set_velocity_physical_boundaries(u, v, w, lid_vel, n):
    k, j, i = cuda.grid(3)
    if i > n or j > n or k > n:
        return

    if 1 <= i <= n-1 and 1 <= j <= n-1 and 1 <= k <= n-1:
        on_wall = (i == 1 or i == n-1 or
                   j == 1 or j == n-1 or
                   k == 1 or k == n-1)
        if on_wall:
            u[i,j,k] = 0.0
            v[i,j,k] = 0.0
            w[i,j,k] = 0.0

        # Same convention as the 2-D code: moving lid value is stored directly.
        # Applied last so the lid wins at the geometric lid edges/corners.
        if i == 1:
            u[i,j,k] = lid_vel
            v[i,j,k] = 0.0
            w[i,j,k] = 0.0


@cuda.jit
def copy_velocity_outer_ghosts(u, v, w, n):
    k, j, i = cuda.grid(3)
    if i > n or j > n or k > n:
        return

    if i == 0 or i == n or j == 0 or j == n or k == 0 or k == n:
        ii = 1 if i == 0 else (n-1 if i == n else i)
        jj = 1 if j == 0 else (n-1 if j == n else j)
        kk = 1 if k == 0 else (n-1 if k == n else k)
        u[i,j,k] = u[ii,jj,kk]
        v[i,j,k] = v[ii,jj,kk]
        w[i,j,k] = w[ii,jj,kk]


# ============================================================
# COMPACT VELOCITY BOUNDARY KERNELS -- performance only
# ============================================================
@cuda.jit
def set_velocity_physical_boundaries_compact(u, v, w, lid_vel, n):
    idx = cuda.grid(1)
    side = n - 1
    per_face = side*side
    total = 6*per_face
    if idx >= total:
        return

    face = idx // per_face
    rem = idx - face*per_face
    a = rem // side + 1
    b = rem - (rem // side)*side + 1

    if face == 0:
        i = 1;   j = a;   k = b
    elif face == 1:
        i = n-1; j = a;   k = b
    elif face == 2:
        i = a;   j = 1;   k = b
    elif face == 3:
        i = a;   j = n-1; k = b
    elif face == 4:
        i = a;   j = b;   k = 1
    else:
        i = a;   j = b;   k = n-1

    # Same final rule as the original 3-D boundary kernel: lid wins.
    if i == 1:
        u[i,j,k] = lid_vel
    else:
        u[i,j,k] = 0.0
    v[i,j,k] = 0.0
    w[i,j,k] = 0.0


@cuda.jit
def copy_velocity_outer_ghosts_compact(u, v, w, n):
    idx = cuda.grid(1)
    side = n + 1
    per_face = side*side
    total = 6*per_face
    if idx >= total:
        return

    face = idx // per_face
    rem = idx - face*per_face
    a = rem // side
    b = rem - (rem // side)*side

    if face == 0:
        i = 0; j = a; k = b
    elif face == 1:
        i = n; j = a; k = b
    elif face == 2:
        i = a; j = 0; k = b
    elif face == 3:
        i = a; j = n; k = b
    elif face == 4:
        i = a; j = b; k = 0
    else:
        i = a; j = b; k = n

    ii = 1 if i == 0 else (n-1 if i == n else i)
    jj = 1 if j == 0 else (n-1 if j == n else j)
    kk = 1 if k == 0 else (n-1 if k == n else k)
    u[i,j,k] = u[ii,jj,kk]
    v[i,j,k] = v[ii,jj,kk]
    w[i,j,k] = w[ii,jj,kk]


# ============================================================
# q BOUNDARY CONDITIONS
# Direct 3-D extension of the 2-D impulse BC pattern.
# Face interiors use the same reflection + tangential phi-gradient rule.
# 3-D edges/corners are filled from neighboring face values afterward.
# ============================================================
@cuda.jit
def set_q_face_boundaries(q1, q2, q3, p, lid_vel, d, n):
    k, j, i = cuda.grid(3)
    if i > n or j > n or k > n:
        return

    inv_d = F1 / d

    # y-normal walls: i = 1, n-1; require j,k away from edges.
    if 2 <= j < n-1 and 2 <= k < n-1:
        if i == 1:
            q1[i,j,k] = (-q1[2,j,k] + F2*lid_vel
                         + (p[i,j+1,k] - p[i,j-1,k])*inv_d)
            q2[i,j,k] = -q2[2,j,k]
            q3[i,j,k] = (-q3[2,j,k]
                         + (p[i,j,k+1] - p[i,j,k-1])*inv_d)
        elif i == n-1:
            q1[i,j,k] = (-q1[n-2,j,k]
                         + (p[i,j+1,k] - p[i,j-1,k])*inv_d)
            q2[i,j,k] = -q2[n-2,j,k]
            q3[i,j,k] = (-q3[n-2,j,k]
                         + (p[i,j,k+1] - p[i,j,k-1])*inv_d)

    # x-normal walls: j = 1, n-1; require i,k away from edges.
    if 2 <= i < n-1 and 2 <= k < n-1:
        if j == 1:
            q1[i,j,k] = -q1[i,2,k]
            q2[i,j,k] = (-q2[i,2,k]
                         + (p[i+1,j,k] - p[i-1,j,k])*inv_d)
            q3[i,j,k] = (-q3[i,2,k]
                         + (p[i,j,k+1] - p[i,j,k-1])*inv_d)
        elif j == n-1:
            q1[i,j,k] = -q1[i,n-2,k]
            q2[i,j,k] = (-q2[i,n-2,k]
                         + (p[i+1,j,k] - p[i-1,j,k])*inv_d)
            q3[i,j,k] = (-q3[i,n-2,k]
                         + (p[i,j,k+1] - p[i,j,k-1])*inv_d)

    # z-normal walls: k = 1, n-1; require i,j away from edges.
    if 2 <= i < n-1 and 2 <= j < n-1:
        if k == 1:
            q1[i,j,k] = (-q1[i,j,2]
                         + (p[i,j+1,k] - p[i,j-1,k])*inv_d)
            q2[i,j,k] = (-q2[i,j,2]
                         + (p[i+1,j,k] - p[i-1,j,k])*inv_d)
            q3[i,j,k] = -q3[i,j,2]
        elif k == n-1:
            q1[i,j,k] = (-q1[i,j,n-2]
                         + (p[i,j+1,k] - p[i,j-1,k])*inv_d)
            q2[i,j,k] = (-q2[i,j,n-2]
                         + (p[i+1,j,k] - p[i-1,j,k])*inv_d)
            q3[i,j,k] = -q3[i,j,n-2]


@cuda.jit
def fill_q_edges(q1, q2, q3, n):
    """Fill wall edges after the six face interiors have been updated."""
    k, j, i = cuda.grid(3)
    if i > n or j > n or k > n:
        return
    if not (1 <= i <= n-1 and 1 <= j <= n-1 and 1 <= k <= n-1):
        return

    bi = (i == 1 or i == n-1)
    bj = (j == 1 or j == n-1)
    bk = (k == 1 or k == n-1)
    count = int(bi) + int(bj) + int(bk)
    if count != 2:
        return

    s1 = 0.0; s2 = 0.0; s3 = 0.0
    if bi:
        ii = 2 if i == 1 else n-2
        s1 += q1[ii,j,k]; s2 += q2[ii,j,k]; s3 += q3[ii,j,k]
    if bj:
        jj = 2 if j == 1 else n-2
        s1 += q1[i,jj,k]; s2 += q2[i,jj,k]; s3 += q3[i,jj,k]
    if bk:
        kk = 2 if k == 1 else n-2
        s1 += q1[i,j,kk]; s2 += q2[i,j,kk]; s3 += q3[i,j,kk]

    q1[i,j,k] = F05*s1
    q2[i,j,k] = F05*s2
    q3[i,j,k] = F05*s3


@cuda.jit
def fill_q_corners(q1, q2, q3, n):
    """Fill 8 wall corners after edge values are available."""
    k, j, i = cuda.grid(3)
    if i > n or j > n or k > n:
        return
    if not (1 <= i <= n-1 and 1 <= j <= n-1 and 1 <= k <= n-1):
        return

    bi = (i == 1 or i == n-1)
    bj = (j == 1 or j == n-1)
    bk = (k == 1 or k == n-1)
    if int(bi) + int(bj) + int(bk) != 3:
        return

    ii = 2 if i == 1 else n-2
    jj = 2 if j == 1 else n-2
    kk = 2 if k == 1 else n-2

    q1[i,j,k] = (q1[ii,j,k] + q1[i,jj,k] + q1[i,j,kk]) / F3
    q2[i,j,k] = (q2[ii,j,k] + q2[i,jj,k] + q2[i,j,kk]) / F3
    q3[i,j,k] = (q3[ii,j,k] + q3[i,jj,k] + q3[i,j,kk]) / F3


# ============================================================
# COMPACT q BOUNDARY KERNELS -- same values, far fewer idle threads
# ============================================================
@cuda.jit
def set_q_face_boundaries_compact(q1, q2, q3, p, lid_vel, inv_d, n):
    idx = cuda.grid(1)
    m = n - 3
    per_face = m*m
    total = 6*per_face
    if idx >= total:
        return

    face = idx // per_face
    rem = idx - face*per_face
    a = rem // m + 2
    b = rem - (rem // m)*m + 2
    if face == 0:  # i = 1
        i=1; j=a; k=b
        q1[i,j,k] = -q1[2,j,k] + F2*lid_vel + (p[i,j+1,k]-p[i,j-1,k])*inv_d
        q2[i,j,k] = -q2[2,j,k]
        q3[i,j,k] = -q3[2,j,k] + (p[i,j,k+1]-p[i,j,k-1])*inv_d
    elif face == 1:  # i = n-1
        i=n-1; j=a; k=b
        q1[i,j,k] = -q1[n-2,j,k] + (p[i,j+1,k]-p[i,j-1,k])*inv_d
        q2[i,j,k] = -q2[n-2,j,k]
        q3[i,j,k] = -q3[n-2,j,k] + (p[i,j,k+1]-p[i,j,k-1])*inv_d
    elif face == 2:  # j = 1
        i=a; j=1; k=b
        q1[i,j,k] = -q1[i,2,k]
        q2[i,j,k] = -q2[i,2,k] + (p[i+1,j,k]-p[i-1,j,k])*inv_d
        q3[i,j,k] = -q3[i,2,k] + (p[i,j,k+1]-p[i,j,k-1])*inv_d
    elif face == 3:  # j = n-1
        i=a; j=n-1; k=b
        q1[i,j,k] = -q1[i,n-2,k]
        q2[i,j,k] = -q2[i,n-2,k] + (p[i+1,j,k]-p[i-1,j,k])*inv_d
        q3[i,j,k] = -q3[i,n-2,k] + (p[i,j,k+1]-p[i,j,k-1])*inv_d
    elif face == 4:  # k = 1
        i=a; j=b; k=1
        q1[i,j,k] = -q1[i,j,2] + (p[i,j+1,k]-p[i,j-1,k])*inv_d
        q2[i,j,k] = -q2[i,j,2] + (p[i+1,j,k]-p[i-1,j,k])*inv_d
        q3[i,j,k] = -q3[i,j,2]
    else:  # k = n-1
        i=a; j=b; k=n-1
        q1[i,j,k] = -q1[i,j,n-2] + (p[i,j+1,k]-p[i,j-1,k])*inv_d
        q2[i,j,k] = -q2[i,j,n-2] + (p[i+1,j,k]-p[i-1,j,k])*inv_d
        q3[i,j,k] = -q3[i,j,n-2]


@cuda.jit
def fill_q_edges_compact(q1, q2, q3, n):
    idx = cuda.grid(1)
    m = n - 3
    total = 12*m
    if idx >= total:
        return
    edge = idx // m
    t = idx - edge*m + 2
    lo=1; hi=n-1; ilo=2; ihi=n-2

    if edge == 0: i=lo; j=lo; k=t; ii=ilo; jj=ilo; kk=k
    elif edge == 1: i=lo; j=hi; k=t; ii=ilo; jj=ihi; kk=k
    elif edge == 2: i=hi; j=lo; k=t; ii=ihi; jj=ilo; kk=k
    elif edge == 3: i=hi; j=hi; k=t; ii=ihi; jj=ihi; kk=k
    elif edge == 4: i=lo; j=t; k=lo; ii=ilo; jj=j; kk=ilo
    elif edge == 5: i=lo; j=t; k=hi; ii=ilo; jj=j; kk=ihi
    elif edge == 6: i=hi; j=t; k=lo; ii=ihi; jj=j; kk=ilo
    elif edge == 7: i=hi; j=t; k=hi; ii=ihi; jj=j; kk=ihi
    elif edge == 8: i=t; j=lo; k=lo; ii=i; jj=ilo; kk=ilo
    elif edge == 9: i=t; j=lo; k=hi; ii=i; jj=ilo; kk=ihi
    elif edge == 10: i=t; j=hi; k=lo; ii=i; jj=ihi; kk=ilo
    else: i=t; j=hi; k=hi; ii=i; jj=ihi; kk=ihi

    # Exactly the original edge rule: average the two adjacent face-interior values.
    if i == lo or i == hi:
        ai = ilo if i == lo else ihi
        a1=q1[ai,j,k]; a2=q2[ai,j,k]; a3=q3[ai,j,k]
    else:
        a1=F0; a2=F0; a3=F0
    if j == lo or j == hi:
        aj = ilo if j == lo else ihi
        b1=q1[i,aj,k]; b2=q2[i,aj,k]; b3=q3[i,aj,k]
    else:
        b1=F0; b2=F0; b3=F0
    if k == lo or k == hi:
        ak = ilo if k == lo else ihi
        c1=q1[i,j,ak]; c2=q2[i,j,ak]; c3=q3[i,j,ak]
    else:
        c1=F0; c2=F0; c3=F0

    q1[i,j,k] = F05*(a1+b1+c1)
    q2[i,j,k] = F05*(a2+b2+c2)
    q3[i,j,k] = F05*(a3+b3+c3)


@cuda.jit
def fill_q_corners_compact(q1, q2, q3, n):
    c = cuda.grid(1)
    if c >= 8:
        return
    lo=1; hi=n-1; ilo=2; ihi=n-2
    i = lo if (c & 1) == 0 else hi
    j = lo if (c & 2) == 0 else hi
    k = lo if (c & 4) == 0 else hi
    ii = ilo if i == lo else ihi
    jj = ilo if j == lo else ihi
    kk = ilo if k == lo else ihi
    q1[i,j,k] = (q1[ii,j,k] + q1[i,jj,k] + q1[i,j,kk]) / F3
    q2[i,j,k] = (q2[ii,j,k] + q2[i,jj,k] + q2[i,j,kk]) / F3
    q3[i,j,k] = (q3[ii,j,k] + q3[i,jj,k] + q3[i,j,kk]) / F3


@cuda.jit
def copy_q_shell_to_stage(q1a, q2a, q3a, q1, q2, q3, n):
    idx = cuda.grid(1)
    side=n-1
    per_face=side*side
    total=6*per_face
    if idx >= total:
        return
    face=idx//per_face
    rem=idx-face*per_face
    a=rem//side+1
    b=rem-(rem//side)*side+1
    if face == 0: i=1; j=a; k=b
    elif face == 1: i=n-1; j=a; k=b
    elif face == 2: i=a; j=1; k=b
    elif face == 3: i=a; j=n-1; k=b
    elif face == 4: i=a; j=b; k=1
    else: i=a; j=b; k=n-1
    q1a[i,j,k]=q1[i,j,k]
    q2a[i,j,k]=q2[i,j,k]
    q3a[i,j,k]=q3[i,j,k]


# ============================================================
# RHS -- literal 2-D structure + z terms
# ============================================================
@cuda.jit(fastmath=True)
def solve_impulse(out_q1, out_q2, out_q3,
                  q1, q2, q3, u, v, w, nu_t,
                  k1, k2, inv_d, inv_4d, n):
    kk, jj, ii = cuda.grid(3)
    i = ii + 2
    j = jj + 2
    k = kk + 2
    if i >= n-1 or j >= n-1 or k >= n-1:
        return

    # Boundary entries of k-arrays are initialized to zero and are never written.
    # This is identical to the previous full-domain kernel, without idle boundary threads.

    # velocity derivatives, exactly the 2-D centered form + z derivatives
    du_dx = u[i,j+1,k] - u[i,j-1,k]
    du_dy = u[i+1,j,k] - u[i-1,j,k]
    du_dz = u[i,j,k+1] - u[i,j,k-1]

    dv_dx = v[i,j+1,k] - v[i,j-1,k]
    dv_dy = v[i+1,j,k] - v[i-1,j,k]
    dv_dz = v[i,j,k+1] - v[i,j,k-1]

    dw_dx = w[i,j+1,k] - w[i,j-1,k]
    dw_dy = w[i+1,j,k] - w[i-1,j,k]
    dw_dz = w[i,j,k+1] - w[i,j,k-1]

    lap_q1 = (q1[i-1,j,k] + q1[i+1,j,k]
              + q1[i,j-1,k] + q1[i,j+1,k]
              + q1[i,j,k-1] + q1[i,j,k+1]
              - F6*q1[i,j,k])

    lap_q2 = (q2[i-1,j,k] + q2[i+1,j,k]
              + q2[i,j-1,k] + q2[i,j+1,k]
              + q2[i,j,k-1] + q2[i,j,k+1]
              - F6*q2[i,j,k])

    lap_q3 = (q3[i-1,j,k] + q3[i+1,j,k]
              + q3[i,j-1,k] + q3[i,j+1,k]
              + q3[i,j,k-1] + q3[i,j,k+1]
              - F6*q3[i,j,k])

    # Velocity-based symmetric Smagorinsky SGS stress:
    #     div{nu_t [grad(u) + grad(u)^T]}
    # This is the tensor LES model from the 3-D solver, while the rest of the
    # RHS keeps the same cell-centered convection structure as the 2-D code.
    nut_xp = F05*(nu_t[i,j,k] + nu_t[i,j+1,k])
    nut_xm = F05*(nu_t[i,j,k] + nu_t[i,j-1,k])
    nut_yp = F05*(nu_t[i,j,k] + nu_t[i+1,j,k])
    nut_ym = F05*(nu_t[i,j,k] + nu_t[i-1,j,k])
    nut_zp = F05*(nu_t[i,j,k] + nu_t[i,j,k+1])
    nut_zm = F05*(nu_t[i,j,k] + nu_t[i,j,k-1])

    # x-momentum SGS: d/dx[2 nut du/dx]
    #                 + d/dy[nut(du/dy+dv/dx)]
    #                 + d/dz[nut(du/dz+dw/dx)]
    txx_xp = F2*(u[i,j+1,k] - u[i,j,k])*inv_d
    txx_xm = F2*(u[i,j,k] - u[i,j-1,k])*inv_d

    dv_dx_yp = (
        v[i,  j+1,k] - v[i,  j-1,k]
        + v[i+1,j+1,k] - v[i+1,j-1,k]
    ) * inv_4d
    dv_dx_ym = (
        v[i,  j+1,k] - v[i,  j-1,k]
        + v[i-1,j+1,k] - v[i-1,j-1,k]
    ) * inv_4d
    txy_yp = (u[i+1,j,k] - u[i,j,k])*inv_d + dv_dx_yp
    txy_ym = (u[i,j,k] - u[i-1,j,k])*inv_d + dv_dx_ym

    dw_dx_zp = (
        w[i,j+1,k] - w[i,j-1,k]
        + w[i,j+1,k+1] - w[i,j-1,k+1]
    ) * inv_4d
    dw_dx_zm = (
        w[i,j+1,k] - w[i,j-1,k]
        + w[i,j+1,k-1] - w[i,j-1,k-1]
    ) * inv_4d
    txz_zp = (u[i,j,k+1] - u[i,j,k])*inv_d + dw_dx_zp
    txz_zm = (u[i,j,k] - u[i,j,k-1])*inv_d + dw_dx_zm

    les_q1 = (
        nut_xp*txx_xp - nut_xm*txx_xm
        + nut_yp*txy_yp - nut_ym*txy_ym
        + nut_zp*txz_zp - nut_zm*txz_zm
    ) * inv_d

    # y-momentum SGS
    du_dy_xp = (
        u[i+1,j,  k] - u[i-1,j,  k]
        + u[i+1,j+1,k] - u[i-1,j+1,k]
    ) * inv_4d
    du_dy_xm = (
        u[i+1,j,  k] - u[i-1,j,  k]
        + u[i+1,j-1,k] - u[i-1,j-1,k]
    ) * inv_4d
    tyx_xp = (v[i,j+1,k] - v[i,j,k])*inv_d + du_dy_xp
    tyx_xm = (v[i,j,k] - v[i,j-1,k])*inv_d + du_dy_xm

    tyy_yp = F2*(v[i+1,j,k] - v[i,j,k])*inv_d
    tyy_ym = F2*(v[i,j,k] - v[i-1,j,k])*inv_d

    dw_dy_zp = (
        w[i+1,j,k] - w[i-1,j,k]
        + w[i+1,j,k+1] - w[i-1,j,k+1]
    ) * inv_4d
    dw_dy_zm = (
        w[i+1,j,k] - w[i-1,j,k]
        + w[i+1,j,k-1] - w[i-1,j,k-1]
    ) * inv_4d
    tyz_zp = (v[i,j,k+1] - v[i,j,k])*inv_d + dw_dy_zp
    tyz_zm = (v[i,j,k] - v[i,j,k-1])*inv_d + dw_dy_zm

    les_q2 = (
        nut_xp*tyx_xp - nut_xm*tyx_xm
        + nut_yp*tyy_yp - nut_ym*tyy_ym
        + nut_zp*tyz_zp - nut_zm*tyz_zm
    ) * inv_d

    # z-momentum SGS
    du_dz_xp = (
        u[i,j,  k+1] - u[i,j,  k-1]
        + u[i,j+1,k+1] - u[i,j+1,k-1]
    ) * inv_4d
    du_dz_xm = (
        u[i,j,  k+1] - u[i,j,  k-1]
        + u[i,j-1,k+1] - u[i,j-1,k-1]
    ) * inv_4d
    tzx_xp = (w[i,j+1,k] - w[i,j,k])*inv_d + du_dz_xp
    tzx_xm = (w[i,j,k] - w[i,j-1,k])*inv_d + du_dz_xm

    dv_dz_yp = (
        v[i,  j,k+1] - v[i,  j,k-1]
        + v[i+1,j,k+1] - v[i+1,j,k-1]
    ) * inv_4d
    dv_dz_ym = (
        v[i,  j,k+1] - v[i,  j,k-1]
        + v[i-1,j,k+1] - v[i-1,j,k-1]
    ) * inv_4d
    tzy_yp = (w[i+1,j,k] - w[i,j,k])*inv_d + dv_dz_yp
    tzy_ym = (w[i,j,k] - w[i-1,j,k])*inv_d + dv_dz_ym

    tzz_zp = F2*(w[i,j,k+1] - w[i,j,k])*inv_d
    tzz_zm = F2*(w[i,j,k] - w[i,j,k-1])*inv_d

    les_q3 = (
        nut_xp*tzx_xp - nut_xm*tzx_xm
        + nut_yp*tzy_yp - nut_ym*tzy_ym
        + nut_zp*tzz_zp - nut_zm*tzz_zm
    ) * inv_d

    # EXACT 2-D convective structure, only with + w*d()/dz.
    out_q1[i,j,k] = -k1*(u[i,j,k]*du_dx + v[i,j,k]*du_dy + w[i,j,k]*du_dz) + k2*lap_q1 + les_q1
    out_q2[i,j,k] = -k1*(u[i,j,k]*dv_dx + v[i,j,k]*dv_dy + w[i,j,k]*dv_dz) + k2*lap_q2 + les_q2
    out_q3[i,j,k] = -k1*(u[i,j,k]*dw_dx + v[i,j,k]*dw_dy + w[i,j,k]*dw_dz) + k2*lap_q3 + les_q3


@cuda.jit(fastmath=True)
def compute_nu_t(nu_t, u, v, w, inv_2d, smag_coeff, n):
    kk, jj, ii = cuda.grid(3)
    k = kk + 1
    j = jj + 1
    i = ii + 1
    if i < n and j < n and k < n:
        du_dx = (u[i,j+1,k] - u[i,j-1,k]) * inv_2d
        du_dy = (u[i+1,j,k] - u[i-1,j,k]) * inv_2d
        du_dz = (u[i,j,k+1] - u[i,j,k-1]) * inv_2d

        dv_dx = (v[i,j+1,k] - v[i,j-1,k]) * inv_2d
        dv_dy = (v[i+1,j,k] - v[i-1,j,k]) * inv_2d
        dv_dz = (v[i,j,k+1] - v[i,j,k-1]) * inv_2d

        dw_dx = (w[i,j+1,k] - w[i,j-1,k]) * inv_2d
        dw_dy = (w[i+1,j,k] - w[i-1,j,k]) * inv_2d
        dw_dz = (w[i,j,k+1] - w[i,j,k-1]) * inv_2d

        strain_mag = math.sqrt(
            F2*du_dx*du_dx
            + F2*dv_dy*dv_dy
            + F2*dw_dz*dw_dz
            + (du_dy + dv_dx)*(du_dy + dv_dx)
            + (du_dz + dw_dx)*(du_dz + dw_dx)
            + (dv_dz + dw_dy)*(dv_dz + dw_dy)
        )

        nu_t[i,j,k] = smag_coeff * strain_mag


# ============================================================
# CLASSICAL RK4 -- streamed storage, same four RK4 stage locations
# ============================================================
@cuda.jit(fastmath=True)
def rk_stage_init(q1a, q2a, q3a,
                  a1, a2, a3,
                  q1, q2, q3,
                  k1, k2, k3,
                  factor, n):
    """Stage 1: q_stage = q + factor*k1 and accumulator = k1."""
    kk, jj, ii = cuda.grid(3)
    i = ii + 2
    j = jj + 2
    k = kk + 2
    if i < n-1 and j < n-1 and k < n-1:
        s1 = k1[i,j,k]
        s2 = k2[i,j,k]
        s3 = k3[i,j,k]
        a1[i,j,k] = s1
        a2[i,j,k] = s2
        a3[i,j,k] = s3
        q1a[i,j,k] = q1[i,j,k] + factor*s1
        q2a[i,j,k] = q2[i,j,k] + factor*s2
        q3a[i,j,k] = q3[i,j,k] + factor*s3


@cuda.jit(fastmath=True)
def rk_stage_accum(q1a, q2a, q3a,
                   a1, a2, a3,
                   q1, q2, q3,
                   k1, k2, k3,
                   factor, weight, n):
    """Stages 2/3: accumulate weighted slope while forming the next stage."""
    kk, jj, ii = cuda.grid(3)
    i = ii + 2
    j = jj + 2
    k = kk + 2
    if i < n-1 and j < n-1 and k < n-1:
        s1 = k1[i,j,k]
        s2 = k2[i,j,k]
        s3 = k3[i,j,k]
        a1[i,j,k] += weight*s1
        a2[i,j,k] += weight*s2
        a3[i,j,k] += weight*s3
        q1a[i,j,k] = q1[i,j,k] + factor*s1
        q2a[i,j,k] = q2[i,j,k] + factor*s2
        q3a[i,j,k] = q3[i,j,k] + factor*s3


@cuda.jit(fastmath=True)
def rk_final_streamed(q1, q2, q3,
                      a1, a2, a3,
                      k1, k2, k3,
                      dt, n):
    """Final classical RK4 update: q += dt/6*(k1 + 2k2 + 2k3 + k4)."""
    kk, jj, ii = cuda.grid(3)
    i = ii + 2
    j = jj + 2
    k = kk + 2
    if i < n-1 and j < n-1 and k < n-1:
        s = dt / F6
        q1[i,j,k] += s*(a1[i,j,k] + k1[i,j,k])
        q2[i,j,k] += s*(a2[i,j,k] + k2[i,j,k])
        q3[i,j,k] += s*(a3[i,j,k] + k3[i,j,k])


# ============================================================
# PRESSURE RHS -- algebraically identical to face interpolation + divergence
# ============================================================
@cuda.jit(fastmath=True)
def build_pressure_rhs_direct(rhs, q1, q2, q3, inv_2d, n):
    """
    Original chain used q-face arithmetic averages followed by face divergence.
    Algebraically this is the centered divergence below, so we avoid writing and
    rereading three full face arrays every projection.
    """
    kk, jj, ii = cuda.grid(3)
    i = ii + 2
    j = jj + 2
    k = kk + 2

    if i < n-1 and j < n-1 and k < n-1:
        rhs[i,j,k] = inv_2d * (
            (q1[i,j+1,k] - q1[i,j-1,k])
            + (q2[i+1,j,k] - q2[i-1,j,k])
            + (q3[i,j,k+1] - q3[i,j,k-1])
        )


# ============================================================
# 3-D Poisson SOR -- same 7-point extension of the 2-D equation
# ============================================================
@cuda.jit(fastmath=True)
def rb_sor(p, rhs, d2, omega, n, color):
    """
    Same packed red/black SOR update, with homogeneous-Neumann face values
    substituted directly into the stencil.  This removes the boundary-shell
    kernel from every SOR iteration without changing the SOR equation.
    """
    hk, jj, ii = cuda.grid(3)
    i = ii + 2
    j = jj + 2

    if i >= n-1 or j >= n-1:
        return

    kp = (color ^ ((i + j) & 1))
    k = 2 + 2*hk + kp
    if k >= n-1:
        return

    pc = p[i,j,k]
    pim = pc if i == 2   else p[i-1,j,k]
    pip = pc if i == n-2 else p[i+1,j,k]
    pjm = pc if j == 2   else p[i,j-1,k]
    pjp = pc if j == n-2 else p[i,j+1,k]
    pkm = pc if k == 2   else p[i,j,k-1]
    pkp = pc if k == n-2 else p[i,j,k+1]

    p_new = (pip + pim + pjp + pjm + pkp + pkm - d2*rhs[i,j,k]) / F6
    p[i,j,k] = (F1 - omega)*pc + omega*p_new


@cuda.jit
def set_pressure_boundaries(p, n):
    """
    Final result is exactly the same as the old face -> edge -> corner sequence,
    but all boundary cells are independent here because the homogeneous-Neumann
    chain reduces to the same nearest interior value.  Edges/corners retain the
    original arithmetic (0.5*(x+x), (x+x+x)/3) so the stored boundary state is
    numerically equivalent after each SOR iteration.
    """
    idx = cuda.grid(1)
    m = n - 3
    face_n = 6*m*m
    edge_n = 12*m
    total = face_n + edge_n + 8
    if idx >= total:
        return

    lo = 1
    hi = n - 1
    ilo = 2
    ihi = n - 2

    # Six face interiors: unique cells, no overlap with edges/corners.
    if idx < face_n:
        face = idx // (m*m)
        rem = idx - face*(m*m)
        a = rem // m + 2
        b = rem - (rem // m)*m + 2

        if face == 0:
            p[lo, a, b] = p[ilo, a, b]
        elif face == 1:
            p[hi, a, b] = p[ihi, a, b]
        elif face == 2:
            p[a, lo, b] = p[a, ilo, b]
        elif face == 3:
            p[a, hi, b] = p[a, ihi, b]
        elif face == 4:
            p[a, b, lo] = p[a, b, ilo]
        else:
            p[a, b, hi] = p[a, b, ihi]
        return

    idx2 = idx - face_n

    # Twelve edge interiors.  After the old face pass, both values entering
    # each edge average are the same nearest interior value x.
    if idx2 < edge_n:
        edge = idx2 // m
        t = idx2 - edge*m + 2

        if edge == 0:
            x = p[ilo, ilo, t]; p[lo, lo, t] = F05*(x + x)
        elif edge == 1:
            x = p[ilo, ihi, t]; p[lo, hi, t] = F05*(x + x)
        elif edge == 2:
            x = p[ihi, ilo, t]; p[hi, lo, t] = F05*(x + x)
        elif edge == 3:
            x = p[ihi, ihi, t]; p[hi, hi, t] = F05*(x + x)
        elif edge == 4:
            x = p[ilo, t, ilo]; p[lo, t, lo] = F05*(x + x)
        elif edge == 5:
            x = p[ilo, t, ihi]; p[lo, t, hi] = F05*(x + x)
        elif edge == 6:
            x = p[ihi, t, ilo]; p[hi, t, lo] = F05*(x + x)
        elif edge == 7:
            x = p[ihi, t, ihi]; p[hi, t, hi] = F05*(x + x)
        elif edge == 8:
            x = p[t, ilo, ilo]; p[t, lo, lo] = F05*(x + x)
        elif edge == 9:
            x = p[t, ilo, ihi]; p[t, lo, hi] = F05*(x + x)
        elif edge == 10:
            x = p[t, ihi, ilo]; p[t, hi, lo] = F05*(x + x)
        else:
            x = p[t, ihi, ihi]; p[t, hi, hi] = F05*(x + x)
        return

    # Eight corners.  The old edge pass makes all three terms identical to x.
    c = idx2 - edge_n
    i = lo if (c & 1) == 0 else hi
    j = lo if (c & 2) == 0 else hi
    k = lo if (c & 4) == 0 else hi
    ii = ilo if i == lo else ihi
    jj = ilo if j == lo else ihi
    kk = ilo if k == lo else ihi
    x = p[ii, jj, kk]
    p[i, j, k] = (x + x + x) / F3


# ============================================================
# DIRECT CELL VELOCITY RECOVERY
# Same arithmetic stencil as: q -> face averages -> projected face u -> cell average.
# Homogeneous-Neumann pressure values are substituted directly at boundary-adjacent cells.
# ============================================================
@cuda.jit(fastmath=True)
def reconstruct_cell_velocity_direct(u, v, w, q1, q2, q3, p, inv_2d, n):
    kk, jj, ii = cuda.grid(3)
    i = ii + 2
    j = jj + 2
    k = kk + 2

    if i < n-1 and j < n-1 and k < n-1:
        pc = p[i,j,k]

        # Same pressure shell values as p_wall = nearest interior p, but without
        # writing the shell during intermediate RK projections.
        pjm = pc if j == 2   else p[i,j-1,k]
        pjp = pc if j == n-2 else p[i,j+1,k]
        pim = pc if i == 2   else p[i-1,j,k]
        pip = pc if i == n-2 else p[i+1,j,k]
        pkm = pc if k == 2   else p[i,j,k-1]
        pkp = pc if k == n-2 else p[i,j,k+1]

        # Average of the two projected face velocities surrounding each cell.
        u[i,j,k] = F025*(q1[i,j-1,k] + F2*q1[i,j,k] + q1[i,j+1,k]) \
                   - (pjp - pjm)*inv_2d
        v[i,j,k] = F025*(q2[i-1,j,k] + F2*q2[i,j,k] + q2[i+1,j,k]) \
                   - (pip - pim)*inv_2d
        w[i,j,k] = F025*(q3[i,j,k-1] + F2*q3[i,j,k] + q3[i,j,k+1]) \
                   - (pkp - pkm)*inv_2d


# ============================================================
# HOST-SIDE ALGORITHM
# ============================================================
def apply_velocity_bc(u, v, w, lid_vel, n, bfull=None):
    set_velocity_physical_boundaries_compact[physical_boundary_blocks(n), 256](u, v, w, lid_vel, n)
    copy_velocity_outer_ghosts_compact[outer_ghost_blocks(n), 256](u, v, w, n)


def apply_q_bc(q1, q2, q3, p, lid_vel, inv_d, n, bfull=None):
    set_q_face_boundaries_compact[q_face_boundary_blocks(n), 256](q1, q2, q3, p, lid_vel, inv_d, n)
    fill_q_edges_compact[q_edge_boundary_blocks(n), 256](q1, q2, q3, n)
    fill_q_corners_compact[1, 8](q1, q2, q3, n)


def solve_pressure_and_velocity(p, rhs,
                                q1, q2, q3,
                                u, v, w,
                                ter, n,
                                bint, bsor, bpress,
                                write_pressure_shell=False):
    build_pressure_rhs_direct[bint, TPB](rhs, q1, q2, q3, INV_2D32, n)

    for _ in range(ter):
        rb_sor[bsor, TPB](p, rhs, D2_32, OMEGA32, n, 0)
        rb_sor[bsor, TPB](p, rhs, D2_32, OMEGA32, n, 1)

    reconstruct_cell_velocity_direct[bint, TPB](u, v, w, q1, q2, q3, p, INV_2D32, n)
    apply_velocity_bc(u, v, w, VEL32, n)

    # q-BC at the next physical timestep needs a stored Neumann pressure shell.
    # Intermediate RK stages do not: both SOR and velocity reconstruction already
    # substitute the identical Neumann value directly.
    if write_pressure_shell:
        set_pressure_boundaries[bpress, 256](p, n)


def eval_rhs(out1, out2, out3,
             q1, q2, q3,
             u, v, w, nu_t,
             n, bactive, bint):
    # Same nu_t values, but launch only the active 1..n-1 cube.
    compute_nu_t[bactive, TPB](nu_t, u, v, w, INV_2D32, SMAG_COEFF32, n)
    solve_impulse[bint, TPB](
        out1, out2, out3,
        q1, q2, q3, u, v, w, nu_t,
        K1C32, K2C32, INV_D32, INV_4D32, n
    )


def one_step(state, bactive, bint, bsor, bpress):
    (u, v, w, q1, q2, q3, p, rhs, nu_t,
     kq1, kq2, kq3,
     a1, a2, a3,
     q1a, q2a, q3a) = state

    # Refresh BASE-state BCs once at the beginning, same ordering as before.
    apply_velocity_bc(u, v, w, VEL32, n)
    apply_q_bc(q1, q2, q3, p, VEL32, INV_D32, n)

    # Stage boundaries stay equal to base q boundaries; RK increments are interior-only.
    copy_q_shell_to_stage[q_shell_copy_blocks(n), 256](q1a, q2a, q3a, q1, q2, q3, n)

    # RK1
    eval_rhs(kq1, kq2, kq3, q1, q2, q3, u, v, w, nu_t, n, bactive, bint)
    rk_stage_init[bint, TPB](q1a, q2a, q3a, a1, a2, a3,
                             q1, q2, q3, kq1, kq2, kq3,
                             HALF_DT32, n)
    solve_pressure_and_velocity(p, rhs, q1a, q2a, q3a, u, v, w,
                                ter_stage, n,
                                bint, bsor, bpress, False)

    # RK2
    eval_rhs(kq1, kq2, kq3, q1a, q2a, q3a, u, v, w, nu_t, n, bactive, bint)
    rk_stage_accum[bint, TPB](q1a, q2a, q3a, a1, a2, a3,
                              q1, q2, q3, kq1, kq2, kq3,
                              HALF_DT32, F2, n)
    solve_pressure_and_velocity(p, rhs, q1a, q2a, q3a, u, v, w,
                                ter_stage, n,
                                bint, bsor, bpress, False)

    # RK3
    eval_rhs(kq1, kq2, kq3, q1a, q2a, q3a, u, v, w, nu_t, n, bactive, bint)
    rk_stage_accum[bint, TPB](q1a, q2a, q3a, a1, a2, a3,
                              q1, q2, q3, kq1, kq2, kq3,
                              DT32, F2, n)
    solve_pressure_and_velocity(p, rhs, q1a, q2a, q3a, u, v, w,
                                ter_stage, n,
                                bint, bsor, bpress, False)

    # RK4, overwrite the same slope triplet and complete the classical weighted sum.
    eval_rhs(kq1, kq2, kq3, q1a, q2a, q3a, u, v, w, nu_t, n, bactive, bint)
    rk_final_streamed[bint, TPB](q1, q2, q3, a1, a2, a3,
                                 kq1, kq2, kq3, DT32, n)

    # Final projection. Store pressure shell once for the next timestep's q BC.
    solve_pressure_and_velocity(p, rhs, q1, q2, q3, u, v, w,
                                ter_stage, n,
                                bint, bsor, bpress, True)


# ============================================================
# PLOTTING / DIAGNOSTICS
# ============================================================
def plot_midplane(u_h, v_h, w_h, step):
    k_mid = n // 2
    F = np.sqrt(u_h*u_h + v_h*v_h + w_h*w_h)
    F_mid = np.flipud(F[2:n-1, 2:n-1, k_mid])

    nc = n - 3
    x = np.linspace(0.0, 1.0, nc)
    y = np.linspace(0.0, 1.0, nc)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(x, y, F_mid, shading="nearest")
    fig.colorbar(im, ax=ax, label="Velocity magnitude")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Mid-z velocity magnitude, step {step}")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.2)
    plt.close(fig)


def main():
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available to Numba.")

    print("CUDA device:", cuda.get_current_device().name)
    print("3-D solver = literal 2-D numerical structure + z axis")
    print("n =", n, " active cells/axis =", n-3)
    print("d =", d, " dt =", delt)
    print("visc =", visc, " Re =", vel/visc)
    print("Cs =", Cs)
    print("Poisson = 3-D red/black SOR, iterations/projection =", ter_stage)
    print("perturbation amplitude =", PERTURB_AMPLITUDE)

    shape = (n+1, n+1, n+1)
    zeros = np.zeros(shape, dtype=DTYPE)

    # Primary fields.
    u = cuda.to_device(zeros)
    v = cuda.to_device(zeros)
    w = cuda.to_device(zeros)
    q1 = cuda.to_device(zeros)
    q2 = cuda.to_device(zeros)
    q3 = cuda.to_device(zeros)
    p = cuda.to_device(zeros)
    rhs = cuda.to_device(zeros)
    nu_t = cuda.to_device(zeros)

    # Streamed classical RK4 storage: one current slope triplet + weighted sum.
    # This preserves the four classical RK4 stage locations while avoiding nine
    # additional full-domain slope arrays.
    kq1 = cuda.to_device(zeros)
    kq2 = cuda.to_device(zeros)
    kq3 = cuda.to_device(zeros)
    a1 = cuda.to_device(zeros)
    a2 = cuda.to_device(zeros)
    a3 = cuda.to_device(zeros)

    q1a = cuda.to_device(zeros)
    q2a = cuda.to_device(zeros)
    q3a = cuda.to_device(zeros)

    del zeros

    bactive = blocks_active(n)
    bint = blocks_interior(n)
    bsor = blocks_sor(n)
    bpress = pressure_boundary_blocks(n)

    initialize_3d_perturbation[bint, TPB](
        u, v, w, q1, q2, q3,
        DTYPE(PERTURB_AMPLITUDE), DTYPE(d), n
    )
    apply_velocity_bc(u, v, w, VEL32, n)
    apply_q_bc(q1, q2, q3, p, VEL32, INV_D32, n)
    cuda.synchronize()

    state = (
        u, v, w, q1, q2, q3, p, rhs, nu_t,
        kq1, kq2, kq3,
        a1, a2, a3,
        q1a, q2a, q3a
    )

    # Mean strain-magnitude history: |S| = sqrt(2*Sij*Sij)
    mean_S_step_hist = []
    mean_S_hist = []

    t0 = time.perf_counter()

    for step in range(1, total_steps + 1):
        one_step(state, bactive, bint, bsor, bpress)

        if step == 1 or step % print_interval == 0:
            cuda.synchronize()
            uh = u.copy_to_host()
            vh = v.copy_to_host()
            wh = w.copy_to_host()
            nth = nu_t.copy_to_host()

            interior = np.s_[2:n-1, 2:n-1, 2:n-1]
            Fint = np.sqrt(uh[interior]**2 + vh[interior]**2 + wh[interior]**2)
            nt = nth[interior]
            w_int = wh[interior]
            u_int = uh[interior]
            u_zmean = np.mean(u_int, axis=2, keepdims=True)

            # nu_t = (Cs*d)^2 |S|  ->  |S| = nu_t / (Cs*d)^2
            mean_S = np.mean(nt) / ((Cs*d)*(Cs*d))
            mean_S_step_hist.append(step)
            mean_S_hist.append(mean_S)

            elapsed = time.perf_counter() - t0
            sps = step / elapsed if elapsed > 0.0 else 0.0

            print(
                f"step={step:7d}  time={step*delt:10.5f}  "
                f"max|V|={np.max(Fint):.6e}  "
                f"max_nu_t/nu={np.max(nt)/visc:.6e}  "
                f"mean_nu_t/nu={np.mean(nt)/visc:.6e}  "
                f"mean|S|={mean_S:.6e}  "
                f"w_rms={np.sqrt(np.mean(w_int*w_int)):.6e}  "
                f"u_span_rms={np.sqrt(np.mean((u_int-u_zmean)**2)):.6e}  "
                f"steps/s={sps:.3f}"
            )

        if ENABLE_PLOTTING and step % plot_interval == 0:
            if not (step == 1 or step % print_interval == 0):
                uh = u.copy_to_host()
                vh = v.copy_to_host()
                wh = w.copy_to_host()
            plot_midplane(uh, vh, wh, step)

        if (step == 1 or step % print_interval == 0) and len(mean_S_hist) > 0:
            figS, axS = plt.subplots(figsize=(7, 5))
            axS.plot(mean_S_step_hist, mean_S_hist, linewidth=1.5)
            axS.set_xlabel("Time step")
            axS.set_ylabel("Mean |S|")
            axS.set_title("Domain mean strain magnitude")
            axS.grid(True)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.2)
            plt.close(figS)

    cuda.synchronize()
    print("Simulation complete. Elapsed seconds =", time.perf_counter() - t0)


if __name__ == "__main__":
    main()