# 3-D extension of lid_cavity_no_massive_phi_history(2).py numerical structure.
# PERFORMANCE PATCH (math-preserving pressure solve): packed red/black pressure
# storage for coalesced SOR memory access, 256-thread SOR blocks, merged velocity
# wall/ghost BC launch, and removal of one redundant velocity-BC pass per step.
# Retains CUDA, float32, streamed RK4 diffusion/SGS, packed red/black SOR,
# and velocity-based tensor SGS. q advection is pure semi-Lagrangian;
# geometric stretching is retained after the semi-Lagrangian advection step.
# Face operations are fused but match explicit reference interpolation/recovery.
# 3-D wall edges/corners use adjacent-face/edge completion; the 2-D file does
# not uniquely specify these. SOR ordering differs from its parallel in-place
# CPU update. This is not a claim of bit-for-bit trajectory equivalence.
# No extra q smoothing: the reference's optional ni==1 path is never enabled.

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda


# ============================================================
# PARAMETERS
# ============================================================
n = 128
vel = 1.0
d = 1.0 / (n - 3)
delt = 0.001
visc = 0.0001          # Re = 100000 for U=L=1
Cs = 0.3

k1c = 1.0 / (2.0 * d)
k2c = visc / (d * d)

omega = 1.90
ter_stage=45

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
TPB = (32, 4, 1)             # 128 threads/block for heavy RHS/LES kernels
TPB_SOR = (32, 8, 1)         # 256 threads/block; packed pressure k-index is contiguous
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
    """Packed red/black pressure arrays: hk is contiguous in memory."""
    m = n - 3
    mh = (m + 1) // 2
    return (
        (mh + TPB_SOR[0] - 1) // TPB_SOR[0],
        (m  + TPB_SOR[1] - 1) // TPB_SOR[1],
        (m  + TPB_SOR[2] - 1) // TPB_SOR[2],
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


def velocity_all_boundary_blocks(n):
    count = 6*(n-1)*(n-1) + 6*(n+1)*(n+1)
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


@cuda.jit
def set_velocity_all_boundaries_compact(u, v, w, lid_vel, n):
    """Physical walls + outer ghosts in one independent launch."""
    idx = cuda.grid(1)

    side_p = n - 1
    per_p = side_p*side_p
    total_p = 6*per_p

    side_g = n + 1
    per_g = side_g*side_g
    total = total_p + 6*per_g
    if idx >= total:
        return

    if idx < total_p:
        face = idx // per_p
        rem = idx - face*per_p
        a = rem // side_p + 1
        b = rem - (rem // side_p)*side_p + 1

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

        u[i,j,k] = lid_vel if i == 1 else F0
        v[i,j,k] = F0
        w[i,j,k] = F0
        return

    gidx = idx - total_p
    face = gidx // per_g
    rem = gidx - face*per_g
    a = rem // side_g
    b = rem - (rem // side_g)*side_g

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

    # Equivalent to copying from the clamped physical-wall cell after wall BCs.
    ii = 1 if i == 0 else (n-1 if i == n else i)
    u[i,j,k] = lid_vel if ii == 1 else F0
    v[i,j,k] = F0
    w[i,j,k] = F0


# ============================================================
# q BOUNDARY CONDITIONS
# Direct 3-D extension of the 2-D impulse BC pattern.
# Face interiors use the same reflection + tangential phi-gradient rule.
# 3-D edges/corners are filled from neighboring face values afterward.
# ============================================================


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
    # Extend the reference 2-D wall rule component-by-component.
    # Tangential endpoints use a one-sided gradient and no reflection.
    idx = cuda.grid(1)
    m = n - 3
    per_face = m*m
    if idx >= 6*per_face:
        return
    face = idx // per_face
    rem = idx - face*per_face
    a = rem // m + 2
    b = rem % m + 2
    if face == 0:
        i=1; j=a; k=b
        if j == 2:
            q1[i,j,k] = lid_vel + (p[i,j+1,k] - p[i,j,k])*inv_d
        elif j == n-2:
            q1[i,j,k] = lid_vel + (p[i,j,k] - p[i,j-1,k])*inv_d
        else:
            q1[i,j,k] = F0 if i == 2 or i == n-2 or k == 2 or k == n-2 else -q1[2,j,k] + F2*lid_vel + (p[i,j+1,k] - p[i,j-1,k])*inv_d
        q2[i,j,k] = F0 if j == 2 or j == n-2 or k == 2 or k == n-2 else -q2[2,j,k]
        if k == 2:
            q3[i,j,k] = F0 + (p[i,j,k+1] - p[i,j,k])*inv_d
        elif k == n-2:
            q3[i,j,k] = F0 + (p[i,j,k] - p[i,j,k-1])*inv_d
        else:
            q3[i,j,k] = F0 if i == 2 or i == n-2 or j == 2 or j == n-2 else -q3[2,j,k] + F2*F0 + (p[i,j,k+1] - p[i,j,k-1])*inv_d
    elif face == 1:
        i=n-1; j=a; k=b
        if j == 2:
            q1[i,j,k] = F0 + (p[i,j+1,k] - p[i,j,k])*inv_d
        elif j == n-2:
            q1[i,j,k] = F0 + (p[i,j,k] - p[i,j-1,k])*inv_d
        else:
            q1[i,j,k] = F0 if i == 2 or i == n-2 or k == 2 or k == n-2 else -q1[n-2,j,k] + F2*F0 + (p[i,j+1,k] - p[i,j-1,k])*inv_d
        q2[i,j,k] = F0 if j == 2 or j == n-2 or k == 2 or k == n-2 else -q2[n-2,j,k]
        if k == 2:
            q3[i,j,k] = F0 + (p[i,j,k+1] - p[i,j,k])*inv_d
        elif k == n-2:
            q3[i,j,k] = F0 + (p[i,j,k] - p[i,j,k-1])*inv_d
        else:
            q3[i,j,k] = F0 if i == 2 or i == n-2 or j == 2 or j == n-2 else -q3[n-2,j,k] + F2*F0 + (p[i,j,k+1] - p[i,j,k-1])*inv_d
    elif face == 2:
        i=a; j=1; k=b
        q1[i,j,k] = F0 if i == 2 or i == n-2 or k == 2 or k == n-2 else -q1[i,2,k]
        if i == 2:
            q2[i,j,k] = F0 + (p[i+1,j,k] - p[i,j,k])*inv_d
        elif i == n-2:
            q2[i,j,k] = F0 + (p[i,j,k] - p[i-1,j,k])*inv_d
        else:
            q2[i,j,k] = F0 if j == 2 or j == n-2 or k == 2 or k == n-2 else -q2[i,2,k] + F2*F0 + (p[i+1,j,k] - p[i-1,j,k])*inv_d
        if k == 2:
            q3[i,j,k] = F0 + (p[i,j,k+1] - p[i,j,k])*inv_d
        elif k == n-2:
            q3[i,j,k] = F0 + (p[i,j,k] - p[i,j,k-1])*inv_d
        else:
            q3[i,j,k] = F0 if i == 2 or i == n-2 or j == 2 or j == n-2 else -q3[i,2,k] + F2*F0 + (p[i,j,k+1] - p[i,j,k-1])*inv_d
    elif face == 3:
        i=a; j=n-1; k=b
        q1[i,j,k] = F0 if i == 2 or i == n-2 or k == 2 or k == n-2 else -q1[i,n-2,k]
        if i == 2:
            q2[i,j,k] = F0 + (p[i+1,j,k] - p[i,j,k])*inv_d
        elif i == n-2:
            q2[i,j,k] = F0 + (p[i,j,k] - p[i-1,j,k])*inv_d
        else:
            q2[i,j,k] = F0 if j == 2 or j == n-2 or k == 2 or k == n-2 else -q2[i,n-2,k] + F2*F0 + (p[i+1,j,k] - p[i-1,j,k])*inv_d
        if k == 2:
            q3[i,j,k] = F0 + (p[i,j,k+1] - p[i,j,k])*inv_d
        elif k == n-2:
            q3[i,j,k] = F0 + (p[i,j,k] - p[i,j,k-1])*inv_d
        else:
            q3[i,j,k] = F0 if i == 2 or i == n-2 or j == 2 or j == n-2 else -q3[i,n-2,k] + F2*F0 + (p[i,j,k+1] - p[i,j,k-1])*inv_d
    elif face == 4:
        i=a; j=b; k=1
        if j == 2:
            q1[i,j,k] = F0 + (p[i,j+1,k] - p[i,j,k])*inv_d
        elif j == n-2:
            q1[i,j,k] = F0 + (p[i,j,k] - p[i,j-1,k])*inv_d
        else:
            q1[i,j,k] = F0 if i == 2 or i == n-2 or k == 2 or k == n-2 else -q1[i,j,2] + F2*F0 + (p[i,j+1,k] - p[i,j-1,k])*inv_d
        if i == 2:
            q2[i,j,k] = F0 + (p[i+1,j,k] - p[i,j,k])*inv_d
        elif i == n-2:
            q2[i,j,k] = F0 + (p[i,j,k] - p[i-1,j,k])*inv_d
        else:
            q2[i,j,k] = F0 if j == 2 or j == n-2 or k == 2 or k == n-2 else -q2[i,j,2] + F2*F0 + (p[i+1,j,k] - p[i-1,j,k])*inv_d
        q3[i,j,k] = F0 if i == 2 or i == n-2 or j == 2 or j == n-2 else -q3[i,j,2]
    elif face == 5:
        i=a; j=b; k=n-1
        if j == 2:
            q1[i,j,k] = F0 + (p[i,j+1,k] - p[i,j,k])*inv_d
        elif j == n-2:
            q1[i,j,k] = F0 + (p[i,j,k] - p[i,j-1,k])*inv_d
        else:
            q1[i,j,k] = F0 if i == 2 or i == n-2 or k == 2 or k == n-2 else -q1[i,j,n-2] + F2*F0 + (p[i,j+1,k] - p[i,j-1,k])*inv_d
        if i == 2:
            q2[i,j,k] = F0 + (p[i+1,j,k] - p[i,j,k])*inv_d
        elif i == n-2:
            q2[i,j,k] = F0 + (p[i,j,k] - p[i-1,j,k])*inv_d
        else:
            q2[i,j,k] = F0 if j == 2 or j == n-2 or k == 2 or k == n-2 else -q2[i,j,n-2] + F2*F0 + (p[i+1,j,k] - p[i-1,j,k])*inv_d
        q3[i,j,k] = F0 if i == 2 or i == n-2 or j == 2 or j == n-2 else -q3[i,j,n-2]


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
@cuda.jit(device=True, inline=True)
def trilerp_q(field, yi, xj, zk, n):
    # Clamp departure point to physical-index range.
    if xj < F1:
        xj = F1
    elif xj > n - F1:
        xj = n - F1

    if yi < F1:
        yi = F1
    elif yi > n - F1:
        yi = n - F1

    if zk < F1:
        zk = F1
    elif zk > n - F1:
        zk = n - F1

    j0 = int(math.floor(xj))
    i0 = int(math.floor(yi))
    k0 = int(math.floor(zk))

    j1 = j0 + 1
    i1 = i0 + 1
    k1 = k0 + 1

    if j1 > n - 1:
        j1 = n - 1
    if i1 > n - 1:
        i1 = n - 1
    if k1 > n - 1:
        k1 = n - 1

    fx = xj - j0
    fy = yi - i0
    fz = zk - k0

    c000 = field[i0,j0,k0]
    c001 = field[i0,j0,k1]
    c010 = field[i0,j1,k0]
    c011 = field[i0,j1,k1]
    c100 = field[i1,j0,k0]
    c101 = field[i1,j0,k1]
    c110 = field[i1,j1,k0]
    c111 = field[i1,j1,k1]

    c00 = c000*(F1-fx) + c010*fx
    c01 = c001*(F1-fx) + c011*fx
    c10 = c100*(F1-fx) + c110*fx
    c11 = c101*(F1-fx) + c111*fx

    c0 = c00*(F1-fy) + c10*fy
    c1 = c01*(F1-fy) + c11*fy

    return c0*(F1-fz) + c1*fz


@cuda.jit(fastmath=True)
def semi_lagrangian_advect_q(q1_out, q2_out, q3_out,
                             q1, q2, q3,
                             u, v, w,
                             dt, inv_d, n):
    """
    Pure semi-Lagrangian advection of q.

    x-direction is j, y-direction is i, z-direction is k.
    NO geometric stretching term is applied.
    """
    kk, jj, ii = cuda.grid(3)
    i = ii + 2
    j = jj + 2
    k = kk + 2

    if i >= n-1 or j >= n-1 or k >= n-1:
        return

    # Backtrace one full dt in grid-index coordinates.
    jd = j - dt*u[i,j,k]*inv_d
    idp = i - dt*v[i,j,k]*inv_d
    kd = k - dt*w[i,j,k]*inv_d

    q1_out[i,j,k] = trilerp_q(q1, idp, jd, kd, n)
    q2_out[i,j,k] = trilerp_q(q2, idp, jd, kd, n)
    q3_out[i,j,k] = trilerp_q(q3, idp, jd, kd, n)


@cuda.jit
def copy_q_interior(dst1, dst2, dst3, src1, src2, src3, n):
    kk, jj, ii = cuda.grid(3)
    i = ii + 2
    j = jj + 2
    k = kk + 2
    if i < n-1 and j < n-1 and k < n-1:
        dst1[i,j,k] = src1[i,j,k]
        dst2[i,j,k] = src2[i,j,k]
        dst3[i,j,k] = src3[i,j,k]


@cuda.jit(fastmath=True)
def solve_impulse(out_q1, out_q2, out_q3,
                  q1, q2, q3, u, v, w, nu_t,
                  k1, k2, inv_d, inv_4d, n):
    """
    RHS after semi-Lagrangian advection:

        dq/dt = -(grad(u))^T q + nu*lap(q) + SGS

    Advection is handled separately by semi_lagrangian_advect_q.
    Geometric stretching is retained.
    """
    kk, jj, ii = cuda.grid(3)
    i = ii + 2
    j = jj + 2
    k = kk + 2
    if i >= n-1 or j >= n-1 or k >= n-1:
        return

    # Velocity derivatives WITHOUT /2d.
    # k1 = 1/(2d) is applied to the stretching term below.
    du_dx = u[i,j+1,k] - u[i,j-1,k]
    du_dy = u[i+1,j,k] - u[i-1,j,k]
    du_dz = u[i,j,k+1] - u[i,j,k-1]

    dv_dx = v[i,j+1,k] - v[i,j-1,k]
    dv_dy = v[i+1,j,k] - v[i-1,j,k]
    dv_dz = v[i,j,k+1] - v[i,j,k-1]

    dw_dx = w[i,j+1,k] - w[i,j-1,k]
    dw_dy = w[i+1,j,k] - w[i-1,j,k]
    dw_dz = w[i,j,k+1] - w[i,j,k-1]

    # Molecular diffusion on impulse q.
    lap_q1 = (
        q1[i-1,j,k] + q1[i+1,j,k]
        + q1[i,j-1,k] + q1[i,j+1,k]
        + q1[i,j,k-1] + q1[i,j,k+1]
        - F6*q1[i,j,k]
    )

    lap_q2 = (
        q2[i-1,j,k] + q2[i+1,j,k]
        + q2[i,j-1,k] + q2[i,j+1,k]
        + q2[i,j,k-1] + q2[i,j,k+1]
        - F6*q2[i,j,k]
    )

    lap_q3 = (
        q3[i-1,j,k] + q3[i+1,j,k]
        + q3[i,j-1,k] + q3[i,j+1,k]
        + q3[i,j,k-1] + q3[i,j,k+1]
        - F6*q3[i,j,k]
    )

    # Existing velocity-based symmetric SGS force.
    nut_xp = F05*(nu_t[i,j,k] + nu_t[i,j+1,k])
    nut_xm = F05*(nu_t[i,j,k] + nu_t[i,j-1,k])
    nut_yp = F05*(nu_t[i,j,k] + nu_t[i+1,j,k])
    nut_ym = F05*(nu_t[i,j,k] + nu_t[i-1,j,k])
    nut_zp = F05*(nu_t[i,j,k] + nu_t[i,j,k+1])
    nut_zm = F05*(nu_t[i,j,k] + nu_t[i,j,k-1])

    # x SGS force
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

    # y SGS force
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

    # z SGS force
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



    # Geometric transpose stretching:
    # [(grad u)^T q]_x = q1 du/dx + q2 dv/dx + q3 dw/dx
    # [(grad u)^T q]_y = q1 du/dy + q2 dv/dy + q3 dw/dy
    # [(grad u)^T q]_z = q1 du/dz + q2 dv/dz + q3 dw/dz
    stretch_q1 = (
        q1[i,j,k]*du_dx
        + q2[i,j,k]*dv_dx
        + q3[i,j,k]*dw_dx
    )
    stretch_q2 = (
        q1[i,j,k]*du_dy
        + q2[i,j,k]*dv_dy
        + q3[i,j,k]*dw_dy
    )
    stretch_q3 = (
        q1[i,j,k]*du_dz
        + q2[i,j,k]*dv_dz
        + q3[i,j,k]*dw_dz
    )

    out_q1[i,j,k] = -k1*stretch_q1 + k2*lap_q1 + les_q1
    out_q2[i,j,k] = -k1*stretch_q2 + k2*lap_q2 + les_q2
    out_q3[i,j,k] = -k1*stretch_q3 + k2*lap_q3 + les_q3


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
def build_pressure_rhs_packed(rhs0, rhs1, q1, q2, q3, inv_2d, n):
    # Same fused face interpolation + divergence, stored directly in packed
    # checkerboard arrays so every SOR warp reads contiguous rhs values.
    kk, jj, ii = cuda.grid(3)
    i, j, k = ii + 2, jj + 2, kk + 2
    if i < n-1 and j < n-1 and k < n-1:
        xp = F0 if j == n-2 else q1[i,j,k] + q1[i,j+1,k]
        xm = F0 if j == 2 else q1[i,j-1,k] + q1[i,j,k]
        yp = F0 if i == n-2 else q2[i,j,k] + q2[i+1,j,k]
        ym = F0 if i == 2 else q2[i-1,j,k] + q2[i,j,k]
        zp = F0 if k == n-2 else q3[i,j,k] + q3[i,j,k+1]
        zm = F0 if k == 2 else q3[i,j,k-1] + q3[i,j,k]
        r = (xp - xm + yp - ym + zp - zm)*inv_2d
        hk = (k - 2) >> 1
        if ((i + j + k) & 1) == 0:
            rhs0[i,j,hk] = r
        else:
            rhs1[i,j,hk] = r


# ============================================================
# 3-D Poisson SOR -- same 7-point extension of the 2-D equation
# ============================================================
@cuda.jit(fastmath=True)
def rb_sor_packed(p_this, p_other, rhs_this, d2, omega, n, color):
    """
    Algebraically identical packed red/black SOR.  For a fixed (i,j), hk is
    contiguous, so center, +/-i, +/-j, +/-k and rhs loads are coalesced.
    """
    hk, jj, ii = cuda.grid(3)
    i = ii + 2
    j = jj + 2
    if i >= n-1 or j >= n-1:
        return

    kp = color ^ ((i + j) & 1)
    k = 2 + (hk << 1) + kp
    if k >= n-1:
        return

    pc = p_this[i,j,hk]

    pim = pc if i == 2   else p_other[i-1,j,hk]
    pip = pc if i == n-2 else p_other[i+1,j,hk]
    pjm = pc if j == 2   else p_other[i,j-1,hk]
    pjp = pc if j == n-2 else p_other[i,j+1,hk]

    # Opposite-color packed index for physical k-1 / k+1.
    pkm = pc if k == 2 else p_other[i,j,hk - (1 - kp)]
    pkp = pc if k == n-2 else p_other[i,j,hk + kp]

    p_new = (pip + pim + pjp + pjm + pkp + pkm - d2*rhs_this[i,j,hk]) / F6
    p_this[i,j,hk] = (F1 - omega)*pc + omega*p_new


@cuda.jit(fastmath=True)
def unpack_pressure(p, p0, p1, n):
    kk, jj, ii = cuda.grid(3)
    i, j, k = ii + 2, jj + 2, kk + 2
    if i < n-1 and j < n-1 and k < n-1:
        hk = (k - 2) >> 1
        if ((i + j + k) & 1) == 0:
            p[i,j,k] = p0[i,j,hk]
        else:
            p[i,j,k] = p1[i,j,hk]


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
# REFERENCE FACE -> CELL VELOCITY RECOVERY (fused, no face-array storage)
# This retains the 2-D reference's two averages; no additional filter is used.
# Boundary normal faces are prescribed zero, including during RK stages.
# ============================================================
@cuda.jit(fastmath=True)
def reconstruct_cell_velocity_direct(u, v, w, q1, q2, q3, p, inv_2d, n):
    kk, jj, ii = cuda.grid(3)
    i, j, k = ii + 2, jj + 2, kk + 2
    if i < n-1 and j < n-1 and k < n-1:
        inv_d = F2*inv_2d
        pc = p[i,j,k]
        uf_p = F0 if j == n-2 else F05*(q1[i,j,k]+q1[i,j+1,k]) - (p[i,j+1,k]-pc)*inv_d
        uf_m = F0 if j == 2 else F05*(q1[i,j-1,k]+q1[i,j,k]) - (pc-p[i,j-1,k])*inv_d
        vf_p = F0 if i == n-2 else F05*(q2[i,j,k]+q2[i+1,j,k]) - (p[i+1,j,k]-pc)*inv_d
        vf_m = F0 if i == 2 else F05*(q2[i-1,j,k]+q2[i,j,k]) - (pc-p[i-1,j,k])*inv_d
        wf_p = F0 if k == n-2 else F05*(q3[i,j,k]+q3[i,j,k+1]) - (p[i,j,k+1]-pc)*inv_d
        wf_m = F0 if k == 2 else F05*(q3[i,j,k-1]+q3[i,j,k]) - (pc-p[i,j,k-1])*inv_d
        u[i,j,k] = F05*(uf_p + uf_m)
        v[i,j,k] = F05*(vf_p + vf_m)
        w[i,j,k] = F05*(wf_p + wf_m)


# ============================================================
# HOST-SIDE ALGORITHM
# ============================================================
def apply_velocity_bc(u, v, w, lid_vel, n, bfull=None):
    set_velocity_all_boundaries_compact[velocity_all_boundary_blocks(n), 256](u, v, w, lid_vel, n)


def apply_q_bc(q1, q2, q3, p, lid_vel, inv_d, n, bfull=None):
    set_q_face_boundaries_compact[q_face_boundary_blocks(n), 256](q1, q2, q3, p, lid_vel, inv_d, n)
    fill_q_edges_compact[q_edge_boundary_blocks(n), 256](q1, q2, q3, n)
    fill_q_corners_compact[1, 8](q1, q2, q3, n)


def solve_pressure_and_velocity(p, p0, p1, rhs0, rhs1,
                                q1, q2, q3,
                                u, v, w,
                                ter, n,
                                bint, bsor, bpress,
                                write_pressure_shell=False):
    build_pressure_rhs_packed[bint, TPB](rhs0, rhs1, q1, q2, q3, INV_2D32, n)

    for _ in range(ter):
        rb_sor_packed[bsor, TPB_SOR](p0, p1, rhs0, D2_32, OMEGA32, n, 0)
        rb_sor_packed[bsor, TPB_SOR](p1, p0, rhs1, D2_32, OMEGA32, n, 1)

    # Only one full-grid pressure write per projection; SOR itself stays packed.
    unpack_pressure[bint, TPB](p, p0, p1, n)
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
    # Advection is already completed by semi_lagrangian_advect_q.
    # This RHS is geometric stretching + diffusion + velocity-based SGS.
    compute_nu_t[bactive, TPB](nu_t, u, v, w, INV_2D32, SMAG_COEFF32, n)
    solve_impulse[bint, TPB](
        out1, out2, out3,
        q1, q2, q3, u, v, w, nu_t,
        K1C32, K2C32, INV_D32, INV_4D32, n
    )


def one_step(state, bactive, bint, bsor, bpress):
    (u, v, w, q1, q2, q3, p, p0, p1, rhs0, rhs1, nu_t,
     kq1, kq2, kq3,
     a1, a2, a3,
     q1a, q2a, q3a) = state

    # Previous final projection already leaves u/v/w boundary-clean; main()
    # does the same before step 1, so only q BCs need refreshing here.
    apply_q_bc(q1, q2, q3, p, VEL32, INV_D32, n)

    # --------------------------------------------------------
    # PURE SEMI-LAGRANGIAN ADVECTION OF q
    # --------------------------------------------------------
    # Keep the current q wall shell in the stage buffers, then advect only
    # interior values backward along the recovered velocity field.
    copy_q_shell_to_stage[q_shell_copy_blocks(n), 256](q1a, q2a, q3a, q1, q2, q3, n)

    semi_lagrangian_advect_q[bint, TPB](
        q1a, q2a, q3a,
        q1, q2, q3,
        u, v, w,
        DT32, INV_D32, n
    )

    # Make the advected field the new base state.
    copy_q_interior[bint, TPB](q1, q2, q3, q1a, q2a, q3a, n)

    # Refresh q boundaries and project once so u,v,w are consistent with
    # the advected q before diffusion/SGS RK4 begins.
    apply_q_bc(q1, q2, q3, p, VEL32, INV_D32, n)
    solve_pressure_and_velocity(
        p, p0, p1, rhs0, rhs1,
        q1, q2, q3, u, v, w,
        ter_stage, n,
        bint, bsor, bpress, False
    )

    # Re-copy the now-advected q shell for RK stage storage.
    copy_q_shell_to_stage[q_shell_copy_blocks(n), 256](q1a, q2a, q3a, q1, q2, q3, n)

    # RK1: diffusion + SGS only
    eval_rhs(kq1, kq2, kq3, q1, q2, q3, u, v, w, nu_t, n, bactive, bint)
    rk_stage_init[bint, TPB](q1a, q2a, q3a, a1, a2, a3,
                             q1, q2, q3, kq1, kq2, kq3,
                             HALF_DT32, n)
    solve_pressure_and_velocity(p, p0, p1, rhs0, rhs1, q1a, q2a, q3a, u, v, w,
                                ter_stage, n,
                                bint, bsor, bpress, False)

    # RK2
    eval_rhs(kq1, kq2, kq3, q1a, q2a, q3a, u, v, w, nu_t, n, bactive, bint)
    rk_stage_accum[bint, TPB](q1a, q2a, q3a, a1, a2, a3,
                              q1, q2, q3, kq1, kq2, kq3,
                              HALF_DT32, F2, n)
    solve_pressure_and_velocity(p, p0, p1, rhs0, rhs1, q1a, q2a, q3a, u, v, w,
                                ter_stage, n,
                                bint, bsor, bpress, False)

    # RK3
    eval_rhs(kq1, kq2, kq3, q1a, q2a, q3a, u, v, w, nu_t, n, bactive, bint)
    rk_stage_accum[bint, TPB](q1a, q2a, q3a, a1, a2, a3,
                              q1, q2, q3, kq1, kq2, kq3,
                              DT32, F2, n)
    solve_pressure_and_velocity(p, p0, p1, rhs0, rhs1, q1a, q2a, q3a, u, v, w,
                                ter_stage, n,
                                bint, bsor, bpress, False)

    # RK4, overwrite the same slope triplet and complete the classical weighted sum.
    eval_rhs(kq1, kq2, kq3, q1a, q2a, q3a, u, v, w, nu_t, n, bactive, bint)
    rk_final_streamed[bint, TPB](q1, q2, q3, a1, a2, a3,
                                 kq1, kq2, kq3, DT32, n)

    # Final projection. Store pressure shell once for the next timestep's q BC.
    solve_pressure_and_velocity(p, p0, p1, rhs0, rhs1, q1, q2, q3, u, v, w,
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
    im = ax.pcolormesh(x, y, F_mid, shading="nearest", cmap="jet")
    fig.colorbar(im, ax=ax, label="Velocity magnitude")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Mid-z velocity magnitude, step {step}")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.2)
    plt.close(fig)


def plot_midplane_vorticity(u_h, v_h, w_h, step):
    """Plot |omega| on the mid-z plane using the jet colormap."""
    # Active-cell interior. Array directions are:
    # axis 0 -> y (i), axis 1 -> x (j), axis 2 -> z (k)
    us = u_h[2:n-1, 2:n-1, 2:n-1]
    vs = v_h[2:n-1, 2:n-1, 2:n-1]
    ws = w_h[2:n-1, 2:n-1, 2:n-1]

    # Velocity gradients with physical spacing d.
    du_dy, du_dx, du_dz = np.gradient(us, d, d, d, edge_order=2)
    dv_dy, dv_dx, dv_dz = np.gradient(vs, d, d, d, edge_order=2)
    dw_dy, dw_dx, dw_dz = np.gradient(ws, d, d, d, edge_order=2)

    # omega = curl(u)
    omega_x = dw_dy - dv_dz
    omega_y = du_dz - dw_dx
    omega_z = dv_dx - du_dy
    omega_mag = np.sqrt(omega_x*omega_x + omega_y*omega_y + omega_z*omega_z)

    k_mid_local = omega_mag.shape[2] // 2
    vort_mid = np.flipud(omega_mag[:, :, k_mid_local])

    nc = n - 3
    x = np.linspace(0.0, 1.0, nc)
    y = np.linspace(0.0, 1.0, nc)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(x, y, vort_mid, shading="nearest", cmap="jet")
    fig.colorbar(im, ax=ax, label=r"Vorticity magnitude $|\omega|$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Mid-z vorticity magnitude, step {step}")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.2)
    plt.close(fig)


def main():
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available to Numba.")

    print("CUDA device:", cuda.get_current_device().name)
    print("3-D reference face projection: zero wall flux, face-to-cell recovery")
    print("n =", n, " active cells/axis =", n-3)
    print("d =", d, " dt =", delt)
    print("visc =", visc, " Re =", vel/visc)
    print("Cs =", Cs)
    print("Poisson = packed 3-D red/black SOR, iterations/projection =", ter_stage)
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

    # Authoritative pressure/RHS storage for SOR is checkerboard-packed along k.
    # p remains as an expanded mirror used only by velocity recovery and q BCs.
    m = n - 3
    mh = (m + 1) // 2
    packed_shape = (n+1, n+1, mh)
    packed_zeros = np.zeros(packed_shape, dtype=DTYPE)
    p0 = cuda.to_device(packed_zeros)
    p1 = cuda.to_device(packed_zeros)
    rhs0 = cuda.to_device(packed_zeros)
    rhs1 = cuda.to_device(packed_zeros)
    del packed_zeros

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
        u, v, w, q1, q2, q3, p, p0, p1, rhs0, rhs1, nu_t,
        kq1, kq2, kq3,
        a1, a2, a3,
        q1a, q2a, q3a
    )

    # Mean strain-magnitude history: |S| = sqrt(2*Sij*Sij)
    mean_S_step_hist = []
    mean_S_hist = []

    t0 = time.perf_counter()
    timed_steps = 0

    for step in range(1, total_steps + 1):
        one_step(state, bactive, bint, bsor, bpress)

        # First step includes lazy Numba compilation; don't report that as solver throughput.
        if step == 1:
            cuda.synchronize()
            t0 = time.perf_counter()
            timed_steps = 0
        else:
            timed_steps += 1

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
            sps = timed_steps / elapsed if elapsed > 0.0 and timed_steps > 0 else 0.0

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
            plot_midplane_vorticity(uh, vh, wh, step)

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