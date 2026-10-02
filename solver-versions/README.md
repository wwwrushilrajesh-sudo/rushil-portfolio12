# Velocity–Impulse Solver Version Archive

This folder contains selected public development snapshots from my velocity–impulse incompressible-flow solver work.

## Public source versions

| Version | File | Main idea |
|---|---|---|
| V1 | `01_el_cpu_sor.py` | 2D E–Liu baseline using RK4 and sequential SOR projection |
| V2 | `02_geometric_cpu_sor.py` | 2D geometric impulse formulation with the stretching contribution |
| V3 | `03_el_cuda_jacobi.py` | CUDA-accelerated E–Liu solver with residual-controlled Jacobi projection |

## Private research branches

All turbulence-modeling source is being kept private for now:

- **V4 — 3D geometric semi-Lagrangian + SGS**
- **V5 — 3D CUDA LES Fast V3**
- **V6 — E–Liu + k–ω SST + direct DCT projection**
- **V7 — Eulerian geometric + k–ω SST + direct DCT projection**
- **V8 — Geometric SST revision 2**
- **V9 — Semi-Lagrangian geometric + k–ω SST**

These branches are listed to show the development path, but their source is intentionally not included in the public tree. The intended release point is after publication of the associated research results.

## Verification and validation already established in the thesis

The solver-development work is supported by a research validation workflow:

- **Lid-driven cavity benchmark:** centerline velocity comparisons against established reference data for Re = 100, 500, and 1000.
- **Grid refinement:** 128², 256², and 512² cases used to assess error reduction and grid sensitivity.
- **Taylor–Green vortex:** periodic analytical benchmark used to study velocity-decay accuracy and long-time numerical error.
- **Projection convergence:** scalar-potential convergence is monitored and projection is performed at each RK4 stage.
- **Time-step control:** reported thesis simulations maintain Courant number below 0.5.
- **Computational performance:** Numba/CUDA acceleration and Poisson-solver implementation are documented and evaluated.

These are research verification/validation practices; they are not yet a fully automated regression-test / CI suite.

## Development path

```text
CPU baseline
    ↓
EL vs geometric comparison
    ↓
GPU projection
    ↓
3D / turbulence research branches
    ↓
current k–ω SST research
```

The public files are research/development snapshots rather than a packaged production CFD library.
