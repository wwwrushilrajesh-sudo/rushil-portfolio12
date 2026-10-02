# Velocity–Impulse Solver Version Archive

This folder preserves public development snapshots from my velocity–impulse incompressible-flow solver work.

The archive is intentionally split between **public historical versions** and **current research branches**. The current k–ω SST turbulence work is listed for transparency, but its source is being held until the associated results are published.

## Public versions

| Version | File | Main idea |
|---|---|---|
| V1 | `01_el_cpu_sor.py` | 2D E–Liu baseline using RK4 and sequential SOR projection |
| V2 | `02_geometric_cpu_sor.py` | 2D geometric impulse formulation with the stretching contribution, retained for controlled EL-vs-geometric comparison |
| V3 | `03_el_cuda_jacobi.py` | CUDA-accelerated E–Liu solver with residual-controlled Jacobi projection |
| V4 | `04_3d_geometric_semilagrangian_smagorinsky.py` | 3D geometric impulse transport with semi-Lagrangian advection, stretching, Smagorinsky SGS, and CUDA |
| V5 | `05_3d_cuda_les_fast_v3.py` | Performance-focused 3D CUDA LES revision with streamed RK4, packed red/black SOR, compact boundary kernels, and FP32-oriented optimization |

## Current research branches — source held until publication

These branches are part of the active research line and are **not public yet**:

- **V6 — E–Liu + k–ω SST + direct DCT projection**
- **V7 — Eulerian geometric + k–ω SST + direct DCT projection**
- **V8 — Geometric SST revision 2**
- **V9 — Semi-Lagrangian geometric + k–ω SST**

The current research branch adds k–ω SST turbulence modeling to the velocity–impulse framework and uses a direct cosine-transform Poisson solve for the scalar-potential projection. Source for these versions is planned for release after publication of the associated results.

## Why keep the versions?

The point of this archive is to show the solver's numerical-development path rather than presenting one monolithic final codebase:

```text
CPU baseline
    ↓
EL vs geometric formulation comparison
    ↓
GPU projection
    ↓
3D extension
    ↓
LES / semi-Lagrangian experiments
    ↓
k–ω SST research branch
```

Each file is a research/development snapshot. Parameters are kept close to the state in which the version was tested, so users should review Reynolds number, grid size, timestep, convergence settings, and GPU memory requirements before running a case.

## Dependencies

Depending on the version:

- Python
- NumPy
- Matplotlib
- Numba
- CUDA-capable NVIDIA GPU for CUDA versions

Some current private research branches also use CuPy / `cupyx.scipy.fft` for direct cosine-transform projection.

## Research status

This repository is a development archive, not a packaged production CFD library. Validation, grid studies, turbulence-model assessment, and publication work are ongoing.


## Verification and validation already established in the thesis

The solver-development work is supported by a research validation workflow:

- **Lid-driven cavity benchmark:** centerline velocity comparisons against established reference data for Re = 100, 500, and 1000.
- **Grid refinement:** 128², 256², and 512² cases used to assess error reduction and grid sensitivity.
- **Taylor–Green vortex:** periodic analytical benchmark used to study velocity-decay accuracy and long-time numerical error.
- **Projection convergence:** scalar-potential convergence is monitored and projection is performed at each RK4 stage.
- **Time-step control:** reported thesis simulations maintain Courant number below 0.5.
- **Computational performance:** Numba/CUDA acceleration and Poisson-solver implementation are documented and evaluated.

These are research verification/validation practices. They are not yet a fully automated regression-test/CI suite.
