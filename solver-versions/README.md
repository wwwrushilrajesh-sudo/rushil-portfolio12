# Velocity–Impulse Solver Releases

## Original baselines

The first two development snapshots remain available:

1. [E–Liu CPU and SOR](./01_el_cpu_sor.py) — 2D baseline with RK4 and iterative projection.
2. [Geometric CPU and SOR](./02_geometric_cpu_sor.py) — 2D formulation comparison with stretching.

## Current release

[Version 1.0 — LES Modeled Geometric Impulse](./v1.0-les-modeled-geometric-impulse/)

The current simulation backend is written entirely in CUDA C++, with Python used for the frontend and post-processing. This release contains the complete frontend, bridge, backend, reporting and build pipeline; see its README for scope and numerical implementation.

Currently working on reducing numerical diffusion artifacts from discretization and optimizing CUDA kernels. Future changes will be documented in subsequent releases.

Older V3–V9 listings have been removed from the website and this release index. Historical repository files and Git history are retained.
