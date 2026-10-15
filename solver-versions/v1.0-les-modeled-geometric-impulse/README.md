# Version 1.0 — LES Modeled Geometric Impulse

Initial release in the versioned LES-modeled geometric-impulse solver series.

## Source

[Download or inspect the CUDA C++ backend](./backend.cu).

The entire simulation backend is written in CUDA C++: GPU kernels and the host routines that coordinate geometry, time stepping, transport, stretching, molecular diffusion, WALE subgrid stresses, boundary conditions, and projection. Python provides the frontend, configuration, and post-processing through the backend's exported C interface.

This release publishes the current `backend.cu` source. The separate Python frontend and build scripts are not bundled in this source snapshot. The file embeds its interface declarations; CUDA Toolkit libraries include cuFFT and cuSOLVER.

## Numerical implementation

- Three-dimensional staggered MAC grid and geometric impulse evolution.
- Semi-Lagrangian transport with RK4 characteristic tracing and bounded MacCormack correction.
- Local implicit stretching and backward-Euler molecular diffusion with red-black iterations.
- WALE large-eddy subgrid modeling with explicit stress forcing.
- Cylinder boundary treatment and Poisson-based incompressibility projection.
- Trial-step stability checks, field queries, and Q-criterion visualization data export.

## Current work

Currently working on reducing numerical diffusion artifacts from discretization and optimizing CUDA kernels. The focus is vortex and wake preservation, GPU memory traffic, kernel fusion where dependencies permit, and computational performance.

## Release history

| Version | Description |
|---|---|
| 1.0 | Initial public CUDA C++ backend snapshot for LES Modeled Geometric Impulse |

Future updates will be documented as subsequent versions. This is research code under active development. This uploaded backend snapshot has not been compiled or executed on a CUDA GPU in the publishing environment; no new performance or validation claim is made for this release.
