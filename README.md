# Rushil Rajesh Engineering Portfolio

Responsive static portfolio for CFD, scientific computing, mechanical design, product development, mechatronics, and thermal-fluid engineering.

## Velocity–Impulse CFD solver releases

The original [E–Liu CPU/SOR](./solver-versions/01_el_cpu_sor.py) and [Geometric CPU/SOR](./solver-versions/02_geometric_cpu_sor.py) baselines remain public.

The current release is [Version 1.0 — LES Modeled Geometric Impulse](./solver-versions/v1.0-les-modeled-geometric-impulse/). Its entire simulation backend is written in CUDA C++; Python provides the frontend and post-processing. The release publishes the current backend source.

Currently working on reducing numerical diffusion artifacts from discretization and optimizing CUDA kernels. Further updates will follow as versioned releases.

## Portfolio coverage

The website presents the Velocity–Impulse CFD solver as a dedicated research section, with separate detailed project writeups for:

- Truncated Linear Aerospike Nozzle
- Centrifugal Pump Impeller Design & CFD
- EV Moped Battery Pack Thermal Management
- Gesture-Controlled Wheelchair Prototype
- Oscillating-Cylinder Lattice Boltzmann Solver
- Heat-Sink / Fan Optimization

## Portfolio stack

- HTML5
- CSS3
- Vanilla JavaScript
- GitHub Pages compatible

## Main site files

- `index.html`
- `styles.css`
- `script.js`

## Deployment

The portfolio is deployed from the `main` branch using GitHub Pages.
