# Rushil Rajesh Engineering Portfolio

Responsive static portfolio for CFD, scientific computing, mechanical design, product development, mechatronics, and thermal-fluid engineering.

## Velocity–Impulse CFD solver archive

The repository includes selected **non-turbulence** development snapshots in [`solver-versions/`](./solver-versions/).

| Version | Status | Description |
|---|---|---|
| V1 | Public | 2D E–Liu CPU solver with RK4 + SOR projection |
| V2 | Public | 2D geometric impulse solver with stretching contribution |
| V3 | Public | CUDA E–Liu solver with residual-controlled Jacobi projection |
| V4 | Private | 3D semi-Lagrangian + SGS research branch |
| V5 | Private | 3D CUDA LES performance branch |
| V6 | Private | E–Liu + k–ω SST + direct DCT projection |
| V7 | Private | Eulerian geometric + k–ω SST + DCT projection |
| V8 | Private | Geometric SST revision 2 |
| V9 | Private | Semi-Lagrangian geometric + k–ω SST |

**All turbulence-modeling source code is being withheld from the public repository for now.** The current plan is to release the relevant research branches after publication of the associated results.

[Browse the public solver archive](./solver-versions/README.md)

## Portfolio project coverage

The website includes detailed technical writeups for:

- Velocity–Impulse Incompressible Flow Solver
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
