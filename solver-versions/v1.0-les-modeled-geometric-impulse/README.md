# Version 1.0 — LES Modeled Geometric Impulse

Complete source pipeline for the LES-modeled geometric-impulse solver.

[Download the complete pipeline ZIP](../../downloads/LES_Modeled_Geometric_Impulse_v1.0.zip)

## Pipeline

`main.py` → `python_api.py` → compiled CUDA C++ library from `backend.cu` → reports and visualization in Python.

The entire simulation backend is written in CUDA C++: GPU kernels and host routines for geometry, timestep coordination, transport, stretching, molecular diffusion, WALE stresses, boundary treatment, and projection. Python supplies run parameters, reporting, and post-processing through a ctypes C interface.

## Included files

| File | Purpose |
|---|---|
| `main.py` | Run settings, solver calls, diagnostics and Q rendering |
| `python_api.py` | ctypes bridge to the native library |
| `backend.cu` | Current merged CUDA C++ backend, including kernels and interface declarations |
| `surface_reporting.py` | Surface-force reporting and pressure reconstruction |
| `validate_gpu.py` | Small-grid CUDA validation runner |
| `build_windows.bat` | Build the Windows DLL with NVCC and MSVC |
| `build_linux.sh` | Build the Linux shared library with NVCC |
| `requirements.txt` | Python dependencies |

The merged backend contains its own interface declarations and GPU kernels. Separate `kernels.cuh` and `solver_api.h` files are not required.

## Windows and Spyder

1. Install an NVIDIA CUDA Toolkit and compatible Visual Studio C++ Build Tools.
2. Open an **x64 Native Tools Command Prompt for Visual Studio** and enter this folder.
3. Run `build_windows.bat` to build `cylinder_solver.dll`.
4. In the Python environment used by Spyder, run `python -m pip install -r requirements.txt`.
5. Run `python validate_gpu.py` for the small-grid checks.
6. Edit the run settings at the top of `main.py`, then run `python main.py` or open it in Spyder.

Keep the compiled library beside the Python files. Rebuild after changing CUDA C++ source. Changing Python run settings does not require rebuilding.

## Linux

```bash
python -m pip install -r requirements.txt
bash build_linux.sh
python validate_gpu.py
python main.py
```

The build uses cuFFT, cuSOLVER and cuBLAS. Python dependencies are NumPy, SciPy, Matplotlib and PyVista. No CuPy, Numba, pybind11 or CMake is required by this native pipeline.

## Configuration and outputs

The recovered frontend defaults are preserved: Re = 3900, 28 cells per diameter on a 20D by 20D domain, 96 spanwise cells, and dt = 0.0005. Mesh, timestep, output frequencies, and other options are editable in `main.py`; the defaults are not a memory-fit guarantee for every GPU. Results are written under `native_results`.

The backend implements RK4 characteristic tracing, bounded MacCormack transport, local implicit stretching, backward-Euler molecular diffusion with red-black iterations, WALE subgrid stresses, and Poisson projection. Optional PIC/FLIP and Eulerian modes are exposed through the configuration.

## Current work

Currently working on reducing numerical diffusion artifacts from discretization and optimizing CUDA kernels, with emphasis on vortex preservation, memory traffic, and computational performance.

## Release history

| Version | Description |
|---|---|
| 1.0 | Complete Python frontend, ctypes bridge, current CUDA C++ backend, build scripts and reporting pipeline |

Further updates will follow as versioned releases. Research code under active development. Python syntax, frontend/backend interface field order, source packaging and links were checked for this release. NVCC compilation and GPU execution have not been verified in the publishing environment; run `validate_gpu.py` on your CUDA machine.
