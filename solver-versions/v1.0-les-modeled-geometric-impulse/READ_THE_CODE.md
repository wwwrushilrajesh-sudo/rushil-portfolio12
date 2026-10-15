# Reading the complete solver pipeline

1. Start with the run settings and `main()` in `main.py`.
2. `python_api.py` passes parameter records and native function calls through ctypes.
3. In `backend.cu`, start at `advance_one_step`. Follow its named numerical stages to the CUDA kernels they launch.
4. `__global__` functions run as GPU kernels. `__device__` helpers run inside GPU threads. Ordinary C++ host functions organize stages and launches.
5. Return to `main.py` and `surface_reporting.py` for diagnostics, field export and visualization.

The backend begins with an extended guide to arrays, indexing and the native interface. Kernels, host orchestration and interface declarations are all in the single merged `backend.cu` file.
