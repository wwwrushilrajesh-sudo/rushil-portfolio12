"""A small ctypes bridge. No numerical solver stages are implemented here."""
import ctypes as ct
import os
from pathlib import Path
import numpy as np

INT_FIELDS = "nx ny nz les adaptive adapt_cfl maccormack be_max_sweeps be_check_every max_retries mode high_order particle_advection particles_per_cell two_d seed".split()
DOUBLE_FIELDS = "diameter inlet reynolds lx ly lz body_x dt dt_max retention stretch_alpha wale_cw be_tolerance correction_blend cfl_target gradient_limit dt_safety diffusion_limit determinant_limit speed_growth energy_growth perturbation flip_fraction vc_epsilon stats_start probe_x".split()

# These two small records must match solver_api.h. They only transfer parameters
# and diagnostic numbers; all flow arrays are owned by the C++ backend.
class Config(ct.Structure):
    _fields_ = [(name, ct.c_int) for name in INT_FIELDS] + [(name, ct.c_double) for name in DOUBLE_FIELDS]

class Report(ct.Structure):
    _fields_ = [("steps", ct.c_longlong)] + [(name, ct.c_double) for name in
        "time dt cfl energy peak_velocity peak_impulse divergence_rms wall_slip nut_ratio_mean nut_ratio_max min_determinant gradient_dt sgs_number cd_wall cl_wall probe_velocity averaging_duration".split()] + [("diffusion_sweeps", ct.c_int), ("retries", ct.c_int)]

library = None
_dll_directories = []

def load_backend():
    global library
    folder = Path(__file__).resolve().parent
    if os.name == "nt":
        cuda_root = os.environ.get("CUDA_PATH")
        if cuda_root:
            for candidate in [Path(cuda_root) / "bin", Path(cuda_root) / "bin" / "x64"]:
                if candidate.is_dir():
                    _dll_directories.append(os.add_dll_directory(str(candidate)))
        filename = folder / "cylinder_solver.dll"
    else:
        filename = folder / "libcylinder_solver.so"
    if not filename.is_file():
        raise FileNotFoundError(f"Build the CUDA backend first. Missing: {filename}")
    library = ct.CDLL(str(filename))
    handle = ct.c_void_p
    floats = ct.POINTER(ct.c_float)
    doubles = ct.POINTER(ct.c_double)
    library.solver_create.argtypes = [ct.POINTER(Config), ct.POINTER(handle)]
    library.solver_advance.argtypes = [handle, ct.c_int, ct.POINTER(Report)]
    library.solver_report.argtypes = [handle, ct.POINTER(Report)]
    library.solver_copy_field.argtypes = [handle, ct.c_int, floats, ct.c_longlong]
    library.solver_copy_line.argtypes = [handle, doubles, doubles, ct.c_int]
    library.solver_q_volume.argtypes = [handle, ct.c_int, ct.c_int, ct.c_int, ct.c_int, floats, floats, ct.c_longlong]
    library.solver_copy_crop.argtypes = [handle, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, floats, ct.c_longlong]
    library.solver_destroy.argtypes = [handle]
    for name in ["solver_create", "solver_advance", "solver_report", "solver_copy_field", "solver_copy_line", "solver_q_volume", "solver_copy_crop", "solver_destroy"]:
        getattr(library, name).restype = ct.c_int
    library.solver_last_error.argtypes = []
    library.solver_last_error.restype = ct.c_char_p

def check(status):
    if status:
        raise RuntimeError(library.solver_last_error().decode("utf-8", errors="replace"))

def initialize(parameters):
    if library is None:
        load_backend()
    config = Config(**parameters)
    handle = ct.c_void_p()
    check(library.solver_create(ct.byref(config), ct.byref(handle)))
    return handle, config

def advance(handle, number_of_steps):
    result = Report()
    check(library.solver_advance(handle, number_of_steps, ct.byref(result)))
    return {name: getattr(result, name) for name, _ in Report._fields_}

def copy_field(handle, config, name):
    field = {"u": 0, "v": 1, "w": 2, "mx": 3, "my": 4, "mz": 5, "phi": 6, "nut": 7}[name]
    sx = config.nx + (field in (0, 3))
    values = np.empty((config.nz, config.ny, sx), dtype=np.float32)
    check(library.solver_copy_field(handle, field, values.ctypes.data_as(ct.POINTER(ct.c_float)), values.size))
    return values

def copy_mean_line(handle, config):
    mean = np.empty(config.nx, dtype=np.float64)
    rms = np.empty(config.nx, dtype=np.float64)
    check(library.solver_copy_line(handle, mean.ctypes.data_as(ct.POINTER(ct.c_double)), rms.ctypes.data_as(ct.POINTER(ct.c_double)), config.nx))
    return mean, rms

def copy_q_region(handle, config, view_x, view_y):
    h = config.lx / config.nx
    x0 = max(0, int((config.body_x + view_x[0] * config.diameter) / h))
    x1 = min(config.nx, int((config.body_x + view_x[1] * config.diameter) / h))
    y0 = max(0, int((config.ly / 2 + view_y[0] * config.diameter) / h))
    y1 = min(config.ny, int((config.ly / 2 + view_y[1] * config.diameter) / h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Q display region is empty")
    q = np.empty((config.nz, y1-y0, x1-x0), dtype=np.float32)
    speed = np.empty_like(q)
    check(library.solver_q_volume(handle, x0, y0, x1-x0, y1-y0,
        q.ctypes.data_as(ct.POINTER(ct.c_float)), speed.ctypes.data_as(ct.POINTER(ct.c_float)), q.size))
    return q, speed, x0, y0

def close(handle):
    check(library.solver_destroy(handle))


def copy_crop(handle, config, name, x0, y0, width, height):
    field = {"u":0,"v":1,"w":2,"mx":3,"my":4,"mz":5,"phi":6,"nut":7}[name]
    values = np.empty((config.nz, height, width), dtype=np.float32)
    check(library.solver_copy_crop(handle, field, x0, y0, width, height,
        values.ctypes.data_as(ct.POINTER(ct.c_float)), values.size))
    return values


def diagnostics(handle):
    result = Report()
    check(library.solver_report(handle, ct.byref(result)))
    return {name: getattr(result, name) for name, _ in Report._fields_}
