"""Python front end: parameters, native solver calls, reporting and plots."""
from pathlib import Path
import csv
import json
import math
import time
import numpy as np
from python_api import initialize, advance, copy_field, copy_mean_line, copy_q_region, close, diagnostics

# ------------------------ EDIT YOUR RUN HERE ------------------------
D = 1.0
U = 1.0
RE = 3900.0
CELLS_PER_D = 28                 # Same selected preset as the supplied source.
NX = 20 * CELLS_PER_D
NY = 20 * CELLS_PER_D
NZ = 96
DT = 0.0005
TOTAL_STEPS = 200000
PRINT_EVERY = 500
Q_EVERY = 1000
Q_LEVEL = 0.5                    # Q* = Q*D^2/U^2; fixed threshold.
SAVE_Q = True
SHOW_Q = True
Q_MAX_TRIANGLES = 5000000
SURFACE_DRAG_ON = True
SURFACE_DRAG_EVERY = 200
OUTPUT_DIR = Path(__file__).resolve().parent / "native_results"
VIEW_X = (-2.0, 14.0)
VIEW_Y = (-4.0, 4.0)

PARAMETERS = dict(
    nx=NX, ny=NY, nz=NZ,
    les=1, adaptive=0, adapt_cfl=0, maccormack=1,
    be_max_sweeps=80, be_check_every=4, max_retries=8,
    mode=0,                        # 0=impulse, 1=Eulerian, 2=Eulerian+confinement
    high_order=0, particle_advection=0, particles_per_cell=2, two_d=0, seed=3,
    diameter=D, inlet=U, reynolds=RE, lx=20*D, ly=20*D, lz=math.pi*D, body_x=5*D,
    dt=DT, dt_max=DT, retention=0.98, stretch_alpha=1.0, wale_cw=0.5,
    be_tolerance=2e-5, correction_blend=0.5,
    cfl_target=0.5, gradient_limit=0.8, dt_safety=0.98,
    diffusion_limit=0.45, determinant_limit=0.10,
    speed_growth=1.10, energy_growth=1.10,
    perturbation=0.01, flip_fraction=1.0, vc_epsilon=10.0,
    stats_start=150*D/U, probe_x=3.0,
)
# No advection, viscosity, WALE, Poisson or projection implementation follows.
# All those operations run inside backend.cu when advance() is called.


def save_q_frame(handle, config, report):
    """VTK mesh extraction and rendering are postprocessing on copied output."""
    import pyvista as pv
    q, speed, x0, y0 = copy_q_region(handle, config, VIEW_X, VIEW_Y)
    h = config.lx/config.nx
    grid = pv.ImageData(dimensions=(q.shape[2], q.shape[1], q.shape[0]),
        spacing=(h/D, h/D, config.lz/config.nz/D),
        origin=((x0+0.5)*h/D-config.body_x/D, (y0+0.5)*h/D-config.ly/(2*D), 0.5*config.lz/config.nz/D))
    grid.point_data["Q"] = q.ravel(order="C")
    grid.point_data["speed"] = speed.ravel(order="C")
    # Match the original renderer: ignore cubes touching solid-cell sentinels.
    good = q > -1e20
    cell_good = (good[:-1,:-1,:-1] & good[:-1,:-1,1:] & good[:-1,1:,:-1] & good[:-1,1:,1:] &
                 good[1:,:-1,:-1] & good[1:,:-1,1:] & good[1:,1:,:-1] & good[1:,1:,1:])
    valid = grid.extract_cells(np.flatnonzero(cell_good.ravel(order="C")))
    surface = valid.contour([Q_LEVEL], scalars="Q")
    if surface.n_cells > Q_MAX_TRIANGLES:
        raise RuntimeError(f"Q surface has {surface.n_cells:,} cells; increase Q_LEVEL or Q_MAX_TRIANGLES")
    plot = pv.Plotter(off_screen=True, window_size=(1600, 900))
    plot.set_background("white")
    if surface.n_points:
        plot.add_mesh(surface, scalars="speed", cmap="jet", clim=(0, 1.8), smooth_shading=True)
    cylinder = pv.Cylinder(center=(0, 0, config.lz/(2*D)), direction=(0, 0, 1), radius=0.5, height=config.lz/D, resolution=72)
    plot.add_mesh(cylinder, color="gray")
    plot.add_text(f"Re={RE:g} | t={report['time']:.3f} | step={report['steps']} | Q*={Q_LEVEL:g}", font_size=12)
    plot.view_isometric()
    path = OUTPUT_DIR/f"Q_step{report['steps']:09d}.png" if SAVE_Q else None
    plot.show(auto_close=False, interactive=False)
    image = plot.screenshot(str(path) if path else None, return_img=True)
    plot.close()
    if SHOW_Q:
        import matplotlib.pyplot as plt
        figure = plt.figure("Native cylinder Q", figsize=(12,7))
        figure.clear()
        axis = figure.add_subplot(111)
        axis.imshow(image)
        axis.axis("off")
        figure.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)


def estimate_strouhal(times, values, start):
    times = np.asarray(times); values = np.asarray(values)
    keep = times >= start
    times = times[keep]; values = values[keep]
    if len(times) < 64 or times[-1]-times[0] < 50*D/U or np.std(values) < 1e-12:
        return float("nan")
    uniform = np.linspace(times[0], times[-1], len(times))
    signal = np.interp(uniform, times, values)
    signal = (signal-signal.mean())*np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(signal, 8*len(signal)))
    frequency = np.fft.rfftfreq(8*len(signal), uniform[1]-uniform[0])
    spectrum[frequency*D/U < 0.02] = 0
    return frequency[np.argmax(spectrum)]*D/U


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR/"settings.json").write_text(json.dumps(PARAMETERS, indent=2))
    print(f"Native CUDA C++ solver: {NX} x {NY} x {NZ}, Re={RE:g}, dt={DT:g}", flush=True)
    print("Preparing geometry, wall constraints and FFT plans...", flush=True)
    handle, config = initialize(PARAMETERS)
    times, probes = [], []
    surface_drag = None
    if SURFACE_DRAG_ON:
        try:
            from surface_reporting import create_surface_reporter
            surface_drag = create_surface_reporter(config)
        except Exception as error:
            print(f"[Surface reporting] unavailable: {error}", flush=True)
    started = time.perf_counter()
    try:
        with (OUTPUT_DIR/"diagnostics.csv").open("w", newline="") as stream:
            writer = None
            for step in range(1, TOTAL_STEPS+1):
                # One native call advances ALL solver stages. Python reads one report.
                old = surface_drag.snapshot(handle) if surface_drag and step % SURFACE_DRAG_EVERY == 0 else None
                report = advance(handle, 1)
                times.append(report["time"]); probes.append(report["probe_velocity"])
                if surface_drag and step % SURFACE_DRAG_EVERY == 0:
                    surface_drag.record(handle, report, old, OUTPUT_DIR/"surface_forces.csv")
                if step % PRINT_EVERY == 0:
                    report = diagnostics(handle)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(report))
                        writer.writeheader()
                    writer.writerow(report); stream.flush()
                    print(f"step={step} t={report['time']:.4g} CFL={report['cfl']:.4g} "
                          f"div={report['divergence_rms']:.3g} nut/nu={report['nut_ratio_mean']:.4g} "
                          f"CD_wall={report['cd_wall']:.4g} elapsed={time.perf_counter()-started:.1f}s", flush=True)
                if (SAVE_Q or SHOW_Q) and step % Q_EVERY == 0:
                    save_q_frame(handle, config, report)
            x = (np.arange(NX)+0.5)*(config.lx/NX)/D-config.body_x/D
            mean, rms = copy_mean_line(handle, config)
            np.savetxt(OUTPUT_DIR/"mean_centerline.csv", np.column_stack((x,mean,rms)), delimiter=",", header="x_over_D,u_mean_over_U,u_rms_over_U", comments="")
            np.savetxt(OUTPUT_DIR/"wake_probe.csv", np.column_stack((times,probes)), delimiter=",", header="time,v_probe", comments="")
            print("Strouhal:", estimate_strouhal(times, probes, PARAMETERS["stats_start"]))
    finally:
        close(handle)


if __name__ == "__main__":
    main()
