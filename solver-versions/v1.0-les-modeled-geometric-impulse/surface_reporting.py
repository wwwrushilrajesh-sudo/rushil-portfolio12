"""Optional CPU postprocessing of copied output; never advances the flow."""
import csv
from types import SimpleNamespace
import numpy as np
from python_api import copy_crop
BODY = "circle"
SURFACE_DRAG_RADIUS_D = 2.6
SURFACE_DRAG_THETA = 256

class SurfaceStressDrag:
    """Estimate physical cylinder surface traction without using impulse phi as pressure.

    At the reporting step, reconstruct the span-averaged pressure gradient from
    the unsteady resolved Navier-Stokes momentum equation in the FLUID region:

      grad(p/rho) = -(du/dt + (u.grad)u - nu*lap(u) - div(2*nu_t*S)).

    Fit a scalar pressure to these gradients (fluid-only least squares),
    extrapolate it to the geometric wall, and integrate pressure + no-slip
    molecular wall shear around a circular cylinder. This is a POSTPROCESSING
    ESTIMATE, NOT a proven grid-converged drag coefficient: cut-cell pressure,
    immersed-boundary consistency and SGS stresses still need validation.

    Runs on a small cropped CPU array every SURFACE_DRAG_EVERY steps; does not
    edit the CFD state or allocate another full GPU velocity volume.
    """

    def __init__(self, sim):
        if BODY != 'circle':
            raise ValueError('SurfaceStressDrag currently supports BODY="circle" only.')
        from scipy import sparse
        from scipy.sparse.linalg import LinearOperator
        self.sparse = sparse
        self.LinearOperator = LinearOperator
        self.h, self.hz = (sim.h, sim.hz)
        self.cx, self.cy = (sim.xc, sim.yc)
        self.nx, self.ny, self.nz = (sim.nx, sim.ny, sim.nz)
        self.radius = D * 0.5
        box_half = int(np.ceil(SURFACE_DRAG_RADIUS_D * D / self.h)) + 3
        xmid, ymid = (int(round(sim.xc)), int(round(sim.yc)))
        self.x0, self.x1 = (max(0, xmid - box_half), min(sim.nx, xmid + box_half))
        self.y0, self.y1 = (max(0, ymid - box_half), min(sim.ny - 1, ymid + box_half))
        self.W, self.H = (self.x1 - self.x0, self.y1 - self.y0)
        self.xx = (np.arange(self.x0, self.x1) + 0.5 - self.cx) * self.h
        self.yy = (np.arange(self.y0, self.y1) + 0.5 - self.cy) * self.h
        X, Y = np.meshgrid(self.xx, self.yy)
        rr = np.hypot(X, Y)
        self.good = (rr >= self.radius + 2.25 * self.h) & (rr <= SURFACE_DRAG_RADIUS_D * D)
        if not (np.any(self.good) and np.all(self.good.sum(axis=1) >= 0)):
            raise RuntimeError('Pressure fitting region is empty.')
        labels = -np.ones((self.H, self.W), dtype=np.int32)
        labels[self.good] = np.arange(self.good.sum(), dtype=np.int32)
        yi, xi = np.nonzero(self.good[:, :-1] & self.good[:, 1:])
        yj, xj = np.nonzero(self.good[:-1, :] & self.good[1:, :])
        self.hx_edges = (yi, xi)
        self.hy_edges = (yj, xj)
        start = np.concatenate((labels[yi, xi], labels[yj, xj]))
        end = np.concatenate((labels[yi, xi + 1], labels[yj + 1, xj]))
        m = len(start)
        rows = np.repeat(np.arange(m), 2)
        columns = np.empty(2 * m, dtype=np.int32)
        columns[0::2], columns[1::2] = (start, end)
        vals = np.tile(np.array([-1.0, 1.0]), m)
        self.A = sparse.coo_matrix((vals, (rows, columns)), shape=(m, int(self.good.sum()))).tocsr()
        lap = (self.A.T @ self.A).tolil()
        lap[0, 0] += 1.0
        self.L = lap.tocsr()
        inverse_diag = 1.0 / self.L.diagonal()
        self.M = LinearOperator(self.L.shape, matvec=lambda v: inverse_diag * v)
        self.theta = np.arange(SURFACE_DRAG_THETA) * 2 * np.pi / SURFACE_DRAG_THETA
        self.sin = np.sin(self.theta)
        self.cos = np.cos(self.theta)
        print(f'[Surface drag] Local fluid pressure fit: {self.good.sum():,} nodes, {m:,} face-gradient constraints, {SURFACE_DRAG_THETA} wall angles. This is an estimated physical surface-stress diagnostic, not validated CD.', flush=True)

    def snapshot(self, handle):
        """Copy only the local surface-reporting crop, never whole flow volumes."""
        uf = copy_crop(handle, self.config, "u", self.x0, self.y0, self.W+1, self.H)
        vf = copy_crop(handle, self.config, "v", self.x0, self.y0, self.W, self.H+1)
        wf = copy_crop(handle, self.config, "w", self.x0, self.y0, self.W, self.H)
        return (0.5*(uf[:,:,:-1]+uf[:,:,1:]),
                0.5*(vf[:,:-1,:]+vf[:,1:,:]),
                0.5*(wf+np.roll(wf,-1,axis=0)))

    def deriv(self, f, axis):
        if axis == 0:
            return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * self.hz)
        return np.gradient(f, self.h, axis=axis, edge_order=2)

    def laplacian(self, f):
        h2, hz2 = (self.h * self.h, self.hz * self.hz)
        return (np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / hz2 + (np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)) / h2 + (np.roll(f, -1, axis=2) - 2 * f + np.roll(f, 1, axis=2)) / h2

    def fit_pressure(self, gx, gy):
        """Find pressure/rho whose FLUID-FACE differences best fit (gx, gy)."""
        from scipy.sparse.linalg import cg
        iy, ix = self.hx_edges
        jy, jx = self.hy_edges
        wanted = np.concatenate((0.5 * self.h * (gx[iy, ix] + gx[iy, ix + 1]), 0.5 * self.h * (gy[jy, jx] + gy[jy + 1, jx])))
        rhs = self.A.T @ wanted
        p, info = cg(self.L, rhs, M=self.M, rtol=2e-05, atol=0.0, maxiter=2200)
        if info != 0:
            raise RuntimeError(f'Pressure gradient least-squares CG failed: info={info}')
        error = np.linalg.norm(self.A @ p - wanted) / max(np.linalg.norm(wanted), 1e-20)
        grid = np.full((self.H, self.W), np.nan, dtype=np.float64)
        grid[self.good] = p
        return (grid, float(error))

    def sample(self, f, radii):
        from scipy.ndimage import map_coordinates
        xs = self.cx + radii * self.cos / self.h - 0.5 - self.x0
        ys = self.cy + radii * self.sin / self.h - 0.5 - self.y0
        return map_coordinates(f, [ys, xs], order=1, mode='nearest')

    def calculate(self, sim, old_velocity):
        """Physical pressure+skin-friction ESTIMATE at a full accepted time step."""
        if old_velocity is None:
            raise ValueError('Need velocity snapshot from immediately preceding step.')
        u, v, w = self.snapshot(sim)
        u0, v0, _ = old_velocity
        dt = self.current_report["dt"]
        if dt <= 0:
            raise RuntimeError('Time step must be positive.')
        ux, uy, uz = (self.deriv(u, 2), self.deriv(u, 1), self.deriv(u, 0))
        vx, vy, vz = (self.deriv(v, 2), self.deriv(v, 1), self.deriv(v, 0))
        wx, wy = (self.deriv(w, 2), self.deriv(w, 1))
        fx = (u - u0) / dt + u * ux + v * uy + w * uz - NU * self.laplacian(u)
        fy = (v - v0) / dt + u * vx + v * vy + w * vz - NU * self.laplacian(v)
        del u0, v0
        if LES_ON:
            nut = copy_crop(sim, self.config, "nut", self.x0, self.y0, self.W, self.H)
            txx = 2 * nut * ux
            txy = nut * (uy + vx)
            txz = nut * (uz + wx)
            tyy = 2 * nut * vy
            tyz = nut * (vz + wy)
            fx -= self.deriv(txx, 2) + self.deriv(txy, 1) + self.deriv(txz, 0)
            fy -= self.deriv(txy, 2) + self.deriv(tyy, 1) + self.deriv(tyz, 0)
            del txx, txy, txz, tyy, tyz, nut
        gx, gy = (-np.mean(fx, axis=0, dtype=np.float64), -np.mean(fy, axis=0, dtype=np.float64))
        p, gradient_misfit = self.fit_pressure(gx, gy)
        h = self.h
        pa = self.sample(p, self.radius + 4.5 * h)
        pb = self.sample(p, self.radius + 6.0 * h)
        pw = (6.0 * pa - 4.5 * pb) / 1.5
        ubar, vbar = (u.mean(axis=0), v.mean(axis=0))
        ua = self.sample(ubar, self.radius + 1.5 * h)
        va = self.sample(vbar, self.radius + 1.5 * h)
        ub = self.sample(ubar, self.radius + 3.0 * h)
        vb = self.sample(vbar, self.radius + 3.0 * h)
        uta = -ua * self.sin + va * self.cos
        utb = -ub * self.sin + vb * self.cos
        dut_dn = (4 * uta - utb) / (3 * h)
        dtheta = 2 * np.pi / len(self.theta)
        norm = dtheta / (U_C * U_C)
        cd_p = -norm * np.sum(pw * self.cos)
        cl_p = -norm * np.sum(pw * self.sin)
        cd_mu = norm * np.sum(NU * dut_dn * -self.sin)
        cl_mu = norm * np.sum(NU * dut_dn * self.cos)
        if not np.all(np.isfinite((cd_p, cl_p, cd_mu, cl_mu, gradient_misfit))):
            raise FloatingPointError('Nonfinite pressure or wall shear in surface-drag diagnostic')
        return dict(CD_pressure_est=float(cd_p), CD_viscous_est=float(cd_mu), CD_surface_est=float(cd_p + cd_mu), CL_pressure_est=float(cl_p), CL_viscous_est=float(cl_mu), CL_surface_est=float(cl_p + cl_mu), p_gradient_fit_misfit=float(gradient_misfit))
    def record(self, handle, report, old_velocity, path):
        self.current_report = report
        try:
            values = self.calculate(handle, old_velocity)
        except Exception as error:
            print(f"[Surface reporting] step={report['steps']}: {error}", flush=True)
            return
        row = dict(step=report["steps"], time=report["time"], **values)
        new_file = not path.exists()
        with path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)


def create_surface_reporter(config):
    global D, U_C, NU, LES_ON
    D = config.diameter
    U_C = config.inlet
    NU = U_C*D/config.reynolds
    LES_ON = bool(config.les)
    geometry = SimpleNamespace(h=config.lx/config.nx, hz=config.lz/config.nz,
        xc=config.body_x/(config.lx/config.nx), yc=config.ny/2,
        nx=config.nx, ny=config.ny, nz=config.nz)
    reporter = SurfaceStressDrag(geometry)
    reporter.config = config
    return reporter

