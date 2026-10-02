"""
Aerospike MOC Contour Generator — Portfolio Demo
Rushil Rajesh

Purpose
-------
This public portfolio script demonstrates a compact Method-of-Characteristics-
inspired workflow for creating a supersonic aerospike/plug contour from gas
properties and a target design Mach number.

It is intentionally a simplified, public-facing example and is NOT the exact
proprietary capstone code used for the original nozzle design.

Outputs
-------
1. Prandtl-Meyer expansion angles
2. A smooth expansion/plug contour
3. Optional geometric truncation
4. CSV point export for CAD / downstream CFD setup
5. A clean contour plot

Dependencies
------------
numpy
matplotlib
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GAMMA = 1.4
M_EXIT = 3.0
N_POINTS = 180
TRUNCATION_FRACTION = 0.40
THROAT_HEIGHT = 1.0
OUTPUT_CSV = "aerospike_contour.csv"


def prandtl_meyer(M: float, gamma: float = GAMMA) -> float:
    """Return Prandtl-Meyer angle nu(M) in radians."""
    if M < 1.0:
        raise ValueError("Prandtl-Meyer function requires M >= 1.")

    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    a = math.sqrt(gp1 / gm1)
    b = math.sqrt((gm1 / gp1) * (M * M - 1.0))
    c = math.sqrt(M * M - 1.0)

    return a * math.atan(b) - math.atan(c)


def inverse_prandtl_meyer(
    nu_target: float,
    gamma: float = GAMMA,
    lo: float = 1.000001,
    hi: float = 50.0,
    tol: float = 1e-10,
) -> float:
    """Invert nu(M) with a robust bisection solve."""
    if nu_target < 0.0:
        raise ValueError("nu_target must be non-negative.")

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        nu_mid = prandtl_meyer(mid, gamma)

        if abs(nu_mid - nu_target) < tol:
            return mid

        if nu_mid < nu_target:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def mach_angle(M: float) -> float:
    """Mach angle mu in radians."""
    return math.asin(1.0 / M)


def build_plug_contour(
    M_exit: float = M_EXIT,
    gamma: float = GAMMA,
    n: int = N_POINTS,
    throat_height: float = THROAT_HEIGHT,
    truncation_fraction: float = TRUNCATION_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a smooth MOC-inspired plug contour.

    The expansion is parameterized through Prandtl-Meyer turning from near-sonic
    throat conditions toward the design exit Mach number. Local characteristic
    inclination is used to integrate the contour downstream.

    This is intended for rapid design exploration / CAD seeding rather than as a
    replacement for a full characteristic net.
    """
    if not (0.0 < truncation_fraction <= 1.0):
        raise ValueError("truncation_fraction must lie in (0, 1].")

    nu_exit = prandtl_meyer(M_exit, gamma)
    nu = np.linspace(1e-8, nu_exit, n)

    mach = np.array([inverse_prandtl_meyer(v, gamma) for v in nu])
    mu = np.arcsin(1.0 / mach)

    # Smooth wall turning schedule.
    theta = 0.50 * nu

    # Characteristic-guided local wall slope.
    local_angle = np.maximum(mu - theta, np.deg2rad(1.0))

    x = np.zeros(n)
    y = np.zeros(n)
    y[0] = throat_height

    dx = throat_height * 0.055
    for i in range(1, n):
        x[i] = x[i - 1] + dx
        y[i] = y[i - 1] - dx * math.tan(local_angle[i - 1])

    # Shift and normalize so the contour terminates cleanly.
    y -= y.min()
    y *= throat_height / max(y.max(), 1e-12)

    # Truncate by axial extent.
    x_cut = x[0] + truncation_fraction * (x[-1] - x[0])
    keep = x <= x_cut

    return x[keep], y[keep], mach[keep]


def export_csv(path: str | Path, x: np.ndarray, y: np.ndarray, mach: np.ndarray) -> None:
    """Export contour points for CAD or downstream preprocessing."""
    path = Path(path)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "Mach"])
        writer.writerows(zip(x, y, mach))


def plot_contour(x: np.ndarray, y: np.ndarray) -> None:
    """Plot the generated plug contour."""
    plt.figure(figsize=(9, 4.8))
    plt.plot(x, y, linewidth=2.2)
    plt.fill_between(x, 0.0, y, alpha=0.12)

    plt.xlabel("x / throat height")
    plt.ylabel("plug height / throat height")
    plt.title("MOC-inspired truncated aerospike contour")
    plt.axis("equal")
    plt.grid(alpha=0.22)
    plt.tight_layout()
    plt.show()


def main() -> None:
    x, y, mach = build_plug_contour()

    export_csv(OUTPUT_CSV, x, y, mach)

    print(f"Generated {len(x)} contour points")
    print(f"Target design Mach number: {M_EXIT:.3f}")
    print(f"Truncation fraction: {TRUNCATION_FRACTION:.2f}")
    print(f"Saved CAD-ready points to: {OUTPUT_CSV}")

    plot_contour(x, y)


if __name__ == "__main__":
    main()
