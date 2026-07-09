"""Standard potentials and their analytic reference results (hbar = m = 1)."""

from __future__ import annotations

import numpy as np

from .core import Grid

__all__ = [
    "harmonic",
    "square_barrier",
    "double_well",
    "harmonic_ground_energy",
    "coherent_center",
    "free_packet_width",
    "barrier_transmission",
]


def harmonic(grid: Grid, omega: float = 1.0, x0: float = 0.0) -> np.ndarray:
    """V = omega^2 (x - x0)^2 / 2."""
    return 0.5 * omega**2 * (grid.x - x0) ** 2


def square_barrier(grid: Grid, height: float, width: float, center: float = 0.0) -> np.ndarray:
    """Rectangular barrier of given height and width."""
    return np.where(np.abs(grid.x - center) <= width / 2.0, height, 0.0)


def double_well(grid: Grid, barrier: float = 1.0, minima: float = 2.0) -> np.ndarray:
    """Quartic double well V = barrier * ((x/minima)^2 - 1)^2."""
    return barrier * ((grid.x / minima) ** 2 - 1.0) ** 2


def harmonic_ground_energy(omega: float = 1.0) -> float:
    """E_0 = omega / 2 in natural units."""
    return omega / 2.0


def coherent_center(t: float, x0: float, omega: float = 1.0) -> float:
    """A coherent state's center follows the classical trajectory x0 cos(wt)."""
    return x0 * np.cos(omega * t)


def free_packet_width(t: float, sigma0: float) -> float:
    """Free Gaussian packet width sigma(t) = sigma0 sqrt(1 + (t / (2 sigma0^2))^2)."""
    return sigma0 * np.sqrt(1.0 + (t / (2.0 * sigma0**2)) ** 2)


def barrier_transmission(energy: float, height: float, width: float) -> float:
    """Plane-wave transmission through a rectangular barrier (exact).

    For E < V0: T = [1 + V0^2 sinh^2(kappa a) / (4 E (V0 - E))]^-1
    For E > V0: T = [1 + V0^2 sin^2(k a)    / (4 E (E - V0))]^-1
    """
    if energy <= 0:
        raise ValueError("energy must be positive")
    if energy == height:
        # limit V0 -> E: T = 1 / (1 + E a^2 / 2)  (from sin(x) ~ x)
        return 1.0 / (1.0 + energy * width**2 / 2.0)
    if energy < height:
        kappa = np.sqrt(2.0 * (height - energy))
        s = np.sinh(kappa * width)
        return float(1.0 / (1.0 + height**2 * s**2 / (4.0 * energy * (height - energy))))
    k = np.sqrt(2.0 * (energy - height))
    s = np.sin(k * width)
    return float(1.0 / (1.0 + height**2 * s**2 / (4.0 * energy * (energy - height))))
