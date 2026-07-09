"""Split-operator time evolution for the 1-D Schrödinger equation.

Natural units hbar = m = 1 throughout. The propagator for one step dt is the
Strang splitting

    exp(-i H dt) ≈ exp(-i V dt/2) exp(-i T dt) exp(-i V dt/2) + O(dt^3),

with the kinetic factor applied exactly in momentum space via FFT. The
scheme is unitary by construction (each factor is a pure phase), so the norm
is conserved to machine precision — and the test suite demands exactly that.

Imaginary time (t -> -i tau) turns the same code into a ground-state solver:
the propagator becomes exp(-H tau), which exponentially damps every excited
component; renormalizing after each step converges to the ground state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["Grid", "gaussian_packet", "evolve", "imaginary_time_ground_state"]


@dataclass(frozen=True)
class Grid:
    """Uniform spatial grid with its conjugate momentum grid."""

    x_min: float
    x_max: float
    n: int

    def __post_init__(self) -> None:
        if self.n < 8 or self.n & (self.n - 1):
            raise ValueError("n must be a power of two >= 8 (FFT efficiency and accuracy)")
        if self.x_max <= self.x_min:
            raise ValueError("x_max must exceed x_min")

    @property
    def x(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.n, endpoint=False)

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.n

    @property
    def k(self) -> np.ndarray:
        return 2.0 * np.pi * np.fft.fftfreq(self.n, self.dx)

    def norm(self, psi: np.ndarray) -> float:
        return float(np.sqrt(np.sum(np.abs(psi) ** 2) * self.dx))

    def expectation_x(self, psi: np.ndarray) -> float:
        density = np.abs(psi) ** 2
        return float(np.sum(self.x * density) * self.dx)

    def expectation_p(self, psi: np.ndarray) -> float:
        psi_k = np.fft.fft(psi)
        density_k = np.abs(psi_k) ** 2
        return float(np.sum(self.k * density_k) / np.sum(density_k))

    def width_x(self, psi: np.ndarray) -> float:
        density = np.abs(psi) ** 2
        mean = self.expectation_x(psi)
        var = np.sum((self.x - mean) ** 2 * density) * self.dx
        return float(np.sqrt(var))

    def energy(self, psi: np.ndarray, potential: np.ndarray) -> float:
        """<H> = <T> + <V> with T evaluated spectrally."""
        psi_k = np.fft.fft(psi)
        kinetic = np.sum(0.5 * self.k**2 * np.abs(psi_k) ** 2) / np.sum(np.abs(psi_k) ** 2)
        density = np.abs(psi) ** 2
        pot = np.sum(potential * density) * self.dx / (np.sum(density) * self.dx)
        return float(kinetic + pot)


def gaussian_packet(
    grid: Grid, x0: float, k0: float, sigma: float
) -> np.ndarray:
    """Normalized Gaussian wavepacket centered at x0 with mean momentum k0."""
    psi = np.exp(-((grid.x - x0) ** 2) / (4.0 * sigma**2) + 1j * k0 * grid.x)
    return psi / grid.norm(psi)


def evolve(
    psi: np.ndarray,
    potential: np.ndarray,
    grid: Grid,
    dt: float,
    n_steps: int,
    *,
    callback: Callable[[int, np.ndarray], None] | None = None,
    callback_every: int = 1,
) -> np.ndarray:
    """Real-time Strang-split evolution; returns the final wavefunction."""
    half_v = np.exp(-0.5j * dt * potential)
    kinetic = np.exp(-0.5j * dt * grid.k**2)
    psi = psi.astype(complex)
    for step in range(n_steps):
        psi = half_v * psi
        psi = np.fft.ifft(kinetic * np.fft.fft(psi))
        psi = half_v * psi
        if callback is not None and step % callback_every == 0:
            callback(step, psi)
    return psi


def imaginary_time_ground_state(
    potential: np.ndarray,
    grid: Grid,
    *,
    dtau: float = 0.005,
    max_steps: int = 200_000,
    tol: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Ground state and its energy by imaginary-time relaxation.

    Converges when the energy changes by less than ``tol`` between checks.
    """
    half_v = np.exp(-0.5 * dtau * potential)
    kinetic = np.exp(-0.5 * dtau * grid.k**2)
    rng = np.random.default_rng(0)
    psi = rng.normal(size=grid.n) + 0.1  # bias away from odd-parity-only start
    psi = psi / grid.norm(psi)
    energy_prev = np.inf
    for step in range(max_steps):
        psi = half_v * psi
        psi = np.fft.ifft(kinetic * np.fft.fft(psi)).real
        psi = half_v * psi
        psi = psi / grid.norm(psi)
        if step % 200 == 199:
            energy = grid.energy(psi, potential)
            if abs(energy - energy_prev) < tol:
                return psi.astype(complex), energy
            energy_prev = energy
    raise RuntimeError("imaginary-time relaxation did not converge")
