"""Every test here compares the numerics against closed-form quantum mechanics."""

import numpy as np
import pytest

from splitop import (
    Grid,
    barrier_transmission,
    coherent_center,
    evolve,
    free_packet_width,
    gaussian_packet,
    harmonic,
    harmonic_ground_energy,
    imaginary_time_ground_state,
    square_barrier,
)


@pytest.fixture
def grid():
    return Grid(-40.0, 40.0, 2048)


def test_norm_conserved_to_machine_precision(grid):
    """Strang splitting is unitary: 2,000 steps must not leak norm."""
    psi = gaussian_packet(grid, x0=-5.0, k0=2.0, sigma=1.0)
    potential = harmonic(grid, omega=0.7)
    final = evolve(psi, potential, grid, dt=0.005, n_steps=2000)
    assert grid.norm(final) == pytest.approx(1.0, abs=1e-12)


def test_energy_conserved(grid):
    psi = gaussian_packet(grid, x0=-4.0, k0=1.5, sigma=1.2)
    potential = harmonic(grid, omega=0.5)
    e0 = grid.energy(psi, potential)
    final = evolve(psi, potential, grid, dt=0.002, n_steps=5000)
    assert grid.energy(final, potential) == pytest.approx(e0, rel=1e-8)


def test_coherent_state_follows_classical_trajectory(grid):
    """A displaced HO ground state oscillates as x(t) = x0 cos(w t) exactly."""
    omega, x0 = 1.0, 5.0
    sigma = 1.0 / np.sqrt(2.0 * omega)  # ground-state width -> coherent state
    psi = gaussian_packet(grid, x0=x0, k0=0.0, sigma=sigma)
    potential = harmonic(grid, omega=omega)

    dt, n_steps = 0.002, 4000  # total t = 8.0
    centers, times = [], []

    def record(step, current):
        times.append((step + 1) * dt)
        centers.append(grid.expectation_x(current))

    evolve(psi, potential, grid, dt, n_steps, callback=record, callback_every=100)
    analytic = [coherent_center(t, x0, omega) for t in times]
    assert np.allclose(centers, analytic, atol=5e-3)


def test_free_packet_spreads_analytically(grid):
    """<free particle> Gaussian width sigma(t) has a closed form."""
    sigma0 = 1.5
    psi = gaussian_packet(grid, x0=0.0, k0=0.0, sigma=sigma0)
    potential = np.zeros(grid.n)
    dt, n_steps = 0.01, 1000  # t = 10
    final = evolve(psi, potential, grid, dt, n_steps)
    expected = free_packet_width(10.0, sigma0)
    assert grid.width_x(final) == pytest.approx(expected, rel=1e-4)


def test_imaginary_time_finds_harmonic_ground_state(grid):
    omega = 1.3
    potential = harmonic(grid, omega=omega)
    psi, energy = imaginary_time_ground_state(potential, grid, dtau=0.002)
    assert energy == pytest.approx(harmonic_ground_energy(omega), abs=1e-4)
    # ground state is a Gaussian of width 1/sqrt(2 omega)
    assert grid.width_x(psi) == pytest.approx(1.0 / np.sqrt(2.0 * omega), rel=1e-3)


@pytest.mark.parametrize("energy_ratio", [0.6, 1.5])
def test_barrier_transmission_matches_plane_wave_formula(energy_ratio):
    """Send a narrow-momentum packet at a thin barrier; the transmitted
    probability must approach the analytic plane-wave T(E)."""
    grid = Grid(-400.0, 400.0, 2**14)
    height, width = 1.0, 1.0
    k0 = np.sqrt(2.0 * energy_ratio * height)
    sigma = 30.0  # narrow momentum spread: dk = 1/(2 sigma) << k0
    psi = gaussian_packet(grid, x0=-120.0, k0=k0, sigma=sigma)
    potential = square_barrier(grid, height, width)

    time_total = 240.0 / k0  # packet reaches and clears the barrier
    n_steps = 3000
    final = evolve(psi, potential, grid, time_total / n_steps, n_steps)

    transmitted = float(np.sum(np.abs(final[grid.x > width]) ** 2) * grid.dx)
    analytic = barrier_transmission(energy_ratio * height, height, width)
    assert transmitted == pytest.approx(analytic, abs=0.02)


def test_transmission_limits():
    assert barrier_transmission(1e-6, 1.0, 1.0) < 1e-4
    assert barrier_transmission(50.0, 1.0, 1.0) > 0.99
    with pytest.raises(ValueError):
        barrier_transmission(-1.0, 1.0, 1.0)


def test_grid_validation():
    with pytest.raises(ValueError):
        Grid(-1.0, 1.0, 100)  # not a power of two
    with pytest.raises(ValueError):
        Grid(1.0, -1.0, 128)


def test_packet_mean_momentum(grid):
    psi = gaussian_packet(grid, x0=0.0, k0=3.0, sigma=2.0)
    assert grid.expectation_p(psi) == pytest.approx(3.0, abs=1e-6)
