"""splitop — split-operator Schrödinger solver, validated against analytic QM."""

from .core import Grid, evolve, gaussian_packet, imaginary_time_ground_state
from .potentials import (
    barrier_transmission,
    coherent_center,
    double_well,
    free_packet_width,
    harmonic,
    harmonic_ground_energy,
    square_barrier,
)

__all__ = [
    "Grid",
    "barrier_transmission",
    "coherent_center",
    "double_well",
    "evolve",
    "free_packet_width",
    "gaussian_packet",
    "harmonic",
    "harmonic_ground_energy",
    "imaginary_time_ground_state",
    "square_barrier",
]

__version__ = "0.1.0"
