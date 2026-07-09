"""README media: a wavepacket tunneling through a barrier (PNG frames + GIF)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from splitop import Grid, evolve, gaussian_packet, square_barrier

GRID = Grid(-200.0, 200.0, 2**13)
HEIGHT, WIDTH = 1.0, 1.0
K0 = np.sqrt(2.0 * 0.8 * HEIGHT)  # E = 0.8 V0: ~57% transmits, the rest reflects
SIGMA = 12.0


def main() -> None:
    psi = gaussian_packet(GRID, x0=-60.0, k0=K0, sigma=SIGMA)
    potential = square_barrier(GRID, HEIGHT, WIDTH)
    dt = 0.05
    steps_per_frame, n_frames = 20, 90

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.fill_between(GRID.x, 0, np.where(np.abs(GRID.x) <= WIDTH / 2, 0.05, 0.0),
                    color="gray", alpha=0.8, label="barrier (not to scale)")
    (line,) = ax.plot(GRID.x, np.abs(psi) ** 2, lw=1.2)
    ax.set_xlim(-120, 120)
    ax.set_ylim(0, 0.045)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|\psi|^2$")
    ax.set_title(rf"tunneling at $E = 0.8\,V_0$   ($k_0={K0:.2f}$)")
    ax.legend(loc="upper right", fontsize=9)

    state = {"psi": psi}

    def update(_frame):
        state["psi"] = evolve(state["psi"], potential, GRID, dt, steps_per_frame)
        line.set_ydata(np.abs(state["psi"]) ** 2)
        return (line,)

    anim = FuncAnimation(fig, update, frames=n_frames, blit=True)
    anim.save("docs/figures/tunneling.gif", writer=PillowWriter(fps=18))
    print("wrote docs/figures/tunneling.gif")

    transmitted = float(np.sum(np.abs(state["psi"][GRID.x > WIDTH]) ** 2) * GRID.dx)
    print(f"transmitted probability: {transmitted:.3f}")
    plt.close(fig)


if __name__ == "__main__":
    main()
