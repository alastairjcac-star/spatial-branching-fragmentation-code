"""Inspect two seeded paths: uv run --locked python examples/single_spine.py."""

import numpy as np

from spine_sim import ModelParameters, simulate_spine
from spine_sim.config import history_generator


def main() -> None:
    parameters = ModelParameters(mu=1, gamma0=1, alpha=0, initial_energy=1)
    times = np.linspace(0, 2, 9)
    for initial_type in ("electron", "photon"):
        rng = history_generator(20261014, initial_type, history_index=0)
        path = simulate_spine(parameters, times, initial_type, rng)
        print(f"Initial {initial_type}; demonstration path")
        print("track_length  type      log(E/E0)  log(W)")
        for time, particle_type, log_energy, log_weight in zip(
            path.times, path.types, path.log_energy, path.log_weight, strict=True
        ):
            print(
                f"{time:12.2f}  {particle_type:8s}  {log_energy:9.4f}  {log_weight:.4f}"
            )


if __name__ == "__main__":
    main()
