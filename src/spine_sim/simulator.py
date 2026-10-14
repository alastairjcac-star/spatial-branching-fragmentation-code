"""Sample a spine on a finite observation grid using persistent clocks."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .model import PARTICLE_TYPES, ModelParameters
from .sampling import open_uniform, positive_exponential, sample_log_loss


@dataclass(frozen=True)
class SpinePath:
    """Finite observations of one spine, not a full branching population.

    ``log_energy`` is log(energy / initial_energy). Types use ``electron``
    and ``photon``. ``electron_track_length`` records cumulative time in
    electron states, and ``switch_count`` includes switches at the horizon.
    """

    times: np.ndarray
    log_energy: np.ndarray
    log_weight: np.ndarray
    types: np.ndarray
    electron_track_length: np.ndarray
    switch_count: int


def observation_grid(values: ArrayLike) -> np.ndarray:
    """Validate a nonempty, finite, nonnegative, nondecreasing grid."""
    result = np.array(values, dtype=float, copy=True)
    if (
        result.ndim != 1
        or result.size == 0
        or np.any(~np.isfinite(result))
        or np.any(result < 0)
        or np.any(np.diff(result) < 0)
    ):
        raise ValueError(
            "observation times must be finite, nonnegative and nondecreasing"
        )
    return result


def _next_switch(time: float, rate: float, rng: np.random.Generator) -> float:
    if rate == 0:
        return math.inf
    waiting = positive_exponential(1 / rate, rng)
    result = time + waiting
    if not math.isfinite(result) or result <= time:
        raise FloatingPointError(
            "switch clock cannot advance in floating-point arithmetic"
        )
    return result


def simulate_spine(
    parameters: ModelParameters,
    observation_times: ArrayLike,
    initial_type: str,
    rng: np.random.Generator,
) -> SpinePath:
    """Sample this homogeneous model at the requested track lengths.

    A pending type-switch clock survives observations. At coincident times,
    switches precede observations (right-continuous convention). Repeated
    observations consume no additional randomness. Electron increments are
    sampled without a jump cutoff or numerical time step. This statement
    describes finite-dimensional sampling in ideal arithmetic; finite
    precision and random-variate implementations impose their usual limits.
    """
    if initial_type not in PARTICLE_TYPES:
        raise ValueError(f"initial_type must be one of {PARTICLE_TYPES}")
    times = observation_grid(observation_times)
    log_energy = np.empty(times.size)
    log_weight = np.empty(times.size)
    types = np.empty(times.size, dtype="U8")
    electron_lengths = np.empty(times.size)
    current_type = initial_type
    current_time = 0.0
    current_energy = 0.0
    current_weight = 0.0
    electron_length = 0.0
    switch_count = 0
    rate = parameters.mu if current_type == "electron" else parameters.gamma0
    switch_time = _next_switch(current_time, rate, rng)

    for index, target in enumerate(times):
        while True:
            endpoint = min(switch_time, float(target))
            h = endpoint - current_time
            if h < 0:
                raise FloatingPointError("simulation clock moved backwards")
            if current_type == "electron" and h > 0:
                loss = sample_log_loss(h, parameters.mu, rng)
                drift = parameters.alpha * h
                current_energy -= loss + drift
                current_weight -= drift
                electron_length += h
                if not all(
                    math.isfinite(value)
                    for value in (current_energy, current_weight, electron_length)
                ):
                    raise FloatingPointError("electron state accumulation overflowed")
            current_time = endpoint
            if switch_time <= target:
                fraction_log = math.log(open_uniform(rng))
                if current_type == "electron":
                    current_energy += fraction_log
                    current_type = "photon"
                    rate = parameters.gamma0
                else:
                    current_energy += 0.5 * fraction_log
                    current_type = "electron"
                    rate = parameters.mu
                if not math.isfinite(current_energy):
                    raise FloatingPointError("log energy overflowed at a type switch")
                switch_count += 1
                switch_time = _next_switch(current_time, rate, rng)
                continue
            break
        log_energy[index] = current_energy
        log_weight[index] = current_weight
        types[index] = current_type
        electron_lengths[index] = electron_length

    return SpinePath(
        times, log_energy, log_weight, types, electron_lengths, switch_count
    )
