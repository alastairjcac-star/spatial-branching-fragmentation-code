"""Bounded threshold payoffs and independent-history Monte Carlo summaries."""

import math

import numpy as np
from numpy.typing import ArrayLike

from .model import PARTICLE_TYPES
from .simulator import SpinePath, observation_grid


def _thresholds(values: ArrayLike) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    if (
        result.ndim != 1
        or result.size == 0
        or np.any(~np.isfinite(result))
        or np.any(result <= 0)
        or np.any(result >= 1)
    ):
        raise ValueError(
            "thresholds must be a nonempty vector strictly between 0 and 1"
        )
    return result


def count_payoffs(path: SpinePath, thresholds: ArrayLike) -> np.ndarray:
    """Return theta*W/v * 1(v > theta, type=j), shaped (time, threshold, 2).

    Observed types are ordered electron, photon. The strict comparison and
    payoff are evaluated in log coordinates without reconstructing energy.
    Values are in [0, 1] for this nonnegative-loss model. Very small positive
    weights/payoffs can round to zero; no survival cutoff is introduced.
    """
    theta = _thresholds(thresholds)
    times = observation_grid(path.times)
    log_energy = np.asarray(path.log_energy, dtype=float)
    log_weight = np.asarray(path.log_weight, dtype=float)
    types = np.asarray(path.types)
    if any(values.shape != times.shape for values in (log_energy, log_weight, types)):
        raise ValueError("path state arrays must match its observation grid")
    if (
        np.any(~np.isfinite(log_energy))
        or np.any(~np.isfinite(log_weight))
        or np.any(log_energy > 0)
        or np.any(log_weight > 0)
        or np.any(~np.isin(types, PARTICLE_TYPES))
    ):
        raise ValueError("path must have finite nonpositive log states and valid types")
    log_theta = np.log(theta)
    above = log_energy[:, None] > log_theta[None, :]
    payoff = np.zeros(above.shape)
    time_index, threshold_index = np.nonzero(above)
    payoff[above] = np.exp(
        log_theta[threshold_index] + log_weight[time_index] - log_energy[time_index]
    )
    return payoff[:, :, None] * (types[:, None, None] == np.array(PARTICLE_TYPES))


class CountAccumulator:
    """Accumulate independent spines with Welford's stable sample variance.

    Curves and thresholds sharing a history are correlated. Marginal
    mean +/- 1.96*SE intervals are pointwise normal approximations, not
    simultaneous bands; all-zero samples do not establish zero expectation.
    Both raw-count means and standard errors divide normalized values by
    theta. Arrays of sums and squared sums are retained for auditability.
    """

    def __init__(self, observation_times: ArrayLike, thresholds: ArrayLike) -> None:
        self.observation_times = observation_grid(observation_times)
        self.thresholds = _thresholds(thresholds)
        shape = (self.observation_times.size, self.thresholds.size, 2)
        self.n = 0
        self.nonzero_count = np.zeros(shape, dtype=np.int64)
        self.sum_payoff = np.zeros(shape)
        self.sum_squared_payoff = np.zeros(shape)
        self.mean = np.zeros(shape)
        self.m2 = np.zeros(shape)

    def update(self, path: SpinePath) -> None:
        """Add one independent history sampled on this accumulator's grid."""
        if not np.array_equal(path.times, self.observation_times):
            raise ValueError("path observation grid differs from accumulator grid")
        values = count_payoffs(path, self.thresholds)
        self.n += 1
        self.nonzero_count += values > 0
        self.sum_payoff += values
        self.sum_squared_payoff += values * values
        delta = values - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (values - self.mean)

    def rows(self, initial_type: str) -> list[dict]:
        """Return serializable rows; at least two histories are required."""
        if initial_type not in PARTICLE_TYPES:
            raise ValueError(f"initial_type must be one of {PARTICLE_TYPES}")
        if self.n < 2:
            raise ValueError(
                "at least two independent histories are needed for standard errors"
            )
        rows = []
        for time_index, time in enumerate(self.observation_times):
            for threshold_index, theta in enumerate(self.thresholds):
                for type_index, observed_type in enumerate(PARTICLE_TYPES):
                    index = (time_index, threshold_index, type_index)
                    mean = float(self.mean[index])
                    variance = max(0.0, float(self.m2[index]) / (self.n - 1))
                    se = math.sqrt(variance / self.n)
                    raw_mean = mean / float(theta)
                    raw_se = se / float(theta)
                    nonzero = int(self.nonzero_count[index])
                    rows.append(
                        {
                            "initial_type": initial_type,
                            "observed_type": observed_type,
                            "track_length": float(time),
                            "threshold": float(theta),
                            "n": self.n,
                            "nonzero_count": nonzero,
                            "sum_payoff": float(self.sum_payoff[index]),
                            "sum_squared_payoff": float(self.sum_squared_payoff[index]),
                            "normalized_mean": mean,
                            "normalized_sample_variance": variance,
                            "normalized_se": se,
                            "raw_count_mean": raw_mean,
                            "raw_count_se": raw_se,
                            "normal_ci_low": raw_mean - 1.96 * raw_se,
                            "normal_ci_high": raw_mean + 1.96 * raw_se,
                            "all_zero": nonzero == 0,
                        }
                    )
        return rows
