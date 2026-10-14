"""Threshold conventions, bounded log-space scoring and sampling uncertainty."""

import math

import numpy as np
import pytest

from spine_sim.estimation import CountAccumulator, count_payoffs
from spine_sim.model import ModelParameters
from spine_sim.simulator import SpinePath, simulate_spine


def _path(times, energies, weights, types):
    times = np.asarray(times, dtype=float)
    return SpinePath(
        times=times,
        log_energy=np.log(np.asarray(energies, dtype=float)),
        log_weight=np.log(np.asarray(weights, dtype=float)),
        types=np.asarray(types),
        electron_track_length=np.zeros_like(times),
        switch_count=0,
    )


def test_strict_threshold_and_initial_count_are_scored_for_correct_type():
    path = _path(
        [0, 1, 2], [1, 0.25, 0.125], [1, 0.5, 0.5], ["electron", "photon", "electron"]
    )
    payoffs = count_payoffs(path, [0.125, 0.25])
    expected = np.array(
        [
            [[0.125, 0], [0.25, 0]],
            [[0, 0.25], [0, 0]],
            [[0, 0], [0, 0]],
        ]
    )
    np.testing.assert_allclose(payoffs, expected, rtol=1e-14, atol=0)
    assert np.all((payoffs >= 0) & (payoffs <= 1))


def test_below_threshold_does_not_overflow_inverse_energy():
    path = SpinePath(
        times=np.array([1.0]),
        log_energy=np.array([-1000.0]),
        log_weight=np.array([0.0]),
        types=np.array(["electron"]),
        electron_track_length=np.array([1.0]),
        switch_count=0,
    )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        payoffs = count_payoffs(path, [0.01])
    np.testing.assert_array_equal(payoffs, np.zeros((1, 1, 2)))


def test_accumulator_matches_hand_computed_variance_and_raw_count_scaling():
    accumulator = CountAccumulator([1.0], [0.25])
    # Three electron payoffs: 1/4, 1/2, 0.  Mean=1/4, sample variance=1/16.
    for energy in (1.0, 0.5, 0.125):
        accumulator.update(_path([1.0], [energy], [1.0], ["electron"]))
    rows = accumulator.rows("electron")
    by_type = {row["observed_type"]: row for row in rows}
    row = by_type["electron"]
    assert row["n"] == 3
    assert row["nonzero_count"] == 2
    assert row["sum_payoff"] == pytest.approx(0.75)
    assert row["sum_squared_payoff"] == pytest.approx(0.3125)
    assert row["normalized_mean"] == pytest.approx(0.25)
    assert row["normalized_sample_variance"] == pytest.approx(0.0625)
    assert row["normalized_se"] == pytest.approx(0.25 / math.sqrt(3))
    assert row["raw_count_mean"] == pytest.approx(1.0)
    assert row["raw_count_se"] == pytest.approx(1 / math.sqrt(3))
    assert row["normal_ci_low"] == pytest.approx(1 - 1.96 / math.sqrt(3))
    assert row["normal_ci_high"] == pytest.approx(1 + 1.96 / math.sqrt(3))
    assert not row["all_zero"]
    zero_row = by_type["photon"]
    assert zero_row["all_zero"]
    assert zero_row["nonzero_count"] == 0
    assert zero_row["raw_count_mean"] == 0
    assert zero_row["raw_count_se"] == 0


def test_uncertainty_requires_two_independent_histories():
    accumulator = CountAccumulator([0.0], [0.1])
    accumulator.update(_path([0.0], [1.0], [1.0], ["electron"]))
    with pytest.raises(ValueError):
        accumulator.rows("electron")


def test_no_electron_fragmentation_limit_matches_one_split_cascade_counts():
    # With mu=alpha=0 the original population has at most one event: a
    # photon splits into U and 1-U at rate gamma0. Either child exceeds
    # theta with probability 1-theta, giving the count reference below.
    theta, horizon, gamma0, n = 0.3, 0.8, 1.5, 6000
    parameters = ModelParameters(mu=0, gamma0=gamma0, alpha=0)
    accumulator = CountAccumulator([horizon], [theta])
    rng = np.random.default_rng(99382)
    for _ in range(n):
        accumulator.update(simulate_spine(parameters, [horizon], "photon", rng))
    rows = {row["observed_type"]: row for row in accumulator.rows("photon")}
    survival_probability = math.exp(-gamma0 * horizon)
    split_probability = 1 - survival_probability
    electron_mean = 2 * theta * (1 - theta) * split_probability
    # The energy-biased child's density is 2v, so its squared normalized
    # payoff integrates to -2*theta**2*log(theta), conditional on a split.
    electron_second_moment = -2 * theta**2 * math.log(theta) * split_probability
    expected = {"electron": electron_mean, "photon": theta * survival_probability}
    variance = {
        "electron": electron_second_moment - electron_mean**2,
        "photon": theta**2 * survival_probability * split_probability,
    }
    for observed_type in ("electron", "photon"):
        standard_error = math.sqrt(variance[observed_type] / n)
        assert (
            abs(rows[observed_type]["normalized_mean"] - expected[observed_type])
            < 6 * standard_error
        )
        assert rows[observed_type]["raw_count_mean"] == pytest.approx(
            rows[observed_type]["normalized_mean"] / theta
        )
