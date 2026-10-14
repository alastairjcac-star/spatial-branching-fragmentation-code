"""Event semantics and independent finite-state-chain validation."""

import math

import numpy as np
import pytest

from spine_sim import simulator
from spine_sim.model import ModelParameters
from spine_sim.simulator import simulate_spine


class ScriptedRNG:
    """Exact clock/fraction draws to expose observation-loop mistakes."""

    def __init__(self, waiting_times, uniforms):
        self.waiting_times = iter(waiting_times)
        self.uniforms = iter(uniforms)
        self.exponential_scales = []
        self.uniform_calls = 0

    def exponential(self, scale=1.0):
        self.exponential_scales.append(scale)
        return next(self.waiting_times)

    def random(self):
        self.uniform_calls += 1
        return next(self.uniforms)


def test_observations_retain_clocks_and_switches_precede_tied_observations(monkeypatch):
    intervals = []

    def deterministic_loss(h, mu, rng):
        intervals.append((h, mu))
        return 0.4 * h

    monkeypatch.setattr(simulator, "sample_log_loss", deterministic_loss)
    parameters = ModelParameters(mu=2, gamma0=4, alpha=0.3)
    times = np.array([0, 0.25, 0.75, 1.0, 1.25, 1.25, 1.5])
    rng = ScriptedRNG([0.75, 0.5, 10.0], [0.25, 0.36])
    path = simulate_spine(parameters, times, "electron", rng)

    # A switch scheduled for .75 is retained across the .25 observation.
    # The switch back at 1.25 occurs before both observations at that time.
    np.testing.assert_array_equal(
        path.types,
        [
            "electron",
            "electron",
            "photon",
            "photon",
            "electron",
            "electron",
            "electron",
        ],
    )
    electron_length = np.array([0, 0.25, 0.75, 0.75, 0.75, 0.75, 1.0])
    switch_log_loss = np.array(
        [
            0,
            0,
            math.log(0.25),
            math.log(0.25),
            math.log(0.25) + math.log(0.6),
            math.log(0.25) + math.log(0.6),
            math.log(0.25) + math.log(0.6),
        ]
    )
    np.testing.assert_allclose(
        path.log_energy, -0.7 * electron_length + switch_log_loss
    )
    np.testing.assert_allclose(path.log_weight, -0.3 * electron_length)
    np.testing.assert_allclose(path.electron_track_length, electron_length)
    np.testing.assert_array_equal(path.times, times)
    assert path.switch_count == 2
    assert rng.exponential_scales == [0.5, 0.25, 0.5]
    assert rng.uniform_calls == 2
    np.testing.assert_allclose([h for h, _ in intervals if h > 0], [0.25, 0.5, 0.25])
    assert all(mu == 2 for _, mu in intervals)


@pytest.mark.parametrize("initial_type,exponent", [("electron", 1), ("photon", 0.5)])
def test_switch_uses_correct_energy_biased_fraction_and_rejects_zero_uniform(
    monkeypatch, initial_type, exponent
):
    monkeypatch.setattr(simulator, "sample_log_loss", lambda h, mu, rng: 0.0)
    rng = ScriptedRNG([0.25, 10.0], [0.0, 0.04])
    path = simulate_spine(ModelParameters(), [0, 0.25], initial_type, rng)
    assert path.log_energy[-1] == pytest.approx(exponent * math.log(0.04))
    assert path.types[-1] != initial_type
    assert path.switch_count == 1
    assert rng.uniform_calls == 2


@pytest.mark.parametrize("initial_type", ["electron", "photon"])
def test_zero_rates_give_deterministic_continuous_loss(initial_type):
    parameters = ModelParameters(mu=0, gamma0=0, alpha=0.7)
    times = np.array([0, 0.5, 3.0])
    path = simulate_spine(parameters, times, initial_type, np.random.default_rng(58))
    expected_length = times if initial_type == "electron" else np.zeros_like(times)
    np.testing.assert_allclose(path.log_energy, -0.7 * expected_length)
    np.testing.assert_allclose(path.log_weight, -0.7 * expected_length)
    np.testing.assert_allclose(path.electron_track_length, expected_length)
    assert np.all(path.types == initial_type)
    assert path.switch_count == 0


@pytest.mark.parametrize("initial_type", ["electron", "photon"])
def test_type_probabilities_match_two_state_forward_equation(initial_type):
    parameters = ModelParameters(mu=0.8, gamma0=1.4, alpha=0.2)
    horizon, n = 1.3, 5000
    rng = np.random.default_rng(8402 if initial_type == "electron" else 8403)
    electron_count = sum(
        simulate_spine(parameters, [horizon], initial_type, rng).types[-1] == "electron"
        for _ in range(n)
    )
    rate_sum = parameters.mu + parameters.gamma0
    equilibrium = parameters.gamma0 / rate_sum
    expected = equilibrium + ((initial_type == "electron") - equilibrium) * math.exp(
        -rate_sum * horizon
    )
    standard_error = math.sqrt(expected * (1 - expected) / n)
    assert abs(electron_count / n - expected) < 6 * standard_error


def test_path_invariants_weight_identity_and_reproducibility():
    parameters = ModelParameters(mu=1.2, gamma0=0.8, alpha=0.35)
    times = np.linspace(0, 6, 61)
    first = simulate_spine(parameters, times, "electron", np.random.default_rng(990))
    second = simulate_spine(parameters, times, "electron", np.random.default_rng(990))
    for field in (
        "times",
        "log_energy",
        "log_weight",
        "types",
        "electron_track_length",
    ):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    assert first.switch_count == second.switch_count
    assert np.all(np.isfinite(first.log_energy))
    assert np.all(np.diff(first.log_energy) <= 0)
    assert np.all(first.log_energy <= 0)
    assert np.all(first.electron_track_length >= 0)
    assert np.all(first.electron_track_length <= times + 1e-14)
    np.testing.assert_allclose(
        first.log_weight, -parameters.alpha * first.electron_track_length
    )


def test_no_loss_weight_and_initial_energy_normalisation():
    times = [0, 1, 2]
    first = simulate_spine(
        ModelParameters(initial_energy=1), times, "photon", np.random.default_rng(77)
    )
    second = simulate_spine(
        ModelParameters(initial_energy=25), times, "photon", np.random.default_rng(77)
    )
    np.testing.assert_array_equal(first.log_weight, np.zeros(3))
    np.testing.assert_array_equal(first.log_energy, second.log_energy)
    np.testing.assert_array_equal(first.types, second.types)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mu": -1},
        {"gamma0": math.inf},
        {"alpha": math.nan},
        {"initial_energy": 0},
        {"initial_energy": -1},
    ],
)
def test_model_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        ModelParameters(**kwargs)
