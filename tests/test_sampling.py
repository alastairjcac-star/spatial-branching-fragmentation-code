"""Checks against analytic laws, not against a second copy of the sampler."""

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

from spine_sim.sampling import sample_log_loss, thinning_probability


def _decimal_thinning_probability(z):
    """Independent high-precision evaluation of the defining expression."""
    if z == 0:
        return 0.5
    with localcontext() as context:
        context.prec = 80
        value = Decimal(str(z))
        return float(1 / (1 - (-value).exp()) - 1 / value)


def test_thinning_probability_matches_high_precision_near_series_boundary():
    # Includes both sides of the documented z=0.01 series boundary.
    z = np.array(
        [
            0,
            1e-14,
            1e-10,
            1e-6,
            0.001,
            0.009999999,
            0.01,
            0.010000001,
            0.1,
            1,
            10,
            100,
            1e8,
        ]
    )
    expected = np.array([_decimal_thinning_probability(value) for value in z])
    actual = thinning_probability(z)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=8e-14)
    assert np.all(actual >= 0.5)
    assert np.all(actual < 1)
    assert thinning_probability(0.0) == 0.5


@pytest.mark.parametrize("h,mu", [(0.0, 1.0), (2.0, 0.0), (0.0, 0.0)])
def test_zero_log_loss_does_not_consume_randomness(h, mu):
    class NoRandomness:
        def __getattr__(self, name):
            raise AssertionError(f"zero increment unexpectedly called RNG.{name}")

    assert sample_log_loss(h, mu, NoRandomness()) == 0.0


@pytest.mark.parametrize("h,mu", [(-1, 1), (1, -1), (math.nan, 1), (1, math.inf)])
def test_invalid_increment_parameters_are_rejected(h, mu):
    with pytest.raises(ValueError):
        sample_log_loss(h, mu, np.random.default_rng(41))


def _laplace_transform(s, h, mu):
    # Integrating (1-exp(-s*z))*mu*sum_{k=2}^infinity exp(-k*z)
    # gives mu * sum_{k=2}^{s+1} 1/k for integer s >= 1.
    return math.exp(-mu * h * sum(1 / k for k in range(2, s + 2)))


def test_log_loss_laplace_transform_matches_independent_integral():
    rng = np.random.default_rng(28405)
    h, mu, n = 0.7, 1.3, 12000
    increments = np.array([sample_log_loss(h, mu, rng) for _ in range(n)])
    assert np.all(np.isfinite(increments))
    assert np.all(increments >= 0)
    for s in (1, 2, 4):
        expected = _laplace_transform(s, h, mu)
        population_variance = _laplace_transform(2 * s, h, mu) - expected**2
        standard_error = math.sqrt(population_variance / n)
        assert abs(np.mean(np.exp(-s * increments)) - expected) < 6 * standard_error
