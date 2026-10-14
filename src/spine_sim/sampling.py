"""Gamma plus thinned-Poisson sampling, with no small-jump cutoff."""

import math

import numpy as np
from numpy.typing import ArrayLike


def thinning_probability(z: ArrayLike) -> float | np.ndarray:
    """Return B(z) = 1 / (1 - exp(-z)) - 1 / z, with B(0) = 1/2.

    Below 0.01, a Bernoulli-number series through z**7 avoids cancellation.
    Elsewhere ``expm1`` evaluates the exponential difference stably. Inputs
    must be finite and nonnegative; the zero value is the analytic limit.
    """
    values = np.asarray(z, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("z must be finite and nonnegative")
    result = np.empty_like(values)
    small = values < 0.01
    x = values[small]
    x2 = x * x
    result[small] = 0.5 + x * (
        1 / 12 + x2 * (-1 / 720 + x2 * (1 / 30240 - x2 / 1209600))
    )
    x = values[~small]
    result[~small] = -1 / np.expm1(-x) - 1 / x
    return float(result) if result.ndim == 0 else result


def open_uniform(rng: np.random.Generator) -> float:
    """Draw a uniform in (0, 1), resampling a rounded endpoint.

    NumPy draws from [0, 1), so zero is the relevant endpoint. A finite
    retry bound also prevents an invalid/custom generator from hanging.
    """
    for _ in range(1024):
        value = float(rng.random())
        if 0 < value < 1:
            return value
        if not math.isfinite(value) or value < 0 or value > 1:
            raise FloatingPointError("uniform generator returned an invalid value")
    raise FloatingPointError("could not draw a uniform strictly inside (0, 1)")


def positive_exponential(scale: float, rng: np.random.Generator) -> float:
    """Draw a positive finite exponential, resampling rounded zeros."""
    if not math.isfinite(scale) or scale <= 0:
        raise FloatingPointError("exponential scale is not positive and finite")
    for _ in range(1024):
        value = float(rng.exponential(scale=scale))
        if math.isfinite(value) and value > 0:
            return value
        if not math.isfinite(value) or value < 0:
            raise FloatingPointError("exponential draw is not finite and nonnegative")
    raise FloatingPointError("could not draw a positive exponential")


def sample_log_loss(h: float, mu: float, rng: np.random.Generator) -> float:
    """Sample the electron subordinator increment over track length ``h``.

    The independent gamma part has shape ``mu*h`` and scale 1/2. Candidate
    jumps have count Poisson(mu*h/2), size Exponential(scale=1/2), and are
    accepted with probability B(size). Zero h or mu consumes no randomness.
    Candidate arrays are batched solely to bound memory, not to approximate
    the process. Floating-point random-variate limitations still apply.
    """
    if not math.isfinite(h) or not math.isfinite(mu) or h < 0 or mu < 0:
        raise ValueError("h and mu must be finite and nonnegative")
    if h == 0 or mu == 0:
        return 0.0
    shape = mu * h
    if not math.isfinite(shape) or shape == 0:
        raise FloatingPointError("mu*h overflowed or underflowed")
    total = float(rng.gamma(shape=shape, scale=0.5))
    if not math.isfinite(total) or total < 0:
        raise FloatingPointError("gamma draw is not finite and nonnegative")
    remaining = int(rng.poisson(lam=shape / 2))
    while remaining:
        count = min(remaining, 65536)
        jumps = np.asarray(rng.exponential(scale=0.5, size=count), dtype=float)
        # Exponential zero is a floating-point endpoint, not a physical jump.
        zero = jumps == 0
        while np.any(zero):
            jumps[zero] = [
                positive_exponential(0.5, rng) for _ in range(int(zero.sum()))
            ]
            zero = jumps == 0
        if np.any(~np.isfinite(jumps)) or np.any(jumps < 0):
            raise FloatingPointError("candidate jump is not finite and nonnegative")
        uniforms = rng.random(size=count)
        total += float(np.sum(jumps[uniforms < thinning_probability(jumps)]))
        if not math.isfinite(total):
            raise FloatingPointError("accumulated log loss overflowed")
        remaining -= count
    return total
