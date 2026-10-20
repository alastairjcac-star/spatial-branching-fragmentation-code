"""Simulate one homogeneous electron–photon spine using standard-library Python.

Run: python3 single_spine.py
Edit the settings at the bottom to change the initial type or observation times.

Energy is normalised by its initial value: v = energy / initial_energy.
The reference parameters are mu = gamma0 = 1 and alpha = 0, so W = 1.
Only energy and type are simulated; there is no spatial motion or killing.
For serial Monte Carlo, create one random.Random(seed) outside the history loop
and pass it to each call as rng=rng. Each call starts a new spine and consumes
fresh draws from the same generator.
"""

import math
import random


def thinning_probability(z):
    """Evaluate B(z) = 1/(1-exp(-z)) - 1/z without cancellation near zero."""
    if z < 0.01:
        return 0.5 + z / 12 - z**3 / 720 + z**5 / 30240
    return -1 / math.expm1(-z) - 1 / z


def next_event_time(current_time, rate, rng):
    """Schedule an exponential clock; a zero rate means no event."""
    if rate == 0:
        return math.inf
    waiting_time = rng.expovariate(rate)
    while waiting_time == 0:
        waiting_time = rng.expovariate(rate)
    event_time = current_time + waiting_time
    if event_time <= current_time:
        raise FloatingPointError("The event time cannot advance at this precision.")
    return event_time


def sample_log_loss(h, mu, rng):
    """Sample S_h: a gamma increment plus a thinned finite Poisson process."""
    if h == 0 or mu == 0:
        return 0.0

    # gammavariate uses shape and SCALE: mathematical rate 2 means scale 1/2.
    loss = rng.gammavariate(mu * h, 0.5)

    # Exponential spacings generate Poisson(mu*h/2) candidate jumps on (0,h].
    candidate_time = next_event_time(0.0, mu / 2, rng)
    while candidate_time <= h:
        z = rng.expovariate(2.0)
        if rng.random() < thinning_probability(z):
            loss += z
        candidate_time = next_event_time(candidate_time, mu / 2, rng)
    return loss


def simulate_spine(
    observation_times,
    initial_type="electron",
    mu=1.0,
    gamma0=1.0,
    alpha=0.0,
    seed=None,
    *,
    rng=None,
):
    """Return (track_length, particle_type, log_v, log_W) at each observation.

    Times must be finite, nonnegative and nondecreasing. Repeated times are
    allowed. Rates are per unit track length. The weight integrates out the
    additional killing clock: W = exp(-alpha * total_electron_track_length).

    """
    times = [float(t) for t in observation_times]
    if not times or any(not math.isfinite(t) or t < 0 for t in times):
        raise ValueError("Provide finite, nonnegative observation times.")
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("Observation times must be in nondecreasing order.")
    if initial_type not in ("electron", "photon"):
        raise ValueError("initial_type must be 'electron' or 'photon'.")
    if any(not math.isfinite(rate) or rate < 0 for rate in (mu, gamma0, alpha)):
        raise ValueError("mu, gamma0 and alpha must be finite and nonnegative.")

    if rng is not None and seed is not None:
        raise ValueError("Supply either rng or seed, not both.")
    if rng is None:
        rng = random.Random(seed)

    # Reset the particle state for each history, but keep a supplied RNG's state.
    track_length = 0.0
    log_v = 0.0
    log_W = 0.0
    particle_type = initial_type
    rate = mu if particle_type == "electron" else gamma0
    switch_time = next_event_time(track_length, rate, rng)
    observations = []

    for output_time in times:
        while track_length < output_time:
            end = min(switch_time, output_time)
            h = end - track_length

            if particle_type == "electron":
                log_v -= sample_log_loss(h, mu, rng) + alpha * h
                log_W -= alpha * h
            track_length = end

            # Process a switch before observing its time (right continuity).
            if track_length == switch_time:
                if particle_type == "electron":
                    # -log(U) ~ Exp(1): new energy is U times the old energy.
                    log_v -= rng.expovariate(1.0)
                    particle_type = "photon"
                    rate = gamma0
                else:
                    # -log(sqrt(U)) ~ Exp(2): retained fraction has density 2v.
                    log_v -= rng.expovariate(2.0)
                    particle_type = "electron"
                    rate = mu
                switch_time = next_event_time(track_length, rate, rng)

            if not math.isfinite(log_v) or not math.isfinite(log_W):
                raise FloatingPointError("Energy or weight exceeded numerical range.")

        # Observation does not replace the previously scheduled switching time.
        observations.append((output_time, particle_type, log_v, log_W))

    return observations


if __name__ == "__main__":
    # Settings for ONE realisation. Observation spacing is not an integration step.
    path = simulate_spine(
        observation_times=[i / 4 for i in range(25)],  # 0, 0.25, ..., 6
        initial_type="electron",  # Change to "photon" for a photon initially.
        mu=1.0,
        gamma0=1.0,
        alpha=0.0,
        seed=20261015,
    )

    print(f"{'ell':>6}  {'type':>8}  {'v = E/E0':>12}  {'W':>12}")
    for ell, particle_type, log_v, log_W in path:
        print(
            f"{ell:6.2f}  {particle_type:>8}  "
            f"{math.exp(log_v):12.6g}  {math.exp(log_W):12.6g}"
        )
