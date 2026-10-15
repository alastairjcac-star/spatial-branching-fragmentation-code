"""Estimate expected particle counts from independent spine histories.

Run: python3 monte_carlo.py
Edit the experiment settings at the bottom of this file.
The example runs both initial types and three thresholds at lengths 0, 0.1, ..., 6.
Each ensemble reuses its paths for all thresholds and both counted types.
Running the script saves counts.csv and counts_settings.json beside this file.
These files are replaced on each run; change their names below to keep a run.

For each observed type j, average X = (W/v) * 1{v > theta, type = j}.
Its mean estimates C_theta,j / theta. The reported standard error describes
Monte Carlo uncertainty in that mean, not fluctuations of the cascade count.
Only Python's standard library and single_spine.py are required.
"""

import csv
import json
import math
import platform
import random
from pathlib import Path

from single_spine import simulate_spine


def estimate_counts_for_thresholds(
    observation_times,
    n=10_000,
    thresholds=(0.1, 0.03, 0.01),
    initial_type="electron",
    mu=1.0,
    gamma0=1.0,
    alpha=0.0,
    seed=20261015,
    *,
    return_nonzero=False,
):
    """Return a dictionary mapping each threshold to count-estimate rows.

    Each row is (ell, electron_mean, electron_se, photon_mean, photon_se).
    n is the number of spine histories for the chosen initial type. All
    thresholds, counted types and observation lengths use those histories,
    so their estimates are correlated. Standard errors are pointwise.
    Thresholds must be distinct; dictionary order follows the supplied order.
    A zero standard error from all-zero contributions can miss rare events.

    With return_nonzero=True, return (results, nonzero_counts). For each
    threshold, nonzero_counts contains [electron_count, photon_count] pairs
    in observation order. They count contributing histories, not particles.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer of at least 2 to estimate uncertainty.")
    thresholds = list(thresholds)
    if not thresholds or any(
        not math.isfinite(theta) or not 0 < theta < 1 for theta in thresholds
    ):
        raise ValueError("Provide finite thresholds strictly between 0 and 1.")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Thresholds must be distinct.")

    times = [float(t) for t in observation_times]
    log_thresholds = {theta: math.log(theta) for theta in thresholds}
    smallest_log_threshold = min(log_thresholds.values())
    rng = random.Random(seed)  # Initialise ONCE, outside the history loop.

    # Each threshold has its own mean and variance at each observation length.
    # Column 0 is electron; column 1 is photon.
    means = {theta: [[0.0, 0.0] for _ in times] for theta in thresholds}
    m2 = {theta: [[0.0, 0.0] for _ in times] for theta in thresholds}
    nonzero_counts = {theta: [[0, 0] for _ in times] for theta in thresholds}

    for history_number in range(1, n + 1):
        path = simulate_spine(
            observation_times=times,
            initial_type=initial_type,
            mu=mu,
            gamma0=gamma0,
            alpha=alpha,
            rng=rng,
        )

        for i, (_, particle_type, log_v, log_W) in enumerate(path):
            type_index = 0 if particle_type == "electron" else 1

            # Evaluate W/v only when this observation passes some threshold.
            weight = 0.0
            if log_v > smallest_log_threshold:
                weight = math.exp(log_W - log_v)

            for theta, log_theta in log_thresholds.items():
                contributions = [0.0, 0.0]
                if log_v > log_theta:  # The energy threshold is strict.
                    contributions[type_index] = weight
                    if weight != 0.0:
                        nonzero_counts[theta][i][type_index] += 1

                # Welford's update for this threshold, including zero values.
                for j in (0, 1):
                    difference = contributions[j] - means[theta][i][j]
                    means[theta][i][j] += difference / history_number
                    m2[theta][i][j] += difference * (
                        contributions[j] - means[theta][i][j]
                    )

    results = {}
    for theta in thresholds:
        rows = []
        for i, ell in enumerate(times):
            # Sample variance = M2/(n-1); standard error = sqrt(variance/n).
            # max(..., 0) protects against a tiny negative roundoff error.
            electron_se = math.sqrt(max(m2[theta][i][0], 0.0) / (n - 1) / n)
            photon_se = math.sqrt(max(m2[theta][i][1], 0.0) / (n - 1) / n)
            rows.append(
                (ell, means[theta][i][0], electron_se, means[theta][i][1], photon_se)
            )
        results[theta] = rows

    if return_nonzero:
        return results, nonzero_counts
    return results


def estimate_counts(
    observation_times,
    n=10_000,
    theta=0.1,
    initial_type="electron",
    mu=1.0,
    gamma0=1.0,
    alpha=0.0,
    seed=20261015,
):
    """Estimate counts for one threshold, preserving the original list of rows."""
    results = estimate_counts_for_thresholds(
        observation_times=observation_times,
        n=n,
        thresholds=(theta,),
        initial_type=initial_type,
        mu=mu,
        gamma0=gamma0,
        alpha=alpha,
        seed=seed,
    )
    return results[theta]


def save_counts_csv(filename, results_by_initial_type, nonzero_by_initial_type):
    """Save one row per initial type, threshold, observation and counted type.

    Estimated counts already equal C_theta,j / theta: no further scaling is
    applied. Floats are saved at full Python precision, unlike the printout.
    """
    fieldnames = (
        "initial_type",
        "threshold",
        "track_length",
        "counted_type",
        "estimated_count",
        "standard_error",
        "nonzero_contributions",
    )
    with open(filename, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for initial_type, results in results_by_initial_type.items():
            for theta, rows in results.items():
                for i, row in enumerate(rows):
                    ell, electron_mean, electron_se, photon_mean, photon_se = row
                    estimates = (
                        ("electron", electron_mean, electron_se),
                        ("photon", photon_mean, photon_se),
                    )
                    for j, (counted_type, mean, se) in enumerate(estimates):
                        writer.writerow(
                            {
                                "initial_type": initial_type,
                                "threshold": theta,
                                "track_length": ell,
                                "counted_type": counted_type,
                                "estimated_count": mean,
                                "standard_error": se,
                                "nonzero_contributions": (
                                    nonzero_by_initial_type[initial_type][theta][i][j]
                                ),
                            }
                        )


if __name__ == "__main__":
    # Experiment settings: n histories PER initial type, shared by all thresholds.
    observation_times = [i / 10 for i in range(61)]  # 0, 0.1, ..., 6
    n = 10_000
    thresholds = (0.1, 0.03, 0.01)
    initial_types = ("electron", "photon")
    mu = 1.0
    gamma0 = 1.0
    alpha = 0.0
    seed = 20261015
    output_csv = Path(__file__).with_name("counts.csv")
    output_settings = Path(__file__).with_name("counts_settings.json")

    # Record the actual seed used for each initial-type ensemble.
    ensemble_seeds = {
        initial_type: seed + i for i, initial_type in enumerate(initial_types)
    }
    results_by_initial_type = {}
    nonzero_by_initial_type = {}

    print(f"Histories per initial type: {n}; thresholds: {thresholds}")
    print(f"mu={mu}; gamma0={gamma0}; alpha={alpha}; seed={seed}")
    print("Expected counts above threshold; SE = Monte Carlo standard error")

    for initial_type in initial_types:
        # Give each initial type a separate, reproducible random stream.
        ensemble_seed = ensemble_seeds[initial_type]
        print(
            f"\nSimulating {n} histories: initial type={initial_type}, "
            f"seed={ensemble_seed}",
            flush=True,
        )
        results, nonzero_counts = estimate_counts_for_thresholds(
            observation_times,
            n=n,
            thresholds=thresholds,
            initial_type=initial_type,
            mu=mu,
            gamma0=gamma0,
            alpha=alpha,
            seed=ensemble_seed,
            return_nonzero=True,
        )
        results_by_initial_type[initial_type] = results
        nonzero_by_initial_type[initial_type] = nonzero_counts

        for theta, rows in results.items():
            print(f"\nInitial type: {initial_type}; threshold: {theta}")
            print(
                f"{'ell':>6}  {'electrons':>12}  {'SE(e)':>12}  "
                f"{'photons':>12}  {'SE(photon)':>12}"
            )
            for ell, electron_mean, electron_se, photon_mean, photon_se in rows:
                print(
                    f"{ell:6.2f}  {electron_mean:12.6f}  {electron_se:12.6f}  "
                    f"{photon_mean:12.6f}  {photon_se:12.6f}"
                )

    save_counts_csv(output_csv, results_by_initial_type, nonzero_by_initial_type)
    settings = {
        "results_file": output_csv.name,
        "model": {
            "medium": "homogeneous, unbounded",
            "initial_energy": 1.0,
            "fragmentation_kernel": "k(u) = mu/u",
            "photon_splitting_density": "q(u) = 1 (uniform)",
            "continuous_energy_loss": "beta(energy) = alpha * energy",
            "mu": mu,
            "gamma0": gamma0,
            "alpha": alpha,
        },
        "histories_per_initial_type": n,
        "total_histories": n * len(initial_types),
        "initial_types": initial_types,
        "thresholds": thresholds,
        "observation_times": observation_times,
        "base_seed": seed,
        "ensemble_seeds": ensemble_seeds,
        "python_version": platform.python_version(),
        "estimator": "mean((W/v) * indicator(v > theta and type == counted_type))",
        "estimated_quantity": "C_theta,j / theta (expected particle count)",
        "standard_error": "sample standard deviation of contributions / sqrt(n)",
        "nonzero_contributions": "number of histories with a nonzero contribution",
        "shared_histories": "all thresholds, counted types and times within each ensemble",
    }
    with output_settings.open("w", encoding="utf-8") as output:
        json.dump(settings, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"\nSaved counts to {output_csv}")
    print(f"Saved settings to {output_settings}")
