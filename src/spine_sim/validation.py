"""Reproducible checks against analytic identities for the implemented model."""

import json
import math
import shlex
import sys
import time
from pathlib import Path

import numpy as np

from .config import RunConfig
from .estimation import count_payoffs
from .io import create_output_directory, write_csv
from .model import PARTICLE_TYPES
from .provenance import capture_source, environment_metadata, file_hash, write_json
from .sampling import sample_log_loss
from .simulator import simulate_spine


def _generator(seed: int, family: int, initial_type: str, index: int):
    sequence = np.random.SeedSequence(
        seed, spawn_key=(2, family, PARTICLE_TYPES.index(initial_type), index)
    )
    return np.random.Generator(np.random.PCG64(sequence))


def _load_config(path: Path) -> tuple[dict, RunConfig, RunConfig]:
    settings = json.loads(path.read_text(encoding="utf-8"))
    fields = {"histories", "seed", "description", "laplace", "type_chain", "one_split"}
    if not isinstance(settings, dict) or set(settings) != fields:
        raise ValueError("validation configuration has missing or unknown fields")
    sections = []
    for family in ("type_chain", "one_split"):
        section = settings[family]
        required = {"parameters", "horizon"}
        if family == "one_split":
            required.add("threshold")
        if not isinstance(section, dict) or set(section) != required:
            raise ValueError(f"{family} has missing or unknown fields")
        sections.append(
            RunConfig.from_dict(
                {
                    "parameters": section["parameters"],
                    "initial_types": list(PARTICLE_TYPES)
                    if family == "type_chain"
                    else ["photon"],
                    "observation_times": [section["horizon"]],
                    "thresholds": [section["threshold"]]
                    if family == "one_split"
                    else [0.5],
                    "histories": settings["histories"],
                    "seed": settings["seed"],
                    "description": settings["description"],
                }
            )
        )
    if sections[1].parameters.mu != 0 or sections[1].parameters.alpha != 0:
        raise ValueError("one_split requires mu=alpha=0 for its analytic reference")
    laplace = settings["laplace"]
    if not isinstance(laplace, dict) or set(laplace) != {"mu", "h", "powers"}:
        raise ValueError("laplace must specify mu, h and powers")
    for name in ("mu", "h"):
        value = laplace[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"laplace.{name} must be finite and nonnegative")
    powers = laplace["powers"]
    if (
        not isinstance(powers, list)
        or not powers
        or any(type(power) is not int or power < 1 for power in powers)
        or len(set(powers)) != len(powers)
    ):
        raise ValueError("laplace.powers must be distinct positive integers")
    return settings, *sections


def _laplace(power: int, h: float, mu: float) -> float:
    return math.exp(-mu * h * sum(1 / k for k in range(2, power + 2)))


def _row(values, expected, variance, **labels) -> dict:
    empirical = float(np.mean(values))
    theoretical_se = math.sqrt(max(0.0, variance) / len(values))
    discrepancy = empirical - expected
    standardized = discrepancy / theoretical_se if theoretical_se > 0 else None
    return {
        "family": "",
        "initial_type": "",
        "observed_type": "",
        "track_length": "",
        "threshold": "",
        "laplace_power": "",
        **labels,
        "n": len(values),
        "empirical_mean": empirical,
        "analytic_expectation": expected,
        "empirical_mc_se": float(np.std(values, ddof=1) / math.sqrt(len(values))),
        "theoretical_mc_se": theoretical_se,
        "difference": discrepancy,
        "standardized_discrepancy": standardized,
    }


def _report(settings: dict, chain: RunConfig, split: RunConfig) -> list[dict]:
    n, seed = settings["histories"], settings["seed"]
    mu, h = settings["laplace"]["mu"], settings["laplace"]["h"]
    increments = np.array(
        [
            sample_log_loss(h, mu, _generator(seed, 0, "electron", index))
            for index in range(n)
        ]
    )
    rows = []
    for power in settings["laplace"]["powers"]:
        expected = _laplace(power, h, mu)
        rows.append(
            _row(
                np.exp(-power * increments),
                expected,
                _laplace(2 * power, h, mu) - expected**2,
                family="residual_laplace",
                initial_type="electron",
                track_length=h,
                laplace_power=power,
            )
        )
    rate = chain.parameters.mu + chain.parameters.gamma0
    horizon = chain.observation_times[0]
    for initial in PARTICLE_TYPES:
        values = np.array(
            [
                simulate_spine(
                    chain.parameters,
                    [horizon],
                    initial,
                    _generator(seed, 1, initial, index),
                ).types[-1]
                == "electron"
                for index in range(n)
            ],
            dtype=float,
        )
        expected = float(initial == "electron")
        if rate > 0:
            relaxation = -math.expm1(-rate * horizon)
            expected += (chain.parameters.gamma0 / rate - expected) * relaxation
        rows.append(
            _row(
                values,
                expected,
                expected * (1 - expected),
                family="type_probability",
                initial_type=initial,
                observed_type="electron",
                track_length=horizon,
            )
        )
    horizon, theta = split.observation_times[0], split.thresholds[0]
    values = np.array(
        [
            count_payoffs(
                simulate_spine(
                    split.parameters,
                    [horizon],
                    "photon",
                    _generator(seed, 2, "photon", index),
                ),
                [theta],
            )[0, 0]
            / theta
            for index in range(n)
        ]
    )
    survival = math.exp(-split.parameters.gamma0 * horizon)
    event = -math.expm1(-split.parameters.gamma0 * horizon)
    expectations = (2 * (1 - theta) * event, survival)
    variances = (-2 * math.log(theta) * event - expectations[0] ** 2, survival * event)
    for index, observed in enumerate(PARTICLE_TYPES):
        rows.append(
            _row(
                values[:, index],
                expectations[index],
                variances[index],
                family="one_split_raw_count",
                initial_type="photon",
                observed_type=observed,
                track_length=horizon,
                threshold=theta,
            )
        )
    return rows


def run_validation(config_path: Path, output: Path) -> dict:
    """Save empirical/analytic comparisons and source for a fresh validation run."""
    start = time.perf_counter()
    settings, chain, split = _load_config(config_path)
    root = Path.cwd().resolve()
    output = create_output_directory(output)
    info = environment_metadata()
    info.update(capture_source(root, output))
    write_json(output / "config.json", settings)
    simulation_start = time.perf_counter()
    rows = _report(settings, chain, split)
    simulation_seconds = time.perf_counter() - simulation_start
    write_csv(rows, output / "validation.csv")
    command = [
        "uv",
        "run",
        "--locked",
        "spine-sim",
        "validate",
        "--config",
        str(config_path),
        "--output",
        str(output.relative_to(root)),
    ]
    info.update(
        {
            "schema_version": 1,
            "experiment": "analytic_validation",
            "configuration": settings,
            "validation_rows": len(rows),
            "sample_count_per_case": settings["histories"],
            "history_count_total": 3 * settings["histories"],
            "residual_increment_count": settings["histories"],
            "rng": {
                "library": "numpy",
                "bit_generator": "PCG64",
                "seed": settings["seed"],
                "stream_policy": (
                    "SeedSequence(seed, spawn_key=(2, family, type_code, "
                    "history_index)); "
                    "family residual=0, type_chain=1, one_split=2; "
                    "type_code electron=0, photon=1; zero-based history_index. "
                    "Namespace 2 is separate from standard history_generator streams."
                ),
                "parallelism": "serial; streams keyed by case and history",
            },
            "stopping_rule": "Paths stop at the configured horizon; no energy cutoff.",
            "numerical_conventions": {
                "jump_cutoff": None,
                "time_step": None,
                "omitted_event_treatment": "none; gamma plus thinned-Poisson sampling",
            },
            "uncertainty": (
                "Empirical SE uses sample variance ddof=1; theoretical SE uses the "
                "documented analytic spine-payoff variance. Standardized discrepancy "
                "divides empirical-minus-analytic mean by theoretical SE; empty when "
                "theoretical variance is zero. Powers share residual increments; "
                "the one-split types share paths. No simultaneous or proof claim."
            ),
            "runtime_seconds": time.perf_counter() - start,
            "simulation_seconds": simulation_seconds,
            "invocation_argv": sys.argv,
            "working_directory": str(root),
            "reproduction_command": shlex.join(command),
            "reproduction_note": (
                "Use a fresh output directory. To replay a source snapshot, extract "
                "source.tar.gz and copy config.json into it as run-config.json. "
                "Use the recorded environment and snapshot_reproduction_command. "
                "Numerical CSV reproduces; timing and provenance vary."
            ),
            "snapshot_reproduction_command": (
                "uv run --locked spine-sim validate --config run-config.json "
                "--output results/reproduced-validation"
            ),
            "outputs_sha256": {
                name: file_hash(output / name)
                for name in ("config.json", "validation.csv")
            },
        }
    )
    write_json(output / "metadata.json", info)
    return info
