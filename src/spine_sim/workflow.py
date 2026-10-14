"""Research workflow: parameters, independent histories, estimates, saved run."""

import shlex
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .config import RunConfig, history_generator
from .estimation import CountAccumulator
from .io import create_output_directory, write_csv
from .provenance import capture_source, environment_metadata, file_hash, write_json
from .simulator import simulate_spine


def estimate_counts(config: RunConfig) -> tuple[list[dict], dict[str, int]]:
    """Estimate each requested count, keeping only online sufficient statistics.

    The history order and stream mapping are preserved from the reference
    implementation, so fixed configurations retain their seeded estimates.
    """
    rows = []
    switch_counts = {}
    for initial in config.initial_types:
        accumulator = CountAccumulator(config.observation_times, config.thresholds)
        total_switches = 0
        for history_index in range(config.histories):
            rng = history_generator(config.seed, initial, history_index)
            path = simulate_spine(
                config.parameters, config.observation_times, initial, rng
            )
            accumulator.update(path)
            total_switches += path.switch_count
        rows.extend(accumulator.rows(initial))
        switch_counts[initial] = total_switches
    return rows, switch_counts


def run_experiment(config_path: Path, output: Path) -> dict:
    """Run serial independent histories; refuse to overwrite an existing run."""
    start = time.perf_counter()
    config = RunConfig.load(config_path)
    output = create_output_directory(output)
    root = Path.cwd().resolve()
    info = environment_metadata()
    info.update(capture_source(root, output))
    settings = asdict(config)
    write_json(output / "config.json", settings)
    simulation_start = time.perf_counter()
    rows, switch_counts = estimate_counts(config)
    simulation_seconds = time.perf_counter() - simulation_start
    write_csv(rows, output / "estimates.csv")
    command = [
        "uv",
        "run",
        "--locked",
        "spine-sim",
        "simulate",
        "--config",
        str(config_path),
        "--output",
        str(output.relative_to(root)),
    ]
    info.update(
        {
            "schema_version": 2,
            "run_kind": "count_estimation",
            "configuration": settings,
            "rng": {
                "library": "numpy",
                "bit_generator": "PCG64",
                "seed": config.seed,
                "stream_policy": (
                    "SeedSequence(seed, spawn_key=(type_code, history_index)); "
                    "type_code electron=0, photon=1; zero-based history_index."
                ),
                "parallelism": "serial; stream identities do not depend on scheduling",
            },
            "history_count_per_initial_type": config.histories,
            "history_count_total": config.histories * len(config.initial_types),
            "switch_counts": switch_counts,
            "stopping_rule": (
                "Each history stops at the last observation; no early energy cutoff."
            ),
            "numerical_conventions": {
                "jump_cutoff": None,
                "time_step": None,
                "omitted_event_treatment": (
                    "none; gamma plus thinned-Poisson decomposition"
                ),
                "coincident_events": "switch before observation (right-continuous)",
            },
            "uncertainty": (
                "Sample variance with ddof=1 across independent histories; "
                "approximate marginal normal mean +/- 1.96 SE. Curves sharing "
                "histories are correlated; no simultaneous coverage claim. "
                "All-zero samples can miss rare events."
            ),
            "runtime_seconds": time.perf_counter() - start,
            "simulation_seconds": simulation_seconds,
            "invocation_argv": sys.argv,
            "working_directory": str(root),
            "reproduction_command": shlex.join(command),
            "reproduction_note": (
                "Use the same relative directory and a fresh output location. "
                "For an archived run, extract source.tar.gz and copy the saved "
                "config.json into the extracted directory as run-config.json, "
                "then use snapshot_reproduction_command. Estimates are "
                "reproducible in the recorded environment; runtime and provenance vary."
            ),
            "snapshot_reproduction_command": (
                "uv run --locked spine-sim simulate --config run-config.json "
                "--output results/reproduced-demo"
            ),
            "outputs_sha256": {
                name: file_hash(output / name)
                for name in ("config.json", "estimates.csv")
            },
        }
    )
    write_json(output / "metadata.json", info)
    return info
