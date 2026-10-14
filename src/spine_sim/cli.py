"""Run the small, reproducible independent-spine demonstration."""

import argparse
import csv
import shlex
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .config import RunConfig, history_generator
from .estimation import CountAccumulator
from .provenance import capture_source, environment_metadata, file_hash, write_json
from .simulator import simulate_spine


def run(config_path: Path, output: Path) -> dict:
    """Run serial independent histories; refuse to overwrite an existing run."""
    start = time.perf_counter()
    config = RunConfig.load(config_path)
    output = output.resolve()
    root = Path.cwd().resolve()
    if output == root / "results" or not output.is_relative_to(root / "results"):
        raise ValueError(
            "output must be a new subdirectory of results/ in this checkout"
        )
    output.mkdir(parents=True, exist_ok=False)
    info = environment_metadata()
    info.update(capture_source(root, output))
    settings = asdict(config)
    write_json(output / "config.json", settings)
    rows = []
    switch_counts = {}
    simulation_start = time.perf_counter()
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
    simulation_seconds = time.perf_counter() - simulation_start
    with (output / "estimates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Import after the simulator so its runtime can be reported separately.
    from .plotting import plot_counts

    plot_counts(rows, output, config.histories)
    command = [
        "uv",
        "run",
        "--locked",
        "spine-sim",
        "--config",
        str(config_path),
        "--output",
        str(output.relative_to(root)),
    ]
    info.update(
        {
            "schema_version": 1,
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
                "uv run --locked spine-sim --config run-config.json "
                "--output results/reproduced-demo"
            ),
            "outputs_sha256": {
                name: file_hash(output / name)
                for name in ("config.json", "estimates.csv", "counts.png", "counts.pdf")
            },
        }
    )
    write_json(output / "metadata.json", info)
    return info


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Tracked experiment JSON"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory under results/"
    )
    args = parser.parse_args(argv)
    try:
        info = run(args.config, args.output)
    except (ValueError, OSError, FloatingPointError) as error:
        parser.exit(2, f"spine-sim: {error}\n")
    print(f"Saved {info['history_count_total']:,} histories to {args.output}")
    print(f"Simulation runtime: {info['simulation_seconds']:.3f} seconds")


if __name__ == "__main__":
    main()
