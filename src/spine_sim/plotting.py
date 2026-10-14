"""Postprocess saved count estimates without running any histories."""

import json
import shlex
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import RunConfig  # noqa: E402
from .io import create_output_directory, load_estimates  # noqa: E402
from .model import PARTICLE_TYPES  # noqa: E402
from .provenance import (  # noqa: E402
    capture_source,
    environment_metadata,
    file_hash,
    write_json,
)


def plot_counts(
    rows: list[dict],
    output: Path,
    histories: int,
    title: str = "Spine count estimates",
) -> None:
    """Four panels, with raw-count marginal normal Monte Carlo intervals."""
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    thresholds = sorted({row["threshold"] for row in rows})
    for row_index, initial in enumerate(PARTICLE_TYPES):
        for column_index, observed in enumerate(PARTICLE_TYPES):
            ax = axes[row_index, column_index]
            for theta in thresholds:
                selected = [
                    row
                    for row in rows
                    if row["initial_type"] == initial
                    and row["observed_type"] == observed
                    and row["threshold"] == theta
                ]
                if not selected:
                    continue
                times = [row["track_length"] for row in selected]
                (line,) = ax.plot(
                    times,
                    [row["raw_count_mean"] for row in selected],
                    label=rf"$\theta={theta:g}$",
                )
                ax.fill_between(
                    times,
                    [row["normal_ci_low"] for row in selected],
                    [row["normal_ci_high"] for row in selected],
                    color=line.get_color(),
                    alpha=0.18,
                )
            ax.set_title(f"Initial {initial} → count {observed}s")
            ax.set_xlabel("Track length")
            ax.set_ylabel("Expected count above strict threshold")
            ax.grid(alpha=0.2)
            if ax.lines:
                ax.legend()
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Initial type not simulated",
                    ha="center",
                    transform=ax.transAxes,
                )
    figure.suptitle(f"{title} — {histories:,} independent histories per initial type")
    figure.text(
        0.5,
        0.015,
        "Shading: approximate pointwise 95% Monte Carlo intervals; "
        "curves are correlated.\n"
        "All-zero samples can miss rare events; these are not simultaneous bands.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    figure.savefig(output / "counts.png", dpi=160)
    figure.savefig(output / "counts.pdf")
    plt.close(figure)


def plot_saved_counts(
    input_dir: Path,
    output_dir: Path | None = None,
    title: str = "Spine count estimates",
) -> dict:
    """Read, verify, and plot a saved run in a fresh figure directory.

    Input estimates and simulation metadata are never modified. Both the
    original schema-1 runs and schema-2 simulation-only runs are supported.
    """
    start = time.perf_counter()
    source = Path(input_dir).resolve()
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") not in (1, 2):
        raise ValueError("unsupported simulation metadata schema")
    expected_hashes = metadata.get("outputs_sha256", {})
    for name in ("config.json", "estimates.csv"):
        if not isinstance(expected_hashes, dict) or expected_hashes.get(
            name
        ) != file_hash(source / name):
            raise ValueError(f"input checksum mismatch for {name}")
    config = RunConfig.load(source / "config.json")
    rows = load_estimates(source / "estimates.csv")
    expected_labels = [
        (initial, observed, t, theta, config.histories)
        for initial in config.initial_types
        for t in config.observation_times
        for theta in config.thresholds
        for observed in PARTICLE_TYPES
    ]
    actual_labels = [
        (
            row["initial_type"],
            row["observed_type"],
            row["track_length"],
            row["threshold"],
            row["n"],
        )
        for row in rows
    ]
    if actual_labels != expected_labels:
        raise ValueError("saved estimates do not match the run configuration")
    output = create_output_directory(output_dir or source / "figures")
    root = Path.cwd().resolve()
    info = environment_metadata()
    info.update(capture_source(root, output))
    plot_counts(rows, output, config.histories, title)
    command = [
        "uv",
        "run",
        "--locked",
        "spine-sim",
        "plot",
        "--input",
        str(input_dir),
        "--output",
        str(output.relative_to(root)),
        "--title",
        title,
    ]
    info.update(
        {
            "schema_version": 1,
            "run_kind": "count_plot",
            "title": title,
            "histories_per_initial_type": config.histories,
            "input_directory": str(source),
            "output_directory": str(output),
            "input_sha256": {
                name: file_hash(source / name)
                for name in ("config.json", "estimates.csv", "metadata.json")
            },
            "outputs_sha256": {
                name: file_hash(output / name) for name in ("counts.png", "counts.pdf")
            },
            "runtime_seconds": time.perf_counter() - start,
            "invocation_argv": sys.argv,
            "reproduction_command": shlex.join(command),
            "reproduction_note": (
                "Preserve the three input files with this record. To replay with "
                "the plotted source, extract source.tar.gz into a fresh directory, "
                "copy input config.json, estimates.csv, metadata.json into run-input/, "
                "install with uv sync --locked, then use snapshot_reproduction_command."
            ),
            "snapshot_reproduction_command": shlex.join(
                [
                    "uv",
                    "run",
                    "--locked",
                    "spine-sim",
                    "plot",
                    "--input",
                    "run-input",
                    "--output",
                    "results/reproduced-figures",
                    "--title",
                    title,
                ]
            ),
        }
    )
    write_json(output / "plot-metadata.json", info)
    return info
