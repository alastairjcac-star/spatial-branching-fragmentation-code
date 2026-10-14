"""Exercise separate simulation/plot commands and replay a saved source archive."""

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


def invoke(root, *arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    environment["MPLCONFIGDIR"] = str(root / "results/matplotlib-cache")
    return subprocess.run(
        [sys.executable, "-m", "spine_sim.cli", *arguments],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.fixture
def checkout(tmp_path):
    original = Path(__file__).parents[1]
    root = tmp_path / "checkout"
    root.mkdir()
    for name in (
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        ".gitignore",
        "README.md",
    ):
        shutil.copy2(original / name, root / name)
    shutil.copytree(
        original / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    settings = json.loads((original / "examples/reference.json").read_text())
    settings.update(histories=12, observation_times=[0, 0.5, 2])
    (root / "small.json").write_text(json.dumps(settings))
    return root


def test_cli_snapshot_replays_exact_estimates(checkout, tmp_path):
    root = checkout
    invoke(root, "simulate", "--config", "small.json", "--output", "results/test")
    output = root / "results/test"
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["history_count_total"] == 24
    assert metadata["rng"]["bit_generator"] == "PCG64"
    assert metadata["git"]["revision"] is None
    assert (
        metadata["source"]["archive_sha256"]
        == hashlib.sha256((output / "source.tar.gz").read_bytes()).hexdigest()
    )
    for name, expected in metadata["outputs_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    with (output / "estimates.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 * 3 * 3 * 2
    assert {int(row["n"]) for row in rows} == {12}
    assert set(metadata["outputs_sha256"]) == {"config.json", "estimates.csv"}
    assert not (output / "counts.png").exists()
    assert not (output / "counts.pdf").exists()
    original_files = {
        name: (output / name).read_bytes()
        for name in ("config.json", "estimates.csv", "metadata.json", "source.tar.gz")
    }
    invoke(root, "plot", "--input", "results/test", "--output", "results/plots")
    plot_output = root / "results/plots"
    plot_metadata = json.loads((plot_output / "plot-metadata.json").read_text())
    for name in ("counts.png", "counts.pdf"):
        assert (plot_output / name).stat().st_size > 1000
        assert (
            hashlib.sha256((plot_output / name).read_bytes()).hexdigest()
            == plot_metadata["outputs_sha256"][name]
        )
    for name, expected in plot_metadata["input_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert (plot_output / "source.tar.gz").is_file()
    invoke(
        root,
        "plot",
        "--input",
        "results/test",
        "--output",
        "results/retitled",
        "--title",
        "The same estimates, a different title",
    )
    retitled = json.loads((root / "results/retitled/plot-metadata.json").read_text())
    assert retitled["title"] == "The same estimates, a different title"
    assert retitled["input_sha256"] == plot_metadata["input_sha256"]
    for name, content in original_files.items():
        assert (output / name).read_bytes() == content
    replay = tmp_path / "replay"
    replay.mkdir()
    with tarfile.open(output / "source.tar.gz") as archive:
        names = archive.getnames()
        assert not any(
            name.startswith(("results/", ".git/", ".venv/")) for name in names
        )
        archive.extractall(replay, filter="data")
    shutil.copy2(output / "config.json", replay / "run-config.json")
    invoke(
        replay,
        "simulate",
        "--config",
        "run-config.json",
        "--output",
        "results/reproduced-demo",
    )
    assert (output / "estimates.csv").read_bytes() == (
        replay / "results/reproduced-demo/estimates.csv"
    ).read_bytes()
    # A completed run is protected against accidental overwriting.
    with pytest.raises(subprocess.CalledProcessError) as error:
        invoke(root, "simulate", "--config", "small.json", "--output", "results/test")
    assert "File exists" in error.value.stderr
    with pytest.raises(subprocess.CalledProcessError) as error:
        invoke(root, "plot", "--input", "results/test", "--output", "results/plots")
    assert "File exists" in error.value.stderr


def test_legacy_flags_alias_simulation_without_figures(checkout):
    invoke(checkout, "--config", "small.json", "--output", "results/legacy")
    invoke(
        checkout, "simulate", "--config", "small.json", "--output", "results/explicit"
    )
    assert (checkout / "results/legacy/estimates.csv").read_bytes() == (
        checkout / "results/explicit/estimates.csv"
    ).read_bytes()
    assert not (checkout / "results/legacy/counts.png").exists()
