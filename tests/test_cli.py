"""Exercise the saved demonstration and reproduce it from its source archive."""

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def invoke(root, config, output):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    environment["MPLCONFIGDIR"] = str(root / "results/matplotlib-cache")
    return subprocess.run(
        [sys.executable, "-m", "spine_sim.cli", "--config", config, "--output", output],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )


def test_cli_snapshot_replays_exact_estimates(tmp_path):
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
    invoke(root, "small.json", "results/test")
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
    assert (output / "counts.png").stat().st_size > 1000
    assert (output / "counts.pdf").stat().st_size > 1000
    replay = tmp_path / "replay"
    replay.mkdir()
    with tarfile.open(output / "source.tar.gz") as archive:
        names = archive.getnames()
        assert not any(
            name.startswith(("results/", ".git/", ".venv/")) for name in names
        )
        archive.extractall(replay, filter="data")
    shutil.copy2(output / "config.json", replay / "run-config.json")
    invoke(replay, "run-config.json", "results/reproduced-demo")
    assert (output / "estimates.csv").read_bytes() == (
        replay / "results/reproduced-demo/estimates.csv"
    ).read_bytes()
    # A completed run is protected against accidental overwriting.
    try:
        invoke(root, "small.json", "results/test")
    except subprocess.CalledProcessError as error:
        assert "File exists" in error.stderr
    else:
        raise AssertionError("existing output was overwritten")
