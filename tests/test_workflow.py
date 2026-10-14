"""Check saved-result workflows without recomputing simulations during plots."""

import csv
import json
from dataclasses import asdict

import numpy as np
import pytest

from spine_sim import config as config_module
from spine_sim import plotting, simulator, workflow
from spine_sim.config import RunConfig, history_generator
from spine_sim.estimation import CountAccumulator
from spine_sim.io import load_estimates, write_csv
from spine_sim.provenance import file_hash
from spine_sim.simulator import simulate_spine


@pytest.fixture
def config():
    return RunConfig.from_dict(
        {
            "parameters": {"mu": 1.0, "gamma0": 1.0, "alpha": 0.0, "initial_energy": 1},
            "initial_types": ["photon", "electron"],
            "observation_times": [0, 0.5, 2],
            "thresholds": [0.1, 0.03],
            "histories": 7,
            "seed": 749,
        }
    )


@pytest.fixture
def rows(config):
    return workflow.estimate_counts(config)[0]


def test_estimate_counts_preserves_stream_and_reduction_order(config):
    """The extracted orchestration preserves the former direct-loop result."""
    expected_rows = []
    expected_switches = {}
    for initial in config.initial_types:
        accumulator = CountAccumulator(config.observation_times, config.thresholds)
        total_switches = 0
        for history_index in range(config.histories):
            path = simulate_spine(
                config.parameters,
                config.observation_times,
                initial,
                history_generator(config.seed, initial, history_index),
            )
            accumulator.update(path)
            total_switches += path.switch_count
        expected_rows.extend(accumulator.rows(initial))
        expected_switches[initial] = total_switches
    actual_rows, actual_switches = workflow.estimate_counts(config)
    assert actual_rows == expected_rows
    assert actual_switches == expected_switches


def test_estimates_csv_roundtrip_preserves_values_and_types(rows, tmp_path):
    path = tmp_path / "estimates.csv"
    write_csv(rows, path)
    loaded = load_estimates(path)
    assert loaded == rows
    for row in loaded:
        assert type(row["n"]) is int
        assert type(row["nonzero_count"]) is int
        assert type(row["all_zero"]) is bool
        assert type(row["threshold"]) is float
        assert type(row["raw_count_mean"]) is float


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("n", "1"),
        ("n", "7.5"),
        ("nonzero_count", "8"),
        ("all_zero", "maybe"),
        ("all_zero", "True"),
        ("observed_type", "muon"),
        ("threshold", "0"),
        ("track_length", "-1"),
        ("raw_count_mean", "nan"),
        ("normalized_sample_variance", "-0.1"),
        ("sum_squared_payoff", "8"),
        ("normalized_mean", "0.3"),
        ("normalized_se", "0.2"),
        ("raw_count_mean", "2"),
        ("normal_ci_low", "-2"),
    ],
)
def test_estimate_loader_rejects_invalid_or_inconsistent_rows(
    rows, tmp_path, field, bad_value
):
    # At time zero the initial photon payoff is positive and deterministic.
    row = next(
        dict(item)
        for item in rows
        if item["track_length"] == 0 and item["observed_type"] == "photon"
    )
    row[field] = bad_value
    path = tmp_path / "invalid.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError):
        load_estimates(path)


def test_estimate_loader_rejects_missing_column_and_empty_data(rows, tmp_path):
    path = tmp_path / "incomplete.csv"
    fields = [key for key in rows[0] if key != "normalized_se"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        load_estimates(path)
    with path.open("w", newline="") as stream:
        csv.DictWriter(stream, fieldnames=list(rows[0])).writeheader()
    with pytest.raises(ValueError):
        load_estimates(path)


def saved_run(root, config, rows, *, schema_version=2):
    directory = root / "results/saved"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(asdict(config)))
    write_csv(rows, directory / "estimates.csv")
    metadata = {
        "schema_version": schema_version,
        "configuration": asdict(config),
        "history_count_per_initial_type": config.histories,
        "history_count_total": config.histories * len(config.initial_types),
        "outputs_sha256": {
            name: file_hash(directory / name)
            for name in ("config.json", "estimates.csv")
        },
    }
    (directory / "metadata.json").write_text(json.dumps(metadata))
    return directory


@pytest.mark.parametrize("schema_version", [1, 2])
def test_saved_plot_uses_no_simulation_or_rng_and_preserves_inputs(
    config, rows, tmp_path, monkeypatch, schema_version
):
    directory = saved_run(tmp_path, config, rows, schema_version=schema_version)
    originals = {path.name: path.read_bytes() for path in directory.iterdir()}
    monkeypatch.chdir(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "plotting must use saved estimates without simulation or RNG"
        )

    monkeypatch.setattr(workflow, "estimate_counts", forbidden)
    monkeypatch.setattr(workflow, "simulate_spine", forbidden)
    monkeypatch.setattr(simulator, "simulate_spine", forbidden)
    monkeypatch.setattr(config_module, "history_generator", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np.random, "PCG64", forbidden)
    monkeypatch.setattr(
        plotting,
        "capture_source",
        lambda root, output: {"git": {"revision": None, "dirty": None}, "source": {}},
    )
    output = tmp_path / "results/replotted"
    metadata = plotting.plot_saved_counts(directory, output, title="Saved-data plot")
    assert metadata["title"] == "Saved-data plot"
    assert metadata["histories_per_initial_type"] == config.histories
    for name in ("counts.png", "counts.pdf"):
        assert (output / name).stat().st_size > 1000
        assert metadata["outputs_sha256"][name] == file_hash(output / name)
    assert (output / "plot-metadata.json").is_file()
    for name, content in originals.items():
        assert (directory / name).read_bytes() == content


def test_saved_plot_rejects_changed_input_hash(config, rows, tmp_path, monkeypatch):
    directory = saved_run(tmp_path, config, rows)
    with (directory / "estimates.csv").open("a") as stream:
        stream.write("\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="hash|checksum|changed|SHA"):
        plotting.plot_saved_counts(directory, tmp_path / "results/plot")


def test_saved_plot_rejects_rows_inconsistent_with_config(
    config, rows, tmp_path, monkeypatch
):
    directory = saved_run(tmp_path, config, rows[:-1])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        plotting.plot_saved_counts(directory, tmp_path / "results/plot")
