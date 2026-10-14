"""Saved analytic reports, limiting cases and deterministic seeded reproduction."""

import csv
import json
import math
import tarfile
from pathlib import Path

import numpy as np
import pytest

from spine_sim import validation
from spine_sim.provenance import file_hash


@pytest.fixture
def settings():
    path = Path(__file__).parents[1] / "experiments/validation/config.json"
    result = json.loads(path.read_text())
    result["histories"] = 12
    return result


@pytest.fixture
def report_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Source archival has its own integration tests. This fixture isolates
    # report computation and asserts that its provenance hooks are invoked.
    marker = tmp_path / "source-marker.txt"
    marker.write_text("validation source fixture\n")

    def capture(root, output):
        assert root == tmp_path
        with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
            archive.add(marker, arcname=marker.name)
        return {
            "git": {"revision": "fixture", "dirty": True},
            "source": {
                "archive": "source.tar.gz",
                "archive_sha256": file_hash(output / "source.tar.gz"),
            },
        }

    monkeypatch.setattr(validation, "capture_source", capture)
    monkeypatch.setattr(
        validation, "environment_metadata", lambda: {"python": "fixture"}
    )
    return tmp_path


def _save(path, settings):
    path.write_text(json.dumps(settings))
    return path


def _rows(output):
    with (output / "validation.csv").open() as stream:
        return list(csv.DictReader(stream))


def test_report_records_finite_comparisons_and_reproduces(report_directory, settings):
    config = _save(report_directory / "small.json", settings)
    first, second = (
        report_directory / "results" / name for name in ("first", "second")
    )
    metadata = validation.run_validation(config, first)
    validation.run_validation(config, second)
    assert (first / "validation.csv").read_bytes() == (
        second / "validation.csv"
    ).read_bytes()
    assert json.loads((first / "config.json").read_text()) == settings
    assert metadata["validation_rows"] == 7
    assert metadata["sample_count_per_case"] == 12
    assert metadata["history_count_total"] == 36
    assert metadata["residual_increment_count"] == 12
    assert metadata["git"]["dirty"] is True
    assert "spawn_key=(2," in metadata["rng"]["stream_policy"]
    assert "spine-sim validate" in metadata["snapshot_reproduction_command"]
    for name, digest in metadata["outputs_sha256"].items():
        assert file_hash(first / name) == digest
    rows = _rows(first)
    assert {row["family"] for row in rows} == {
        "residual_laplace",
        "type_probability",
        "one_split_raw_count",
    }
    for row in rows:
        assert int(row["n"]) == 12
        for key in (
            "empirical_mean",
            "analytic_expectation",
            "empirical_mc_se",
            "theoretical_mc_se",
            "standardized_discrepancy",
            "difference",
        ):
            assert math.isfinite(float(row[key]))
        assert float(row["standardized_discrepancy"]) == pytest.approx(
            float(row["difference"]) / float(row["theoretical_mc_se"])
        )
    with pytest.raises(FileExistsError):
        validation.run_validation(config, first)


def test_analytic_report_matches_closed_form_special_cases(report_directory, settings):
    settings["laplace"].update(mu=1.0, h=1.0, powers=[1, 2])
    settings["type_chain"]["parameters"].update(mu=1.0, gamma0=1.0)
    settings["type_chain"]["horizon"] = 1.0
    settings["one_split"]["parameters"]["gamma0"] = 1.0
    settings["one_split"].update(horizon=1.0, threshold=0.5)
    path = _save(report_directory / "special.json", settings)
    validation.run_validation(path, report_directory / "results/special")
    rows = _rows(report_directory / "results/special")
    means = [float(row["analytic_expectation"]) for row in rows]
    np.testing.assert_allclose(
        means,
        [
            math.exp(-0.5),
            math.exp(-5 / 6),
            (1 + math.exp(-2)) / 2,
            (1 - math.exp(-2)) / 2,
            1 - math.exp(-1),
            math.exp(-1),
        ],
    )
    electron = rows[-2]
    event = 1 - math.exp(-1)
    # Raw electron payoff second moment is 2*log(2) times split probability.
    variance = 2 * math.log(2) * event - event**2
    assert float(electron["theoretical_mc_se"]) == pytest.approx(
        math.sqrt(variance / settings["histories"])
    )


def test_zero_rate_limits_do_not_invent_standardized_discrepancies(
    report_directory, settings
):
    settings["laplace"]["mu"] = 0
    settings["type_chain"]["parameters"].update(mu=0, gamma0=0)
    settings["one_split"]["parameters"]["gamma0"] = 0
    path = _save(report_directory / "zero.json", settings)
    validation.run_validation(path, report_directory / "results/zero")
    rows = _rows(report_directory / "results/zero")
    for row, expected in zip(rows, [1, 1, 1, 1, 0, 0, 1], strict=True):
        assert float(row["analytic_expectation"]) == expected
        assert float(row["empirical_mean"]) == pytest.approx(expected)
        assert float(row["theoretical_mc_se"]) == 0
        assert row["standardized_discrepancy"] == ""


@pytest.mark.parametrize(
    "change",
    [
        {"histories": 1},
        {"seed": True},
        {"unexpected": 1},
        {"laplace": {"mu": -1, "h": 1, "powers": [1]}},
        {"laplace": {"mu": 1, "h": 1, "powers": [1, True]}},
    ],
)
def test_invalid_configuration_fails_before_creating_output(
    report_directory, settings, change
):
    settings.update(change)
    path = _save(report_directory / "invalid.json", settings)
    output = report_directory / "results/invalid"
    with pytest.raises(ValueError):
        validation.run_validation(path, output)
    assert not output.exists()


def test_one_split_reference_requires_its_model_assumptions(report_directory, settings):
    settings["one_split"]["parameters"]["alpha"] = 0.2
    path = _save(report_directory / "wrong-model.json", settings)
    with pytest.raises(ValueError, match="mu=alpha=0"):
        validation.run_validation(path, report_directory / "results/wrong-model")
