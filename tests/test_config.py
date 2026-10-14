"""Experiment validation and stable independent stream identities."""

import json
from pathlib import Path

import numpy as np
import pytest

from spine_sim.config import RunConfig, history_generator


def example_dict():
    return json.loads(
        (Path(__file__).parents[1] / "examples/reference.json").read_text()
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("histories", 1),
        ("histories", True),
        ("seed", -1),
        ("seed", 1.5),
        ("initial_types", ["electron", "electron"]),
        ("thresholds", [0.1, 0.1]),
        ("observation_times", [1, 0]),
        ("thresholds", [True]),
        ("description", 9),
    ],
)
def test_invalid_configuration(field, value):
    values = example_dict()
    values[field] = value
    with pytest.raises(ValueError):
        RunConfig.from_dict(values)


def test_parameters_must_be_explicit():
    values = example_dict()
    del values["parameters"]["alpha"]
    with pytest.raises(ValueError, match="explicitly"):
        RunConfig.from_dict(values)


def test_seed_streams_do_not_depend_on_iteration_order():
    keys = [("electron", 0), ("electron", 1), ("photon", 0)]
    forward = {key: history_generator(42, *key).random(10) for key in keys}
    reverse = {key: history_generator(42, *key).random(10) for key in reversed(keys)}
    for key in keys:
        np.testing.assert_array_equal(forward[key], reverse[key])
    assert not np.array_equal(forward[keys[0]], forward[keys[1]])
    assert not np.array_equal(forward[keys[0]], forward[keys[2]])
