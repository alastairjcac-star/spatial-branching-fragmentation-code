"""Explicit JSON experiment settings; no implicit numerical step or cutoff."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .estimation import CountAccumulator
from .model import PARTICLE_TYPES, ModelParameters


@dataclass(frozen=True)
class RunConfig:
    parameters: ModelParameters
    initial_types: tuple[str, ...]
    observation_times: tuple[float, ...]
    thresholds: tuple[float, ...]
    histories: int
    seed: int
    description: str = ""

    @classmethod
    def from_dict(cls, values: dict) -> RunConfig:
        required = {
            "parameters",
            "initial_types",
            "observation_times",
            "thresholds",
            "histories",
            "seed",
        }
        if not isinstance(values, dict):
            raise ValueError("configuration must be a JSON object")
        if not required <= values.keys() or values.keys() - required - {"description"}:
            raise ValueError("configuration has missing or unknown fields")
        for field, minimum in (("histories", 2), ("seed", 0)):
            if type(values[field]) is not int or values[field] < minimum:
                raise ValueError(f"{field} must be an integer >= {minimum}")
        if not isinstance(values["parameters"], dict):
            raise ValueError("parameters must be an object")
        if set(values["parameters"]) != {"mu", "gamma0", "alpha", "initial_energy"}:
            raise ValueError(
                "parameters must explicitly specify mu, gamma0, alpha, initial_energy"
            )
        for value in values["parameters"].values():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("model parameters must be numbers")
        parameters = ModelParameters(**values["parameters"])
        initial_types = values["initial_types"]
        if (
            not isinstance(initial_types, list)
            or not initial_types
            or any(value not in PARTICLE_TYPES for value in initial_types)
            or len(set(initial_types)) != len(initial_types)
        ):
            raise ValueError(
                "initial_types must be a nonempty list of distinct particle types"
            )
        for field in ("observation_times", "thresholds"):
            if not isinstance(values[field], list) or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in values[field]
            ):
                raise ValueError(f"{field} must be a list of numbers")
        accumulator = CountAccumulator(
            values["observation_times"], values["thresholds"]
        )
        if len(set(values["thresholds"])) != len(values["thresholds"]):
            raise ValueError("thresholds must be distinct")
        description = values.get("description", "")
        if not isinstance(description, str):
            raise ValueError("description must be text")
        return cls(
            parameters=parameters,
            initial_types=tuple(initial_types),
            observation_times=tuple(float(x) for x in accumulator.observation_times),
            thresholds=tuple(float(x) for x in accumulator.thresholds),
            histories=values["histories"],
            seed=values["seed"],
            description=description,
        )

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def history_generator(
    seed: int, initial_type: str, history_index: int
) -> np.random.Generator:
    """PCG64 stream keyed by seed, type (electron=0/photon=1), and history."""
    if initial_type not in PARTICLE_TYPES:
        raise ValueError("unknown initial type")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(history_index) is not int or history_index < 0:
        raise ValueError("history_index must be a nonnegative integer")
    type_code = PARTICLE_TYPES.index(initial_type)
    sequence = np.random.SeedSequence(seed, spawn_key=(type_code, history_index))
    return np.random.Generator(np.random.PCG64(sequence))
