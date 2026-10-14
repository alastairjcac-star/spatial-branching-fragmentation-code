"""Small CSV helpers and explicit boundaries for generated run directories."""

import csv
import math
from pathlib import Path

from .model import PARTICLE_TYPES

TEXT_FIELDS = {"initial_type", "observed_type"}
INTEGER_FIELDS = {"n", "nonzero_count"}
FLOAT_FIELDS = {
    "track_length",
    "threshold",
    "sum_payoff",
    "sum_squared_payoff",
    "normalized_mean",
    "normalized_sample_variance",
    "normalized_se",
    "raw_count_mean",
    "raw_count_se",
    "normal_ci_low",
    "normal_ci_high",
}
COUNT_FIELDS = TEXT_FIELDS | INTEGER_FIELDS | FLOAT_FIELDS | {"all_zero"}


def create_output_directory(path: str | Path) -> Path:
    """Create a fresh directory strictly below the checkout's ignored results/."""
    output = Path(path).resolve()
    results = Path.cwd().resolve() / "results"
    if output == results or not output.is_relative_to(results):
        raise ValueError(
            "output must be a new subdirectory of results/ in this checkout"
        )
    output.mkdir(parents=True, exist_ok=False)
    return output


def write_csv(rows: list[dict], path: str | Path) -> None:
    """Save named statistics in their supplied order, using round-trip floats."""
    if not rows:
        raise ValueError("cannot save an empty table")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _check_count_row(row: dict) -> None:
    if any(row[field] not in PARTICLE_TYPES for field in TEXT_FIELDS):
        raise ValueError("unknown particle type")
    if not all(math.isfinite(row[field]) for field in FLOAT_FIELDS):
        raise ValueError("count statistics must be finite")
    if not (0 < row["threshold"] < 1) or row["track_length"] < 0:
        raise ValueError("invalid threshold or observation time")
    n, nonzero = row["n"], row["nonzero_count"]
    if n < 2 or not 0 <= nonzero <= n:
        raise ValueError("invalid sample or nonzero count")
    if row["all_zero"] != (nonzero == 0):
        raise ValueError("all_zero is inconsistent with nonzero_count")
    if not 0 <= row["normalized_mean"] <= 1:
        raise ValueError("normalized mean must lie in [0,1]")
    for field in (
        "normalized_sample_variance",
        "normalized_se",
        "raw_count_se",
        "sum_payoff",
        "sum_squared_payoff",
    ):
        if row[field] < 0:
            raise ValueError(f"{field} must be nonnegative")
    tolerance = 1e-10 * n
    if (
        row["sum_payoff"] > nonzero + tolerance
        or row["sum_squared_payoff"] > row["sum_payoff"] + tolerance
    ):
        raise ValueError("sufficient statistics exceed bounded-payoff limits")
    reconstructed_variance = (
        row["sum_squared_payoff"] - row["sum_payoff"] ** 2 / n
    ) / (n - 1)
    if not math.isclose(
        row["normalized_sample_variance"],
        reconstructed_variance,
        rel_tol=1e-9,
        abs_tol=1e-10,
    ):
        raise ValueError("sample variance is inconsistent with sufficient statistics")
    expected = {
        "normalized_mean": row["sum_payoff"] / n,
        "normalized_se": math.sqrt(row["normalized_sample_variance"] / n),
        "raw_count_mean": row["normalized_mean"] / row["threshold"],
        "raw_count_se": row["normalized_se"] / row["threshold"],
        "normal_ci_low": row["raw_count_mean"] - 1.96 * row["raw_count_se"],
        "normal_ci_high": row["raw_count_mean"] + 1.96 * row["raw_count_se"],
    }
    for field, value in expected.items():
        if not math.isclose(row[field], value, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"inconsistent {field}")


def load_estimates(path: str | Path) -> list[dict]:
    """Read and check saved count estimates, restoring numbers and booleans.

    The schema accepts both original and refactored count runs. Validation
    checks identities between the saved statistics; it does not resimulate.
    """
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != COUNT_FIELDS:
            raise ValueError("estimate CSV has missing or unknown columns")
        if len(reader.fieldnames) != len(COUNT_FIELDS):
            raise ValueError("estimate CSV has duplicate columns")
        for line, fields in enumerate(reader, start=2):
            try:
                if None in fields or any(value is None for value in fields.values()):
                    raise ValueError("wrong number of columns")
                if fields["all_zero"] not in ("True", "False"):
                    raise ValueError("all_zero must be True or False")
                row = {
                    field: int(value)
                    if field in INTEGER_FIELDS
                    else float(value)
                    if field in FLOAT_FIELDS
                    else value == "True"
                    if field == "all_zero"
                    else value
                    for field, value in fields.items()
                }
                _check_count_row(row)
            except (ValueError, OverflowError) as error:
                raise ValueError(f"invalid estimate CSV row {line}: {error}") from error
            rows.append(row)
    if not rows:
        raise ValueError("estimate CSV contains no data rows")
    return rows
