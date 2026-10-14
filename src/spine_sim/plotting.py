"""Noninteractive count plots for the reference demonstration."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .model import PARTICLE_TYPES  # noqa: E402


def plot_counts(rows: list[dict], output: Path, histories: int) -> None:
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
    figure.suptitle(
        f"Demonstration output — {histories:,} independent histories per initial type"
    )
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
