"""Plot the saved Monte Carlo counts without running any simulations.

Run: python3 plot_counts.py
For a simple monochrome comparison: python3 plot_counts.py --simple
Dependency: Matplotlib (activate the project's .venv before running).

Reads counts.csv and counts_settings.json beside this script. Writes
counts_figure.pdf (vector graphics) and counts_figure.png (300 dpi) there.
Existing figure files are replaced. Estimates are already C_theta,j / theta.
Shading shows the unmodified estimate +/- 1.96 * standard_error: approximate
pointwise 95% Monte Carlo intervals, not simultaneous confidence bands.
The --simple version shows means only and writes counts_figure_simple.pdf/png,
preserving the original figure with intervals. Report uncertainty in its caption.
"""

import argparse
import csv
import json
import math
from pathlib import Path


PANELS = (
    ("A", "electron", "electron"),
    ("B", "electron", "photon"),
    ("C", "photon", "electron"),
    ("D", "photon", "photon"),
)
# Colour and line style both distinguish thresholds, including in greyscale.
STYLES = (("#0072B2", "-"), ("#D55E00", "--"), ("#009E73", "-."))
SIMPLE_STYLES = (("black", "-"), ("black", "--"), ("black", ":"))


def load_counts(csv_path, settings_path):
    """Read saved estimates and check that all panels match the settings."""
    with open(settings_path, encoding="utf-8") as source:
        settings = json.load(source)
    thresholds = settings["thresholds"]
    if (
        len(thresholds) != 3
        or len(set(thresholds)) != 3
        or any(not math.isfinite(theta) or not 0 < theta < 1 for theta in thresholds)
    ):
        raise ValueError("The figure requires three distinct thresholds between 0 and 1.")
    if set(settings["initial_types"]) != {"electron", "photon"}:
        raise ValueError("The figure requires both electron and photon initial types.")
    expected_times = sorted(settings["observation_times"])
    if len(set(expected_times)) < 2 or any(
        not math.isfinite(t) or t < 0 for t in expected_times
    ):
        raise ValueError("The figure requires at least two finite, nonnegative times.")

    series = {
        (initial_type, counted_type, theta): []
        for _, initial_type, counted_type in PANELS
        for theta in thresholds
    }
    with open(csv_path, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {
            "initial_type", "threshold", "track_length", "counted_type",
            "estimated_count", "standard_error",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("The CSV is missing columns needed for plotting.")
        for row in reader:
            key = row["initial_type"], row["counted_type"], float(row["threshold"])
            if key not in series:
                raise ValueError(f"CSV types or thresholds disagree with the settings: {key}")
            values = tuple(float(row[name]) for name in (
                "track_length", "estimated_count", "standard_error"
            ))
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError("CSV times, estimates and standard errors must be finite and nonnegative.")
            series[key].append(values)

    for key, rows in series.items():
        rows.sort(key=lambda row: row[0])
        if [row[0] for row in rows] != expected_times:
            raise ValueError(f"Missing or extra observations for {key}; CSV and settings must match.")
    return settings, series


def make_figure(settings, series, *, simple=False):
    """Build the count panels, optionally using a plain monochrome, means-only style."""
    try:
        import matplotlib
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Matplotlib is required. Activate .venv with 'source .venv/bin/activate', "
            "or install Matplotlib in your Python environment."
        ) from error
    matplotlib.use("Agg")  # Save files on machines with or without a display.
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
    }):
        figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True, sharey=True)
        figure.subplots_adjust(left=0.10, right=0.98, bottom=0.13 if simple else 0.15,
                               top=0.91 if simple else 0.84,
                               wspace=0.15 if simple else 0.12,
                               hspace=0.20 if simple else 0.30)
        styles = SIMPLE_STYLES if simple else STYLES
        lower_limit = 0.0
        upper_limit = 0.0
        for ax, (letter, initial_type, counted_type) in zip(axes.flat, PANELS):
            for theta, (colour, line_style) in zip(settings["thresholds"], styles):
                rows = series[initial_type, counted_type, theta]
                times = [row[0] for row in rows]
                means = [row[1] for row in rows]  # Already C_theta,j / theta.
                lower = [row[1] - 1.96 * row[2] for row in rows]
                upper = [row[1] + 1.96 * row[2] for row in rows]
                lower_limit = min(lower_limit, min(lower))
                upper_limit = max(upper_limit, max(upper))
                if not simple:
                    ax.fill_between(times, lower, upper, color=colour, alpha=0.14,
                                    linewidth=0, zorder=1)
                ax.plot(times, means, color=colour, linestyle=line_style,
                        linewidth=0.9 if simple else 1.5,
                        label=rf"$\theta={theta:g}$", zorder=2)
            if simple:
                ax.text(0.04, 0.94, f"({letter.lower()})", transform=ax.transAxes,
                        ha="left", va="top", fontsize=9)
                ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("0.3")
                    spine.set_linewidth(0.6)
                ax.tick_params(direction="out", labelsize=9, color="0.3",
                               width=0.6, length=3, top=False, right=False)
            else:
                ax.set_title(f"{letter}  Initially {initial_type}; count {counted_type}s",
                             loc="left", fontsize=9.5, pad=8)
                ax.set_axisbelow(True)
                ax.grid(axis="y", color="0.88", linewidth=0.6)
                ax.spines[["top", "right"]].set_visible(False)
                ax.spines[["left", "bottom"]].set_color("0.45")
                ax.tick_params(direction="out", labelsize=9, color="0.45")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))

        # Shared limits retain the entire interval, including any negative lower end.
        span = max(upper_limit - lower_limit, 1.0)
        bottom = lower_limit - 0.02 * span if lower_limit < 0 else 0.0
        axes[0, 0].set_ylim(bottom, upper_limit + 0.06 * span)
        axes[0, 0].set_xlim(min(settings["observation_times"]),
                            max(settings["observation_times"]))
        figure.supxlabel(r"Track length $\ell$", y=0.035 if simple else 0.065,
                         fontsize=10 if simple else 11)
        figure.supylabel("Expected count" if simple else "Expected number above threshold",
                         x=0.025 if simple else 0.018, fontsize=10 if simple else 11)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
                      ncol=3, frameon=False, fontsize=9 if simple else 10,
                      handlelength=3, columnspacing=2.5)
        if not simple:
            model = settings["model"]
            figure.text(
                0.5, 0.925,
                rf"$\mu={model['mu']:g},\ \gamma_0={model['gamma0']:g},\ "
                rf"\alpha={model['alpha']:g}$"
                f"    |    {settings['histories_per_initial_type']:,} histories per initial type",
                ha="center", fontsize=9, color="0.3",
            )
            figure.text(0.5, 0.012, "Shading: approximate pointwise 95% Monte Carlo intervals",
                        ha="center", va="bottom", fontsize=8, color="0.35")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simple", action="store_true",
                        help="save a plain monochrome version showing means only")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    settings, series = load_counts(directory / "counts.csv",
                                   directory / "counts_settings.json")
    figure = make_figure(settings, series, simple=args.simple)
    import matplotlib.pyplot as plt
    with plt.rc_context({"pdf.fonttype": 42}):
        for extension in ("pdf", "png"):
            stem = "counts_figure_simple" if args.simple else "counts_figure"
            filename = directory / f"{stem}.{extension}"
            figure.savefig(filename, dpi=300, facecolor="white")
            print(f"Saved {filename}")
    plt.close(figure)


if __name__ == "__main__":
    main()
