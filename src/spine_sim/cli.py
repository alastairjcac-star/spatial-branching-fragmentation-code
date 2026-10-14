"""Run a configured experiment, plot saved results, or inspect validation."""

import argparse
import sys
from pathlib import Path

from .workflow import run_experiment as run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    simulate = commands.add_parser("simulate", help="Save independent-spine estimates")
    validate = commands.add_parser(
        "validate", help="Compare samples with analytic laws"
    )
    for command in (simulate, validate):
        command.add_argument(
            "--config", type=Path, required=True, help="Experiment JSON"
        )
        command.add_argument(
            "--output", type=Path, required=True, help="New results/ directory"
        )
    plot = commands.add_parser("plot", help="Plot saved estimates without simulating")
    plot.add_argument(
        "--input", type=Path, required=True, help="Saved simulation directory"
    )
    plot.add_argument(
        "--output", type=Path, help="New figure directory; default INPUT/figures"
    )
    plot.add_argument("--title", default="Spine count estimates", help="Figure title")
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"--config", "--output"}:
        arguments.insert(0, "simulate")
        print(
            "Legacy invocation saves data only; use 'spine-sim plot' for figures.",
            file=sys.stderr,
        )
    args = parser.parse_args(arguments)
    try:
        if args.command == "simulate":
            info = run(args.config, args.output)
            print(f"Saved {info['history_count_total']:,} histories to {args.output}")
            print(f"Simulation runtime: {info['simulation_seconds']:.3f} seconds")
        elif args.command == "plot":
            from .plotting import plot_saved_counts

            info = plot_saved_counts(args.input, args.output, args.title)
            print(f"Saved figures to {info['output_directory']}")
        else:
            from .validation import run_validation

            info = run_validation(args.config, args.output)
            print(
                f"Saved {info['validation_rows']} analytic comparisons to {args.output}"
            )
    except (ValueError, OSError, FloatingPointError) as error:
        parser.exit(2, f"spine-sim: {error}\n")


if __name__ == "__main__":
    main()
