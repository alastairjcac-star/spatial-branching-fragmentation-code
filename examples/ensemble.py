"""A small ensemble workflow with complete saved-run provenance.

Run from the checkout root:
    uv run --locked python examples/ensemble.py --output results/ensemble-example

See workflow.estimate_counts for the short independent-history loop and online
count accumulation used below. The default uses 100 histories per initial type.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

from spine_sim.config import RunConfig
from spine_sim.io import load_estimates
from spine_sim.provenance import write_json
from spine_sim.workflow import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/ensemble-example"))
    args = parser.parse_args()
    args.output = args.output.resolve()
    results = Path.cwd().resolve() / "results"
    if args.output == results or not args.output.is_relative_to(results):
        parser.error("--output must be a new subdirectory of results/")

    # 1. Parameters: a modest API demonstration, using both initial types.
    config = RunConfig.from_dict(
        {
            "parameters": {
                "mu": 1.0,
                "gamma0": 1.0,
                "alpha": 0.0,
                "initial_energy": 1.0,
            },
            "initial_types": ["electron", "photon"],
            "observation_times": [0.0, 0.5, 1.0, 2.0],
            "thresholds": [0.01, 0.03, 0.1],
            "histories": 100,
            "seed": 20261015,
            "description": "Small ensemble API demonstration; not production settings.",
        }
    )
    # Keep the effective input outside the run directory, which must be new.
    # It is generated under ignored results/ and also saved inside the run.
    config_path = args.output.with_name(args.output.name + "-input.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists() or args.output.exists():
        raise FileExistsError("choose a new --output directory for this example")
    write_json(config_path, asdict(config))

    # 2. Independent histories → 3. strict-threshold estimates.
    # run_experiment calls estimate_counts, then saves the configuration,
    # estimates, RNG policy, environment, and source snapshot together.
    run_experiment(config_path, args.output)

    # 4. Saved results are ordinary typed rows, ready for analysis or plotting.
    rows = load_estimates(args.output / "estimates.csv")
    for row in rows:
        if row["track_length"] == config.observation_times[-1]:
            print(
                f"{row['initial_type']} → {row['observed_type']}, "
                f"theta={row['threshold']:g}: "
                f"count={row['raw_count_mean']:.4f}, SE={row['raw_count_se']:.4f}"
            )
    print(f"Saved complete run to {args.output}; use spine-sim plot to draw it.")


if __name__ == "__main__":
    main()
