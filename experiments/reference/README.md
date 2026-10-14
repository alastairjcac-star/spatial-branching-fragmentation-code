# Reference count demonstration

This experiment estimates first moments from independent energy-biased spines
in the homogeneous model. It does not simulate the branching population or
spatial motion. See [the model specification](../../docs/model.md).

`config.json` uses `mu=gamma0=1`, `alpha=0`, initial energy `1`, 2,000 histories
for each initial type, seed `20261014`, and relative thresholds `0.01`, `0.03`,
and `0.1`. The observations run from track length `0` to `6`, inclusive, in
increments of `0.25`. That spacing selects outputs, not numerical integration
steps. There is no jump cutoff. These are modest demonstration settings,
not production settings.

From the repository root:

```sh
uv sync --locked
uv run --locked spine-sim simulate --config experiments/reference/config.json --output results/refactored-demo
uv run --locked spine-sim plot --input results/refactored-demo --output results/refactored-demo/figures --title 'Demonstration output'
```

Both output directories must be new. Simulation saves `config.json`,
`estimates.csv`, `metadata.json`, and `source.tar.gz`. Plotting reads these
saved data, verifies their hashes, and saves `counts.png`, `counts.pdf`,
`plot-metadata.json`, and a separate source snapshot in `figures/`.

Rows in the four-panel figure select the initial type; columns select the
counted type. The three curves use the three thresholds. Counts and standard
errors are both normalized payoff values divided by the threshold. Shading
shows approximate pointwise 95% Monte Carlo intervals; correlated curves do
not have simultaneous coverage. Rare and all-zero payoffs require particular
care. The CSV retains nonzero counts and sufficient statistics for auditing.

To revise a title or plotting implementation without sampling again:

```sh
uv run --locked spine-sim plot --input results/refactored-demo --output results/refactored-demo/figures-revised --title 'Reference model — demonstration'
```

To repeat the simulation, keep the tracked configuration and choose a new
simulation output path. The fixed stream mapping preserves estimates for an
unchanged configuration and environment. See
[reproducibility](../../docs/reproducibility.md) for source-snapshot replay.

`examples/reference.json` is preserved as a compatibility copy with the same
values. New run instructions use this experiment directory.
