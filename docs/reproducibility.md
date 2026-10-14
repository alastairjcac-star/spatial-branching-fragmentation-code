# Reproducing an experiment

Run commands from the repository root. Python is pinned in `.python-version`
and package versions are locked in `uv.lock`.

```sh
uv python install
uv sync --locked
uv run --locked spine-sim --config examples/reference.json --output results/reference-demo
```

The output directory must be a new subdirectory of `results/`. The CLI requires
an editable installation and runs from that source root so its snapshot matches
the imported code. To rerun in the same checkout, choose a new output directory
while keeping the configuration unchanged.
`results/` is ignored by Git, so running an example does not add generated data
to the public source tree.

## Random streams

Every history uses an explicit `numpy.random.Generator` with the `PCG64` bit
generator. For history index `a`, starting at zero, construct

```python
seed_sequence = numpy.random.SeedSequence(seed, spawn_key=(initial_type_code, a))
rng = numpy.random.Generator(numpy.random.PCG64(seed_sequence))
```

The initial type code is `0` for electron and `1` for photon. This separates
streams across histories and initial types and makes a history's stream
independent of traversal order or batch size. The implementation runs histories
serially. A future parallel implementation must preserve the mapping and
document how estimates are reduced; floating-point sums can depend on reduction
order.

Observations and thresholds within one history reuse that history, so their
payoffs are correlated. Changing the observation grid changes how increments
are split and random numbers are consumed; it does not preserve the same
realized path for the same seed. The law remains the intended law. Fixed seeds
are reproducibility tools, not evidence that different experiments are
statistically independent when stream identities overlap.

The exact NumPy version matters: distribution sampling and floating-point
results need not match across versions or platforms. The lock file and recorded
environment support reproducing a run; bitwise agreement on arbitrary hardware
is not promised.

## Saved records

| File | Contents |
| --- | --- |
| `config.json` | Effective parameters, initial types, thresholds, observation grid, history count, and seed |
| `estimates.csv` | Pointwise normalized/raw count estimates, Monte Carlo errors, sample counts, nonzero counts, and sufficient statistics |
| `metadata.json` | Runtime, Python/dependency versions, RNG policy, Git revision/dirty state, source hashes, and exact reproduction command |
| `counts.png`, `counts.pdf` | Four-panel demonstration figure |
| `source.tar.gz` | Source snapshot used for the run, including relevant uncommitted and untracked source |

The snapshot captures public source files, configuration, and dependency
specifications while excluding Git internals, ignored local environments, and
generated outputs. Its file manifest and archive checksum are recorded with
the run. A dirty Git revision alone cannot reproduce an experiment; the
snapshot preserves the actual source used.

## Replay a source snapshot

Extract the run's `source.tar.gz` into a new, empty directory and copy the
saved effective configuration there. For example, substituting the absolute
path to your run:

```sh
mkdir spine-replay
tar -xzf /absolute/path/to/results/reference-demo/source.tar.gz -C spine-replay
cp /absolute/path/to/results/reference-demo/config.json spine-replay/run-config.json
cd spine-replay
uv python install
uv sync --locked
uv run --locked spine-sim --config run-config.json --output results/reproduced-demo
```

The `snapshot_reproduction_command` recorded in `metadata.json` uses this
copied configuration, including when the original configuration was supplied
from outside the repository. The `reproduction_command` instead records the
original configuration path. Preserve the run's saved `config.json` alongside
the archive. A replay directory need not have Git metadata: its
outputs will record that absence, while the original metadata retains the
original revision and dirty state.

Compare `estimates.csv` for numerical reproduction. Runtime, creation times,
paths, and provenance metadata can differ even when estimates agree; plot
files can also contain format-specific metadata. Source snapshots preserve
code, not an entire operating system image.

## Demonstration versus a numerical study

The reference run uses 2,000 histories for each initial type, seed `20261014`,
relative thresholds `0.01`, `0.03`, `0.1`, and observations from `0` to `6` in
increments of `0.25`. Rates are `mu=gamma0=1`, continuous loss is `alpha=0`,
and initial energy is `1`. Track length is measured in units for which
`mu=1`; it is not penetration depth.

These settings are intended to verify the end-to-end workflow and illustrate
the count curves. A precision study must choose history counts based on
uncertainty, inspect nonzero-payoff counts, and investigate numerical limits
for its parameter range. The observation spacing controls plot resolution,
not a discretisation of the stochastic evolution. Any reported maximum from
these data is a maximum on the observation grid.

The raw count intervals are approximate pointwise normal intervals. Their
coverage can be poor for rare payoffs; all-zero records should not be read as
proof of a zero expected count. Shared-history curves are correlated. Neither
the figure nor its intervals establish the many-to-one identity or provide
simultaneous coverage across the curves.
