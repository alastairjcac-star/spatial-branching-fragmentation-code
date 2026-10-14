# spatial-branching-fragmentation-code
Reproducible simulation code for spatial branching-fragmentation processes

This Python package samples the energy-biased electron–photon spine in a
homogeneous medium and estimates cascade first moments from independent spine
histories. The implemented model has `k(u) = mu/u`, `q(u) = 1`, constant photon
conversion rate `gamma0`, and proportional electron energy loss
`beta(energy) = alpha * energy`.

This is an initial research implementation. It simulates energy and type along
track length; it does not construct the branching population or simulate spatial
motion. The interpretation as cascade counts assumes the supplied many-to-one
identity. See [the model specification](docs/model.md) for assumptions,
identities, and unresolved mathematical questions.

## Install and check

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
these commands from the repository root:

```sh
uv python install
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
uv build --no-sources
```

`.python-version` pins Python 3.14.4; `uv.lock` records dependency versions.
NumPy supplies arrays and random sampling. Matplotlib produces example plots.
pytest and Ruff are development dependencies for checks and formatting. The
package uses the standard library for configuration and run metadata.
Hatchling is the pinned build backend for source distributions and wheels.

## Simulate, then plot

```sh
uv run --locked spine-sim simulate --config experiments/reference/config.json --output results/refactored-demo
uv run --locked spine-sim plot --input results/refactored-demo --output results/refactored-demo/figures --title 'Demonstration output'
```

Run the CLI from its editable source checkout so it can preserve the source
used. Each output directory must be new and lie under `results/`.

`simulate` saves estimates, configuration, metadata, and a source snapshot.
`plot` reads those saved files, checks their recorded hashes, and generates
figures without sampling any new histories. It defaults to an input run's
`figures/` subdirectory. The reference experiment uses:

| Setting | Demonstration value |
| --- | --- |
| `mu`, `gamma0`, `alpha` | `1`, `1`, `0` |
| Initial energy | `1` |
| Initial types | Electron and photon |
| Independent histories | 2,000 per initial type |
| Observation track lengths | 0 through 6, spaced by 0.25 |
| Relative energy thresholds | 0.01, 0.03, 0.1 |
| Seed | 20261014 |

These are modest demonstration settings, not a precision or production study.
The observation spacing specifies where results are requested; it is not an
integration time step. The sampler introduces no jump cutoff.

The simulation directory contains `estimates.csv`, the effective `config.json`,
`metadata.json`, and `source.tar.gz`. The separate figure directory contains
`counts.png`, `counts.pdf`, `plot-metadata.json`, and its own `source.tar.gz`
for the plotting code. The four-panel figure shows estimated raw expected
counts: rows select the initial type and columns the counted type. Curves
distinguish thresholds. The supplied title identifies this run as demonstration
output.

Both normalized and raw estimates and their Monte Carlo standard errors are
saved. Raw counts and their standard errors equal the normalized values divided
by the threshold. Approximate pointwise intervals describe Monte Carlo
uncertainty only. Curves evaluated from the same histories are correlated, and
the intervals are not simultaneous confidence bands. Few nonzero payoffs can
make the normal approximation unreliable; an all-zero sample does not establish
that the expected payoff is zero.

To change the title or plotting code, render the saved estimates into a new
figure directory:

```sh
uv run --locked spine-sim plot --input results/refactored-demo --output results/refactored-demo/figures-revised --title 'Reference model — demonstration'
```

Each simulation records its reproduction command, random streams, environment,
Git state, and source snapshot. Plot metadata records the input hashes and the
source used to render the figures. See the [reference experiment guide](experiments/reference/README.md)
and [reproduction instructions](docs/reproducibility.md).

## Analytical validation and API examples

```sh
uv run --locked spine-sim validate --config experiments/validation/config.json --output results/validation
uv run --locked python examples/single_spine.py
uv run --locked python examples/ensemble.py
```

The validation command saves a reproducible comparison with analytical
quantities, including Monte Carlo uncertainty. It supplements the automated
tests; numerical agreement is not a proof. See the
[validation experiment](experiments/validation/README.md) for its settings and
interpretation.

The [API examples](examples/README.md) show one path and an ensemble. Use
[named experiments](experiments/README.md) for saved research runs.

The earlier command `spine-sim --config ... --output ...` remains an alias
for `simulate`, and `examples/reference.json` remains a compatibility copy of
the reference configuration. That command now saves simulation data only;
use `spine-sim plot` to produce figures.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/spine_sim/` | Reusable simulation, estimation, and command-line code |
| `tests/` | Mathematical, sampling, event-loop, and implementation checks |
| `experiments/` | Named studies with tracked configurations and run instructions |
| `examples/` | Short reusable-API examples and the legacy reference configuration |
| `docs/` | Model, numerical conventions, and reproducibility |
| `results/` | Generated local outputs, ignored by Git |

The gamma-plus-thinned-Poisson construction samples the finite-observation law
of this specific model in ideal arithmetic. It does not enumerate its infinitely
many small jumps. Floating-point arithmetic and random-variate algorithms have
separate limitations; see [numerical conventions](docs/numerics.md). Passing
simulation tests is not a proof of the cascade identity or convergence for other
models.

The separation of model code, experiment drivers, postprocessing, and
analytical checks is inspired by the organisation of
[proton-beam-sde](https://github.com/JereKoskela/proton-beam-sde). This package's
implementation is original; its energy/type model and gamma-plus-thinned-Poisson
sampler are unchanged by that workflow choice.

## Licence

TODO: select a licence and identify the copyright holder before distributing this
package under an open-source licence. No licence has been selected by this
scaffold.
