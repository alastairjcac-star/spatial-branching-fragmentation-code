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

## Run the demonstration

```sh
uv run --locked spine-sim --config examples/reference.json --output results/reference-demo
```

The output directory must be a new subdirectory of `results/`. Run the CLI
from its editable source checkout so it can preserve the source used. The
tracked configuration uses:

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

For a short example of the reusable Python API, run
`uv run --locked python examples/single_spine.py` to print one path per initial
type, including log energy and log weight.

The run saves `estimates.csv`, the effective `config.json`, `metadata.json`,
`counts.png`, `counts.pdf`, and a `source.tar.gz` snapshot under `results/`.
The four-panel figure shows estimated raw expected counts: rows select the
initial type and columns the counted type. Curves distinguish thresholds; all
panels are labelled as demonstration output.

Both normalized and raw estimates and their Monte Carlo standard errors are
saved. Raw counts and their standard errors equal the normalized values divided
by the threshold. Approximate pointwise intervals describe Monte Carlo
uncertainty only. Curves evaluated from the same histories are correlated, and
the intervals are not simultaneous confidence bands. Few nonzero payoffs can
make the normal approximation unreliable; an all-zero sample does not establish
that the expected payoff is zero.

For an independent output directory using the same source and configuration:

```sh
uv run --locked spine-sim --config examples/reference.json --output results/reference-demo-repeat
```

Each run records its reproduction command, random streams, environment, Git
state, and source snapshot. See [reproduction instructions](docs/reproducibility.md)
for restoring a run made from uncommitted source.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/spine_sim/` | Reusable simulation, estimation, and command-line code |
| `tests/` | Mathematical, sampling, event-loop, and implementation checks |
| `examples/` | Small tracked experiment configurations |
| `docs/` | Model, numerical conventions, and reproducibility |
| `results/` | Generated local outputs, ignored by Git |

The gamma-plus-thinned-Poisson construction samples the finite-observation law
of this specific model in ideal arithmetic. It does not enumerate its infinitely
many small jumps. Floating-point arithmetic and random-variate algorithms have
separate limitations; see [numerical conventions](docs/numerics.md). Passing
simulation tests is not a proof of the cascade identity or convergence for other
models.

## Licence

TODO: select a licence and identify the copyright holder before distributing this
package under an open-source licence. No licence has been selected by this
scaffold.
