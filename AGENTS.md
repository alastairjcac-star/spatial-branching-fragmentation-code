# Agent instructions

This repository contains public simulation code for an early-stage mathematical
research project. Keep changes conservative, mathematically honest, and
reproducible. Preserve existing user work.

## Scope and privacy

The current milestone implements the homogeneous energy/type spine and estimates
cascade first moments. It does not simulate spatial motion or a full branching
population. Keep private manuscripts, editorial notes, attachments, credentials,
and private repository history out of this repository. Describe mathematics in
self-contained public documentation.

Do not select a licence or copyright holder without the owner's instruction.
Preserve an existing licence. Until one is chosen, retain the explicit README
TODOs.

## Layout and dependencies

- `src/spine_sim/`: reusable implementation and command-line interface.
- `tests/`: meaningful mathematical and implementation checks.
- `experiments/`: named studies with tracked configurations and run instructions.
- `examples/`: small API examples; retain the legacy reference configuration.
- `docs/`: model specification, numerical conventions, and reproduction.
- `results/`: generated outputs, ignored by Git.

Use the pinned Python version, `pyproject.toml`, and `uv.lock`. Keep runtime
dependencies minimal: NumPy for sampling and arrays; Matplotlib for plots.
Development checks use pytest and Ruff. Update the lock file using uv when
dependencies change; never hand-edit generated lock files.

## Mathematical and numerical honesty

Distinguish model assumptions, identities, derivations, implementation choices,
numerical approximations, and empirical observations. State the model, initial
conditions, track-length variable, rate conventions, and payoff definitions.
Preserve unresolved questions as TODOs. Do not invent theorems, references,
numbered results, or claims about Harris's equations.

Infinite activity is not an ordinary finite-rate compound Poisson process.
This model uses a gamma increment plus a finite residual process. Do not add a
jump cutoff, compensation, drift correction, or numerical time step silently.
Any future approximation must state its definition, units, retained intensity,
omitted-event treatment, and justification. Keep Monte Carlo uncertainty
separate from truncation and discretisation error.

Preserve log-energy and log-weight calculations, strict threshold comparison,
the pending switching clock across observations, and the documented convention
for coincident switches and observations. Validate NumPy scale versus
mathematical rate parameters. Describe exactness only for the finite-observation
law of this particular model in ideal arithmetic, with numerical limitations
stated separately.

Keep simulation and postprocessing separate. `simulate` writes estimates and
provenance; `plot` reads those saved results without drawing histories or
changing input files. Analytical validation reports supplement automated
tests. Preserve the numerical core and stream policy during workflow refactors
unless a mathematical change is explicitly requested and justified.

## Reproducibility and checks

Every experiment must save model parameters, observation grid, initial types,
thresholds, history counts, seed and RNG stream policy, Python and dependency
versions, Git revision and dirty state, runtime, and the reproduction command.
Retain enough source to reproduce dirty runs. Keep sufficient statistics and
nonzero-payoff counts for uncertainty assessment. Raw count estimates and
standard errors both divide normalized values by the threshold.

Use an explicit NumPy `Generator`. Keep streams independent across histories
and initial types. Record changes to stream mapping. Generated results and
local environments remain ignored; do not hand-edit or commit generated output
unless explicitly requested. Do not copy private data into run snapshots.

Record plotting input hashes, settings, environment, and source separately
from simulation provenance, so revised figures can be reproduced from existing
estimates. Require new output directories rather than overwriting run records.
When updating experiment configurations, identify whether changes alter the
model, sampling, or presentation. Keep the compatibility reference configuration
consistent with `experiments/reference/config.json`.

Add substantive checks for simulation changes: invariants, limiting cases,
analytic sampling identities, event scheduling, and estimator behaviour.
Seed statistical checks and use justified tolerances. Numerical agreement is
not a proof. After relevant changes run:

```sh
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

For packaging or experiment changes also run `uv build --no-sources` and the
separate simulate, plot, and validation commands documented in README, using
new output directories. Check that plotting does not invoke simulation and
that an unchanged seeded configuration preserves its estimates. Report unresolved
failures and numerical limitations. Do not push commits, change repository
visibility, or publish packages without explicit authorization.
