# Analytic validation report

From the source checkout, run:

```sh
uv sync --locked
uv run --locked spine-sim validate --config experiments/validation/config.json --output results/validation
```

The output directory must be new and beneath this checkout's `results/`.
This small study uses the existing sampler and compares it with independently
derived analytic identities. It produces a numerical report, not a plot or
an automatic proof/pass certificate. A large discrepancy warrants checking
the implementation, configuration and sampling uncertainty.

## Configuration and outputs

The tracked JSON specifies 6,000 samples per case, one root seed, all model
parameters and the following settings:

- Residual increments: electron track length `h=0.7`, `mu=1.3`; Laplace
  powers `s=1,2,4`, evaluated on the same 6,000 increments.
- Type chain: `mu=0.8`, `gamma0=1.4`, `alpha=0`, initial energy `1`, horizon
  `1.3`; separate 6,000-history ensembles for initial electrons and photons.
- One-split limit: `mu=alpha=0`, `gamma0=1.5`, initial energy `1`, horizon
  `0.8`, threshold `0.3`; 6,000 histories starting from photons, with both
  observed types scored on each history.

Thus the default run has 18,000 spine histories and 6,000 additional residual
increments. These are modest validation settings, not production history
counts. Editing the JSON creates a different study. `histories >= 2`, the
seed, and every setting must be explicit. Residual powers must be distinct
positive integers. The one-split reference requires `mu=alpha=0`; the runner
rejects other choices rather than silently applying the wrong formula.

The runner saves:

- `validation.csv`: one row per identity, with empirical and analytic means,
  empirical and theoretical Monte Carlo standard errors, their difference,
  and a standardized discrepancy.
- `config.json`: the exact JSON settings used.
- `metadata.json`: configuration, Python/dependency versions, RNG stream
  policy, Git revision and dirty state, runtime, commands and output hashes.
- `source.tar.gz`: actual source bytes, including uncommitted code, as
  described in the [reproduction guide](../../docs/reproducibility.md).

CSV fields `threshold` and `laplace_power` are empty where inapplicable.
`track_length` means electron track length for the residual test and evolution
horizon for the other two families. Type-chain means are probabilities of
being an electron. One-split means are **raw count** expectations, not
threshold-normalized counts. There is no jump cutoff or numerical time step.

## Analytic references and uncertainty

For the residual log-loss process, the independently integrated exponent at
positive integer `s` is

```text
psi(s) = mu * sum_{k=2}^{s+1} 1/k
M(s)   = E[exp(-s*S_h)] = exp(-h*psi(s)).
```

The tested random variable is `exp(-s*S_h)`, with variance
`M(2*s)-M(s)^2`. This reference uses the Lévy density, rather than another
implementation of the gamma-plus-thinned-Poisson algorithm.

For the type chain, writing `r=mu+gamma0`, the probability of an electron at
track length `t` is

```text
p_e(t) = gamma0/r + (p_e(0)-gamma0/r)*exp(-r*t),    r > 0.
```

Here `p_e(0)` is one for an initial electron and zero for an initial photon.
If both rates vanish it remains `p_e(0)`. The scored variable is the electron
indicator, whose variance is `p_e(t)*(1-p_e(t))`. Energy loss does not change
this chain's rates.

The one-split limit independently considers the original particle population:
an initial photon splits once at rate `gamma0` into two electrons with energy
fractions `U` and `1-U`; the electrons do not evolve because `mu=alpha=0`.
At threshold `theta`, let `p=1-exp(-gamma0*t)` and `q=exp(-gamma0*t)`. Its
expected raw counts are

```text
electron: 2*(1-theta)*p
photon:   q.
```

These means are estimated with the spine's raw count payoffs. Their
theoretical **spine-payoff** variances, used for Monte Carlo uncertainty, are

```text
electron: -2*log(theta)*p - (2*(1-theta)*p)^2
photon:   q*p.
```

For example, the electron second moment follows by integrating `1/v^2`
against density `2v` over `theta < v < 1`, then multiplying by the split
probability. These variances are not the variances of full-cascade counts.
Further derivations and the scope of the model are in
[the model documentation](../../docs/model.md).

`empirical_mc_se` is `sqrt(sample_variance/n)` with `ddof=1`.
`theoretical_mc_se` is `sqrt(analytic_variance/n)` for the sampled observable.
The standardized discrepancy is
`(empirical_mean-analytic_expectation)/theoretical_mc_se`. It is left empty
if the theoretical variance is zero, since that ratio is undefined; the
ordinary difference is still reported. Analytic variances that round slightly
below zero are clamped to zero for the square root.

Residual powers share increments, and the two one-split count estimates share
spines, so their errors are correlated. The standardized values are descriptive
checks, not simultaneous confidence statements. Small samples can have zero
empirical variance while the theoretical variance is positive. Numerical
agreement does not prove the many-to-one theorem or validate additional models;
the unresolved mathematical TODOs in `docs/model.md` still apply.

## Streams and replay

Every sample uses PCG64 with

```text
SeedSequence(seed, spawn_key=(2, family_code, initial_type_code, history_index))
```

Family codes are `0` for residual increments, `1` for the type chain, and `2`
for the one-split limit. Initial-type codes are electron `0` and photon `1`;
the residual case uses `0`. History indices start at zero. The leading `2`
reserves a namespace distinct from the ordinary simulator's leading
initial-type codes `0` and `1`. Sample identity is independent of scheduling,
although this implementation runs serially. Reusing a validation seed reuses
streams; use a different seed for a separately randomized validation study.

To replay archived sources, extract `source.tar.gz`, copy the saved
`config.json` into that extracted source directory as `run-config.json`, and
run there:

```sh
uv run --locked spine-sim validate --config run-config.json --output results/reproduced-validation
```

Keep the recorded environment. The CSV should reproduce exactly in that
environment; wall-clock runtimes, invocation paths and provenance metadata
will differ. Neither the original run nor replay overwrites an existing
output directory.
