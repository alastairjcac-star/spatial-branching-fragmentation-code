# Numerical conventions

The model and payoff identities are specified in [model.md](model.md). This
document describes how the implementation evaluates them.

## State and event ordering

The simulator stores `log_energy = log(v)`, where `v` is energy divided by the
initial energy, and `log_weight = log(W)`. The initial state has both logs zero.
It also records the current type and accumulated electron track length.
Only relative thresholds `0 < theta < 1` are used by the count estimator;
their physical energy values are `theta * initial_energy`.

Each type has a pending absolute switching time. The loop advances to the
earlier of that time and the next observation. At an observation it retains the
pending switch; it draws the next waiting time only after switching. An exact
tie processes the switch first and records the state after the switch
(right-continuous observation convention). Splitting an electron interval at
observation times uses independent subordinator increments, as required by its
law.

For an interval of length `h`, an electron loses `S_h + alpha*h` in log energy
and `alpha*h` in log weight. A photon changes neither log. An electron-to-photon
switch adds `log(U)` to log energy; a photon-to-electron switch adds
`0.5*log(U)`. Switching does not change the weight. Zero rate means no switch
from that type. Zero-length intervals have zero loss and require no increment
draw. If a positive waiting time cannot advance floating-point track length,
the implementation reports the numerical failure instead of inventing a time
step.

## Increment sampling and parameter conventions

For an electron interval, the sampler draws independent quantities:

| Quantity | Mathematical parameterization | NumPy parameterization |
| --- | --- | --- |
| Gamma increment | Shape `mu*h`, rate `2` | `gamma(shape=mu*h, scale=0.5)` |
| Candidate count | Poisson mean `mu*h/2` | `poisson(lam=mu*h/2)` |
| Candidate jump | Exponential rate `2` | `exponential(scale=0.5)` |
| Electron waiting time | Exponential rate `mu` | `exponential(scale=1/mu)` |
| Photon waiting time | Exponential rate `gamma0` | `exponential(scale=1/gamma0)` |

Accept a candidate `z` if an independent uniform value is smaller than

```text
B(z) = 1 / (1 - exp(-z)) - 1/z.
```

The gamma component samples the aggregate of infinitely many small losses;
the accepted Poisson candidates sample the finite residual. No losses are
discarded according to size, and there is no jump cutoff or compensation term.

Direct evaluation of `B` is unstable near zero because it subtracts two large
terms. For `0 <= z < 0.01`, the implementation uses the polynomial

```text
B(z) approximately 1/2 + z/12 - z^3/720 + z^5/30240 - z^7/1209600.
```

and `1/(-expm1(-z)) - 1/z` away from zero. The limiting value is `B(0)=1/2`.
The branch boundary and series order are numerical evaluation choices, checked
against high-precision arithmetic in the tests. That boundary does not remove
any jumps. The function lies between `1/2` and `1` for positive finite `z` in
exact arithmetic.

NumPy uniform draws use `[0,1)`. Uniform endpoints zero and one are resampled
wherever a logarithm is required, giving an open interval for switch fractions
(only zero can arise from NumPy's uniform generator). The
thinning comparison is strict, `U < B(z)`; a zero acceptance uniform is valid.
Zero exponential values are resampled when strict positivity is required,
including candidate jump sizes. Endpoint retries are bounded at 1,024 attempts;
exhaustion raises an error.
These endpoint conventions avoid `log(0)` and zero-time event loops without
silently replacing a draw by an arbitrary epsilon.

## Thresholds and uncertainty

The normalized count payoff for observed type `j` is

```text
theta * W/v * indicator(v > theta and type == j).
```

The code compares `log_energy > log(theta)` before evaluating the exponential
`exp(log(theta) + log_weight - log_energy)`. Equality is excluded. This avoids
forming a reciprocal of an underflowed energy. For nonnegative loss rates the
payoff lies in `[0,1]`; `alpha=0` gives `W=1`.

Independent histories supply the sample mean and sample variance with
denominator `n-1`. The standard error is the sample standard deviation divided
by `sqrt(n)`. Both the mean and standard error are divided by `theta` to report
raw expected counts. The saved sums, squared sums, nonzero counts, and variance
support auditing the estimates; a stable online variance calculation avoids
subtracting nearly equal large accumulated quantities.

Normal intervals are approximate, pointwise, and untrimmed. They may have
negative lower endpoints even though true expected counts are nonnegative.
Rare or all-zero payoffs make their approximation unreliable. In particular,
an observed standard error of zero does not prove the expectation is zero.
Different times, thresholds, and observed types reuse a history; their
estimates are correlated. The saved marginal statistics do not specify that
cross-covariance and the displayed intervals are not simultaneous bands.

## Meaning and limits of exactness

In ideal arithmetic with exact random variates, this construction samples the
joint law at finitely many observation times for the specified homogeneous
model. It does not resolve individual infinite-activity jump times or provide
an exact continuous path between observations. Lines in plots only connect
observations.

The implementation uses finite-precision floating-point arithmetic and NumPy
random-variate routines. Very small gamma shapes can yield a rounded or
underflowed zero; extreme parameters can exceed a distribution routine's range
or prevent a time increment from being representable. Logarithms avoid many
energy underflows but do not make arithmetic exact. The stable `B` series is a
numerical approximation to a function, not a truncation of the jump process.
The tests examine ordinary and selected limiting cases, not every extreme
parameter combination. A finite log payoff can still exponentiate to an
underflowed zero. Duplicate observation times are allowed and record the same
state without consuming interval draws.

Monte Carlo standard errors quantify sampling uncertainty only. They do not
include floating-point error, uncertainty about model assumptions, or error in
an unverified cascade representation. No convergence or precision claim for
other kernels, spatial models, or production settings follows from this demo.
