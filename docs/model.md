# Model and estimator

This package implements a particular homogeneous electron–photon **spine**.
One simulated path follows one energy-biased lineage. It is not a branching
population, and its fluctuation law is not the fluctuation law of that
population. Only energy, particle type and a continuous-loss weight are
recorded. There are no spatial boundaries, scattering or absorption in this
implementation.

## State and parameters

The evolution variable `t` is track length. Let the initial energy be
`E0 > 0`, let `v = E/E0` denote relative energy, and let `J` be either
`electron` or `photon`. The stored energy field is `log_energy = log(v)`,
not `log(E)`. Initially `v = W = 1` and the initial type is supplied by the
caller. The extra recorded quantity `L_e(t)` is the total electron track
length up to `t`.

The model uses the relative fragmentation kernel `k(u) = mu/u`, the uniform
splitting density `q(u) = 1`, the photon rate `gamma0`, and the electron
continuous-loss coefficient `alpha*E`. The parameters `mu`, `gamma0` and
`alpha` are finite and nonnegative, with units of inverse track length.
The code also accepts zero rates to make the limiting cases explicit.
The default reference case is `mu = gamma0 = E0 = 1`, `alpha = 0`.

The spine is defined by the following independent mechanisms, conditional on
the present type:

- An electron switches to a photon at rate `mu`. Its energy is multiplied
  by an independent uniform variable `U` in `(0,1)`.
- A photon switches to an electron at rate `gamma0`. Its energy is
  multiplied by `sqrt(U)`, whose density on `(0,1)` is `2v`.
- During electron track length, relative energy suffers the residual
  log-loss process described below and the continuous loss `alpha`.
  There is no energy change between switches in the photon state.

If an electron interval has length `h`, its updates are

```text
log(v) <- log(v) - S_h - alpha*h
log(W) <- log(W) - alpha*h.
```

Thus `W(t) = exp(-alpha*L_e(t))`. It is identically one when `alpha = 0`.
The weights average over additional killing; this code does not introduce
a random killing clock.

## Infinite residual activity and its increment law

For an electron, change relative loss fraction `u` to log loss
`z = -log(1-u)`. Its Lévy intensity per unit electron track length is

```text
Pi(z) dz = mu * exp(-2*z) / (1-exp(-z)) dz,    z > 0.
```

For `mu > 0`, this density behaves as `mu/z` near zero, so its total mass
is infinite. Nevertheless,

```text
integral_0^infinity min(1,z) Pi(z) dz < infinity.
```

Indeed the integrand is bounded near zero, and the tail decays
exponentially. This is a nonnegative finite-variation jump process. We
choose zero deterministic drift and sum its jumps without compensation.
There is no subtraction of small jumps or drift correction.

The identity

```text
Pi(z) = mu*exp(-2*z)/z + mu*exp(-2*z)*B(z)
B(z)  = 1/(1-exp(-z)) - 1/z
```

separates a gamma subordinator from a finite-intensity remainder.
For `z > 0`, `0 < B(z) < 1`: the lower inequality follows from
`1-exp(-z) < z`; the upper one from `exp(z)-1 > z`.
Consequently the remainder is dominated by `mu*exp(-2*z) dz`, whose
total intensity is `mu/2`.

An independent residual increment over length `h` can therefore be sampled
as follows:

1. Draw `G` with gamma shape `mu*h` and **rate** `2`.
2. Draw `K` with Poisson mean `mu*h/2`.
3. For each of these `K` candidates, draw `Z` with exponential rate `2`
   and an independent uniform `V`; retain `Z` if `V < B(Z)`.
4. Return `G` plus the sum of retained candidates.

NumPy's scale convention requires gamma `scale=0.5` and candidate
exponential `scale=0.5`. A zero interval or `mu=0` returns zero directly.
Only the finite remainder is sampled as individual jumps. The gamma draw
aggregates the infinite-activity component; this is not a finite-rate
approximation to the whole residual process.

The finite-time spine is well defined under the stated parameter choices.
Switching intensities are bounded by `max(mu,gamma0)`, so only finitely many
switches occur on a finite track-length interval, by domination with a
constant-rate Poisson process. Residual accumulated loss is finite almost
surely on such an interval. One direct integrability check is

```text
E[S_h] = mu*h * sum_{k=2}^infinity 1/k^2
       = mu*h * (pi^2/6 - 1).
```

This follows by expanding `Pi(z) = mu*sum_{k=2}^infinity exp(-k*z)` and
integrating `z*Pi(z)`. The nonnegative sum permits interchange of the sum
and integral. Each switch fraction is strictly positive, so the ideal
spine retains positive energy at every finite track length. Zero energy
is not an artificial absorbing boundary in the simulator. These facts
concern this explicitly constructed spine; they do not supply an existence
or uniqueness theorem for a general branching-fragmentation cascade.

## Observation lengths and exactness

The simulator advances to the earlier of a scheduled switch and the next
requested observation. Residual loss is sampled for the elapsed electron
interval only. An observation does not replace the pending switch time.
At a switch, the energy fraction is applied, the type is changed, and a
new waiting time is drawn. Zero switching rate means an infinite waiting
time. Coincident switches are processed before observations, so the stored
state follows the right-continuous convention; repeated observation lengths
do not advance the process.

Independent residual increments and the constant switching rates give the
specified joint law at finitely many observation lengths in ideal
arithmetic. The method has no jump cutoff and no time integration step.
Changing the observation grid changes which states are requested, and can
also change the particular seeded realization because random draws are
consumed differently. The method neither enumerates all residual jumps nor
provides a general-purpose exact simulator for other kernels. Floating-point
and random-variate limitations are described in the numerical documentation.

## Threshold counts

Let `0 < theta < 1`. The strict physical threshold is `E > theta*E0`.
For observed type `j`, one spine contributes

```text
Y(theta,j,t) = theta * W(t)/v(t) * 1{v(t)>theta, J(t)=j}.
```

The mathematical input used for interpreting this observable is the
energy-biased many-to-one identity: the expectation of `Y` under the
specified spine equals `theta` times the expected number of cascade
particles of type `j` above threshold. The package adopts that identity for
this model. It does not establish the underlying full-cascade theory or a
correspondence with a particular external theorem or book equation.

**Mathematical TODO:** independently verify the assumptions and the
many-to-one identity for the intended underlying cascade before claiming
that identification as an established result of this package. Check any
proposed correspondence with Harris's equations against the original
source, including the initial-type and observed-type conventions.

The code first checks `log(v) > log(theta)` and the type, then evaluates
`exp(log(theta)+log(W)-log(v))` only where the indicator is true. Every
contribution is in `[0,1]` since `0 < W <= 1` and `theta/v < 1` on that set.
Scoring in this order avoids an overflowing reciprocal when the energy is
far below threshold.

For `n >= 2` independent histories, the sample mean estimates the normalized
count. The reported sample variance uses denominator `n-1`, and its standard
error is `sqrt(sample_variance/n)`. Divide both the
mean and this standard error by `theta` to obtain the raw expected count
and its standard error. The reported normal interval is
`raw_count_mean +/- 1.96*raw_count_se` without clipping its endpoints.
It is an approximate pointwise interval, not a simultaneous band.

Different observations and thresholds on the same histories are correlated.
The code records how many payoffs were nonzero; an all-zero sample gives
an empirical variance of zero but does not establish that the expectation
or its uncertainty is zero. Sparse samples may make the normal interval
unreliable. Replicate counts and the observation grid in example runs are
demonstration choices, not completed research findings.

## Independent checks used by the tests

The type process has electron probability `p_e(t)` satisfying

```text
p_e'(t) = gamma0 - (mu+gamma0)*p_e(t).
```

For `mu+gamma0 > 0`, its solution is

```text
p_e(t) = gamma0/(mu+gamma0)
         + (p_e(0)-gamma0/(mu+gamma0))*exp(-(mu+gamma0)*t).
```

For two zero rates it is constant. The type-probability test compares an
ensemble with this solution; continuous energy loss does not affect it.

For any positive integer `s`, the residual Laplace exponent is

```text
psi(s) = integral_0^infinity (1-exp(-s*z))*Pi(z) dz
       = mu * sum_{k=2}^infinity (1/k - 1/(k+s))
       = mu * sum_{k=2}^{s+1} 1/k.
E[exp(-s*S_h)] = exp(-h*psi(s)).
```

The integral follows from the geometric expansion above; the final series
telescopes. The transform test checks multiple `s` values against this
formula, rather than reproducing the gamma/thinning implementation.
Its Monte Carlo tolerance is six theoretical standard errors, with
`Var(exp(-s*S_h)) = exp(-h*psi(2*s))-exp(-2*h*psi(s))`.
Statistical checks are seeded and supplement deterministic event-loop,
threshold and uncertainty checks. Passing tests is numerical evidence
about the implementation, not a mathematical convergence proof.

A separate count check uses the elementary limit `mu = alpha = 0` with an
initial photon. The original population then undergoes at most one event:
a photon split at rate `gamma0`, after which its two electron children do
not evolve. Since each uniformly split child exceeds `theta` with
probability `1-theta`, its expected raw counts are

```text
electron: 2*(1-theta)*(1-exp(-gamma0*t))
photon:   exp(-gamma0*t).
```

The test compares spine estimates with these population calculations.
For its sampling tolerance, the normalized electron payoff has second
moment `-2*theta^2*log(theta)*(1-exp(-gamma0*t))`, obtained by integrating
`(theta/v)^2` against the spine child's density `2v` on `(theta,1)` and
multiplying by the split probability. This check exercises the sampler and
count estimator together without simulating a full cascade.
