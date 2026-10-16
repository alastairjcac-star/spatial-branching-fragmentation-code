# Homogeneous electron–photon spine simulation

This homogeneous, unbounded electron–photon model uses electron fragmentation
kernel $k(u)=\mu/u$ for $0<u<1$.
Photons split into two electrons at rate $\gamma_0$, with uniform energy
splitting density $q(u)=1$. Electrons also lose energy continuously according
to $dE/d\ell=-\alpha E$. Energy is normalised as $v=E/E_0$, starting at $v=1$.

The code simulates energy and type along a spine and uses weighted averages
of independent spine histories to estimate expected particle counts.

| Script | Purpose |
| --- | --- |
| [single_spine.py](single_spine.py) | Prints one spine's track length, type, normalised energy and weight. Saves no files. |
| [monte_carlo.py](monte_carlo.py) | Runs an ensemble per initial type, sharing histories across thresholds, counted types and observation lengths. Saves estimates and settings. |
| [plot_counts.py](plot_counts.py) | Produces four-panel figures from saved results without rerunning simulations. |

## Setup and running

Tested with **Python 3.14.4** and **Matplotlib 3.11.2**. The simulation scripts
use only Python's standard library; plotting requires Matplotlib, pinned in
[requirements.txt](requirements.txt).

From the project directory, using `uv` as the environment manager and an
installed Python 3.14.4:

```bash
uv venv --python 3.14.4 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

If `.venv` already has Matplotlib installed, just activate it. Then run:

```bash
python3 single_spine.py          # Print one realisation
python3 monte_carlo.py           # Generate counts.csv and counts_settings.json
python3 plot_counts.py           # Coloured curves with uncertainty intervals
python3 plot_counts.py --simple  # Simple black curves, means only
```

Skip the Monte Carlo command when plotting existing results.

## Changing the experiment

Edit the final `if __name__ == "__main__":` block in each simulation script.
In `monte_carlo.py`, change `mu`, `gamma0`, `alpha`, `initial_types`,
`thresholds`, `n`, `observation_times` and `seed`. Here `n` is the number of
histories **per initial type**. Defaults are:

- $\mu=\gamma_0=1$, $\alpha=0$, both initial types;
- 10,000 histories per initial type and thresholds 0.1, 0.03 and 0.01;
- observation lengths 0, 0.1, …, 6 and base seed `20261015`.

The single-spine demonstration has separate settings and observes every 0.25
up to length 6. Observation spacing is a sampling grid, not an integration
step. Plotting requires exactly three distinct thresholds, both initial types
and at least two distinct observation lengths.

## Results and uncertainty

[counts.csv](counts.csv) contains one row per initial type, threshold, track
length and counted type, with `estimated_count`, `standard_error` and
`nonzero_contributions`.

For counted type $j$, each history contributes
$X=(W/v)\mathbf{1}\{v>\theta,\ J=j\}$, where $J$ is its type and
$W=\exp(-\alpha L_e)$, with $L_e$ its accumulated electron track length.
For $\alpha=0$, $W=1$. The mean already represents
$\widehat C_{\theta,j}(\ell)/\theta$, the estimated expected number of
particles with energy strictly above $\theta E_0$. 

- `standard_error` is the sample standard deviation of contributions divided
  by $\sqrt{n}$, including zeros. It measures uncertainty in the estimated
  mean.
- `nonzero_contributions` counts contributing spine histories, not particles.
  An all-zero sample can miss rare contributions despite reporting zero
  standard error.

The simple figure shows means only. The
top/bottom rows start from an
electron/photon, and the left/right columns count electrons/photons.

[counts_settings.json](counts_settings.json) records the model parameters,
sample sizes, thresholds, observation lengths, seeds, Python version and
estimator definitions. Keep it together with its matching CSV.

## Reproducing runs and keeping outputs

Reproduce seeded runs using the same code, Python version, parameters, sample
size, observation grid, initial-type order and seeds. Defaults are `20261015`
for electrons and `20261016` for photons (`seed + position` in `initial_types`).
Each ensemble seeds its generator once, then advances it across histories.
Changing the grid can change histories despite keeping the seed. Use the tested
Matplotlib version for the plots.

Outputs are saved beside the scripts. Each Monte Carlo run overwrites
`counts.csv` and `counts_settings.json`. Default plotting overwrites
`counts_figure.pdf` and `counts_figure.png`; `--simple` overwrites
`counts_figure_simple.pdf` and `counts_figure_simple.png`. Each plotting style
replaces only its own pair. PDFs are vector graphics; PNGs are 300 dpi.
Copy outputs you want to keep before rerunning.

## Licence

The project's code is licensed under the [MIT licence](LICENSE).
