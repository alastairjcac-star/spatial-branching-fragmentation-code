# API examples

These short scripts demonstrate reusable package functions. Run them from the
repository root after `uv sync --locked`:

```sh
uv run --locked python examples/single_spine.py
uv run --locked python examples/ensemble.py
```

`single_spine.py` shows finite observations of one path for each initial type,
including log energy and log weight. `ensemble.py` shows the configuration,
independent-history simulation, and count-estimation workflow.

For a saved experiment with source snapshots, metadata, and separate plots,
use the [reference experiment](../experiments/reference/README.md). For
analytical comparisons use [validation](../experiments/validation/README.md).

`reference.json` is retained for earlier commands and contains the same settings
as `experiments/reference/config.json`. The legacy CLI spelling
`spine-sim --config ... --output ...` now runs simulation only. Generate figures
from the saved results with `spine-sim plot`.
