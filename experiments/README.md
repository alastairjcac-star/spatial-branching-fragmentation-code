# Experiments

Each named experiment contains tracked settings and a short guide. Generated
results belong in ignored `results/` directories at the repository root.

- [reference](reference/README.md): the homogeneous count demonstration for both
  initial types, with separate simulation and figure commands.
- [validation](validation/README.md): reproducible numerical comparisons with
  analytical identities, supplementing the automated tests.

Run the commands from the repository root with `uv run --locked`. Give each
simulation, validation run, and figure render a new output directory. Keep the
effective saved configuration and provenance with its outputs. Figure changes
can read existing estimates without rerunning simulation.

For a new study, add a named directory with its configuration and instructions
explaining its purpose, model parameters, history count, seed policy, requested
observations, and uncertainty assessment. Distinguish demonstration settings
from settings justified for a precision study. Small API tutorials belong in
`examples/` instead.
