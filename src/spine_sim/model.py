"""Parameters for the homogeneous energy/type spine."""

import math
from dataclasses import dataclass

PARTICLE_TYPES = ("electron", "photon")


@dataclass(frozen=True)
class ModelParameters:
    """Rates per unit track length; initial energy sets the energy unit.

    ``mu`` controls electron losses and electron-to-photon switches;
    ``gamma0`` is the photon-to-electron rate; ``alpha`` is electron
    continuous fractional energy loss and weight decay. Zero rates are
    supported limiting cases. The simulator records energy relative to
    ``initial_energy``.
    """

    mu: float = 1.0
    gamma0: float = 1.0
    alpha: float = 0.0
    initial_energy: float = 1.0

    def __post_init__(self) -> None:
        for name in ("mu", "gamma0", "alpha", "initial_energy"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if name == "initial_energy" and value == 0:
                raise ValueError("initial_energy must be positive")
            object.__setattr__(self, name, value)
