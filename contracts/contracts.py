from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Contract:
    """Two contracts from the thesis: c_Y and c_S."""

    name: str
    fixed_price: float | None = None
    gamma: float | None = None

    def price(self, signal: str, v_bar: float, delta_v: float) -> float:
        if self.name == "c_Y":
            if self.fixed_price is None:
                raise ValueError("c_Y requires a fixed price.")
            return self.fixed_price
        if self.name == "c_S":
            if self.gamma is None:
                raise ValueError("c_S requires gamma.")
            value = v_bar + delta_v if signal == "sigma_L" else v_bar
            return self.gamma * value
        raise ValueError(f"Unknown contract {self.name!r}")
