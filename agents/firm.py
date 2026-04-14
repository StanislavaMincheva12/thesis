from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from contracts.contracts import Contract


@dataclass
class Firm:
    """Firm strategy for the stage-1 signal-contingent contracting game."""

    cost_C: float
    delta_m: float
    gamma: float
    v_bar: float
    delta_v: float

    @property
    def rho(self) -> float:
        """Price-pressure term before hospital friction is applied."""
        return self.gamma

    @property
    def two_part_price(self) -> float:
        """
        Assumption 7 pins down the two-part tariff total price:
        P0 + Pm = gamma * (v_bar + delta_v).
        """
        return self.gamma * (self.v_bar + self.delta_v)

    def build_menu(self) -> Dict[str, Contract]:
        """Create the menu used in both pure and mixed simulations."""
        return {
            "c_Y": Contract(name="c_Y", fixed_price=self.two_part_price, gamma=None),
            "c_S": Contract(name="c_S", fixed_price=None, gamma=self.gamma),
        }

    def offer_for_signal(self, signal: str, q_l: float = 0.0) -> Contract:
        """
        Firm equilibrium strategy:
        - Pure strategy: offer c_Y at sigma_L and c_S at sigma_H.
        - Mixed boundary: at sigma_L offer c_S with probability q_l.
        """
        menu = self.build_menu()
        if signal == "sigma_H":
            return menu["c_S"]
        if q_l > 0.0:
            # The simulation layer handles the random draw and calls this
            # method only after deciding which contract to instantiate.
            return menu["c_S"]
        return menu["c_Y"]
