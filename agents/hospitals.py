from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

from contracts.contracts import Contract
from utils import PayoffCalculator


Action = Literal["Try", "Delay", "Reject"]


@dataclass
class Hospital:
    """
    Hospital agent with behavioral and organizational characteristics.

    The thesis model pins down beta_h and delta_h directly. The extra fields
    `size` and `ownership` help us compare hospital profiles in simulation.
    """

    name: str
    beta_h: float
    delta_h: float
    kappa: float
    trial_cost: float
    size: str
    ownership: str
    alpha: float = 0.0
    mixed_h_lya_override: float | None = None

    def value(self, signal: str, v_bar: float, delta_v: float) -> float:
        if signal == "sigma_L":
            return v_bar + delta_v
        return v_bar

    def rho(self, gamma: float) -> float:
        return gamma * (1.0 + self.kappa)

    def adoption_payoff(
        self,
        signal: str,
        contract: Contract,
        v_bar: float,
        delta_v: float,
    ) -> float:
        value = self.value(signal, v_bar, delta_v)
        price = contract.price(signal=signal, v_bar=v_bar, delta_v=delta_v)
        return PayoffCalculator.hospital_adoption_utility(
            value, price, self.kappa, self.beta_h, self.delta_h
        )

    def h_lya(self, firm_gamma: float, v_bar: float, delta_v: float) -> float:
        """
        Low-signal payoff under c_Y from the thesis:
        h_LYA is the low-signal surplus when c_Y is offered.

        In the pure-strategy section, Assumption 7 implies:
        P0 + Pm = gamma * (v_bar + delta_v).

        The mixed-strategy section additionally treats h_LYA as strictly positive
        at the boundary beta_h * delta_h = rho. Because those two statements are
        not jointly consistent, the simulation allows an optional override so the
        intended mixed boundary can still be explored explicitly.
        """
        if self.mixed_h_lya_override is not None:
            return self.mixed_h_lya_override
        rho = self.rho(firm_gamma)
        return (self.beta_h * self.delta_h - rho) * (v_bar + delta_v)

    def try_payoff(self, gamma: float, p_l: float, v_bar: float, delta_v: float) -> float:
        rho = self.rho(gamma)
        return PayoffCalculator.try_utility(
            c_trial=self.trial_cost,
            beta_h=self.beta_h,
            delta_h=self.delta_h,
            p_l=p_l,
            delta_v=delta_v,
            v_bar=v_bar,
            rho=rho,
        )

    def continuation_value(self, p_try: float, u_try: float) -> float:
        return PayoffCalculator.continuation_value(
            p_try=p_try,
            u_try=u_try,
            beta_h=self.beta_h,
            delta_h=self.delta_h,
        )

    def pure_strategy_action(self, gamma: float, p_l: float, v_bar: float, delta_v: float) -> Action:
        u_try = self.try_payoff(gamma=gamma, p_l=p_l, v_bar=v_bar, delta_v=delta_v)
        if u_try > 0:
            return "Try"
        if u_try < 0:
            return "Delay"
        return "Try"

    def mixed_boundary_probability(
        self,
        gamma: float,
        p_l: float,
        v_bar: float,
        delta_v: float,
    ) -> float | None:
        """
        Mixed equilibrium at the boundary beta_h * delta_h = rho.
        q_L* = 1 - c_trial / (delta_h^2 * p_L * h_LYA)
        """
        h_lya = self.h_lya(firm_gamma=gamma, v_bar=v_bar, delta_v=delta_v)
        denominator = (self.delta_h ** 2) * p_l * h_lya
        if denominator <= 0:
            return None
        q_l = 1.0 - self.trial_cost / denominator
        if 0.0 < q_l < 1.0:
            return q_l
        return None

    def summary(self) -> Dict[str, float | str]:
        return {
            "name": self.name,
            "beta_h": self.beta_h,
            "delta_h": self.delta_h,
            "kappa": self.kappa,
            "trial_cost": self.trial_cost,
            "size": self.size,
            "ownership": self.ownership,
        }
