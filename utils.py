from __future__ import annotations

from typing import Iterable


class PayoffCalculator:
    """Payoff helpers matching the thesis notation."""

    @staticmethod
    def hospital_adoption_utility(
        v_sigma: float,
        price: float,
        kappa: float,
        beta_h: float,
        delta_h: float,
    ) -> float:
        return beta_h * delta_h * v_sigma - price * (1.0 + kappa)

    @staticmethod
    def firm_profit(price: float, cost_c: float) -> float:
        return price - cost_c

    @staticmethod
    def try_utility(
        c_trial: float,
        beta_h: float,
        delta_h: float,
        p_l: float,
        delta_v: float,
        v_bar: float,
        rho: float,
    ) -> float:
        return -c_trial + (delta_h ** 2) * (beta_h * delta_h - rho) * (v_bar + p_l * delta_v)

    @staticmethod
    def continuation_value(
        p_try: float,
        u_try: float,
        beta_h: float,
        delta_h: float,
    ) -> float:
        denominator = 1.0 - (1.0 - p_try) * beta_h * (delta_h ** 4)
        if denominator <= 0:
            raise ValueError("Continuation value does not converge under these parameters.")
        return p_try * u_try / denominator


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
