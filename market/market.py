from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import random

from agents.firm import Firm
from agents.hospitals import Hospital
from contracts.contracts import Contract
from utils import PayoffCalculator, mean


@dataclass
class StageOneConfig:
    v_bar: float = 10.0
    delta_v: float = 4.0
    p_l: float = 0.45
    gamma: float = 0.7
    cost_c: float = 5.0
    delta_m: float = 0.95
    loops: int = 12
    runs: int = 1000
    seed: int = 7
    boundary_tolerance: float = 1e-6


class StageOneMarket:
    """Monte Carlo simulation for the thesis stage-1 firm-hospital game."""

    def __init__(self, config: StageOneConfig):
        self.config = config
        self.rng = random.Random(config.seed)

    def build_firm(self) -> Firm:
        return Firm(
            cost_C=self.config.cost_c,
            delta_m=self.config.delta_m,
            gamma=self.config.gamma,
            v_bar=self.config.v_bar,
            delta_v=self.config.delta_v,
        )

    def _draw_signal(self) -> str:
        return "sigma_L" if self.rng.random() < self.config.p_l else "sigma_H"

    def _choose_contract(
        self,
        firm: Firm,
        signal: str,
        q_l: float | None,
    ) -> Contract:
        menu = firm.build_menu()
        if signal == "sigma_H":
            return menu["c_S"]
        if q_l is not None and self.rng.random() < q_l:
            return menu["c_S"]
        return menu["c_Y"]

    def _adopts_in_mixed_boundary(
        self,
        hospital: Hospital,
        signal: str,
        contract: Contract,
        q_l: float,
    ) -> bool:
        if contract.name == "c_Y":
            return signal == "sigma_L"
        return self.rng.random() < q_l

    def _simulate_one_run(self, hospital: Hospital) -> Dict[str, float | int | str]:
        firm = self.build_firm()
        gamma = self.config.gamma
        v_bar = self.config.v_bar
        delta_v = self.config.delta_v

        u_try = hospital.try_payoff(gamma=gamma, p_l=self.config.p_l, v_bar=v_bar, delta_v=delta_v)
        pure_action = hospital.pure_strategy_action(
            gamma=gamma, p_l=self.config.p_l, v_bar=v_bar, delta_v=delta_v
        )
        boundary_gap = abs(hospital.beta_h * hospital.delta_h - hospital.rho(gamma))
        q_l = None
        if boundary_gap <= self.config.boundary_tolerance:
            q_l = hospital.mixed_boundary_probability(
                gamma=gamma, p_l=self.config.p_l, v_bar=v_bar, delta_v=delta_v
            )

        for cycle in range(1, self.config.loops + 1):
            if pure_action == "Delay" and q_l is None:
                continue

            signal = self._draw_signal()
            contract = self._choose_contract(firm=firm, signal=signal, q_l=q_l)

            if q_l is not None:
                adopted = self._adopts_in_mixed_boundary(
                    hospital=hospital, signal=signal, contract=contract, q_l=q_l
                )
            else:
                adopted = hospital.adoption_payoff(
                    signal=signal,
                    contract=contract,
                    v_bar=v_bar,
                    delta_v=delta_v,
                ) > 0.0

            if not adopted:
                continue

            value = hospital.value(signal=signal, v_bar=v_bar, delta_v=delta_v)
            price = contract.price(signal=signal, v_bar=v_bar, delta_v=delta_v)
            hospital_payoff = hospital.adoption_payoff(
                signal=signal, contract=contract, v_bar=v_bar, delta_v=delta_v
            )
            firm_payoff = PayoffCalculator.firm_profit(price=price, cost_c=firm.cost_C)
            return {
                "hospital": hospital.name,
                "equilibrium": "mixed" if q_l is not None else "pure",
                "u_try": u_try,
                "signal": signal,
                "contract": contract.name,
                "value": value,
                "price": price,
                "hospital_payoff": hospital_payoff,
                "firm_payoff": firm_payoff,
                "adopted": 1,
                "delayed_forever": 0,
                "cycle_of_adoption": cycle,
                "q_l": q_l if q_l is not None else 0.0,
            }

        continuation = hospital.continuation_value(p_try=0.0 if pure_action == "Delay" else 1.0, u_try=max(u_try, 0.0))
        return {
            "hospital": hospital.name,
            "equilibrium": "mixed" if q_l is not None else "pure",
            "u_try": u_try,
            "signal": "none",
            "contract": "none",
            "value": 0.0,
            "price": 0.0,
            "hospital_payoff": 0.0,
            "firm_payoff": 0.0,
            "adopted": 0,
            "delayed_forever": 1 if pure_action == "Delay" and q_l is None else 0,
            "cycle_of_adoption": 0,
            "q_l": q_l if q_l is not None else 0.0,
            "continuation_value": continuation,
        }

    def simulate_profile(self, hospital: Hospital) -> Dict[str, float | str]:
        runs = [self._simulate_one_run(hospital) for _ in range(self.config.runs)]
        return {
            "hospital": hospital.name,
            "size": hospital.size,
            "ownership": hospital.ownership,
            "beta_h": hospital.beta_h,
            "delta_h": hospital.delta_h,
            "kappa": hospital.kappa,
            "trial_cost": hospital.trial_cost,
            "equilibrium": runs[0]["equilibrium"],
            "u_try": runs[0]["u_try"],
            "q_l": runs[0]["q_l"],
            "adoption_rate": mean(run["adopted"] for run in runs),
            "delay_rate": mean(run["delayed_forever"] for run in runs),
            "avg_cycle_of_adoption": mean(
                run["cycle_of_adoption"] for run in runs if run["cycle_of_adoption"] > 0
            ),
            "avg_hospital_payoff": mean(run["hospital_payoff"] for run in runs),
            "avg_firm_payoff": mean(run["firm_payoff"] for run in runs),
            "share_c_Y": mean(1.0 if run["contract"] == "c_Y" else 0.0 for run in runs),
            "share_c_S": mean(1.0 if run["contract"] == "c_S" else 0.0 for run in runs),
        }

    def compare_profiles(self, hospitals: List[Hospital]) -> List[Dict[str, float | str]]:
        return [self.simulate_profile(hospital) for hospital in hospitals]


def default_hospital_profiles(config: StageOneConfig) -> List[Hospital]:
    """
    These profile differences are not in the thesis directly, so we encode them
    as transparent simulation assumptions:
    - smaller/community hospitals face higher implementation friction
    - less biased/private hospitals have higher beta_h and slightly lower trial cost
    """
    return [
        Hospital(
            name="Low-bias large private",
            beta_h=0.93,
            delta_h=0.95,
            kappa=0.05,
            trial_cost=0.20,
            size="large",
            ownership="private",
        ),
        Hospital(
            name="Moderate-bias large community",
            beta_h=0.85,
            delta_h=0.92,
            kappa=0.10,
            trial_cost=0.45,
            size="large",
            ownership="community",
        ),
        Hospital(
            name="High-bias small private",
            beta_h=0.79,
            delta_h=0.90,
            kappa=0.14,
            trial_cost=0.80,
            size="small",
            ownership="private",
        ),
        Hospital(
            name="High-bias small community",
            beta_h=0.76,
            delta_h=0.88,
            kappa=0.18,
            trial_cost=1.05,
            size="small",
            ownership="community",
        ),
        Hospital(
            name="Boundary mixer",
            beta_h=0.84,
            delta_h=0.90,
            kappa=0.10,
            trial_cost=0.30,
            size="medium",
            ownership="community",
            mixed_h_lya_override=2.0,
        ),
    ]
