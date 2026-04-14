# agents/hospital.py
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
from contracts.contract import Contract
from utils import PayoffCalculator

@dataclass
class Hospital:
    """Hospital agent with behavioral biases from Stage 1."""
    sigma: str  # 'L' or 'H' (low/high bias)
    beta_h: float  # Present bias (h < 1)
    delta_h: float  # Impatience (h < 1)
    v_bar: float    # Baseline technology value
    Delta_v: float  # Value differential L vs H
    size: str       # 'small' or 'large' (affects switching costs)
    ownership: str  # 'private' or 'community' (affects risk aversion)
    id: int         # Unique ID for tracking
    adopted: bool = False
    trial_cost: float = 0.1  # ctrial from thesis
    
    def get_v_sigma(self) -> float:
        """Technology value v_σ from thesis."""
        if self.sigma == 'L':
            return self.v_bar + self.Delta_v
        return self.v_bar
    
    def adoption_utility(self, contract: Contract) -> float:
        """h_{σ c A} = β_h δ_h v_σ - P_c (1+κ) from Stage 1."""
        v_sigma = self.get_v_sigma()
        P_c = contract.F + contract.gamma * v_sigma * (1 + contract.kappa)
        return self.beta_h * self.delta_h * v_sigma - P_c * (1 + contract.kappa)
    
    def try_utility(self, contract_menu: Dict[str, Contract], p_L: float) -> float:
        """U_h^{Try} = -c_trial + δ_h² (β_h δ_h - ρ) (v_bar + p_L Δv) from thesis."""
        # Expected adoption payoff under signal-contingent menu
        expected_adoption = p_L * self.adoption_utility(contract_menu['c_Y']) + \
                           (1 - p_L) * self.adoption_utility(contract_menu['c_S'])
        return -self.trial_cost + (self.delta_h ** 2) * expected_adoption
    
    def decide_action(self, 
                     contract_menu: Dict[str, Contract], 
                     p_L: float, 
                     continuation_value: float = 0.0) -> Literal['Try', 'Delay', 'Reject']:
        """
        t=0 decision: Try, Delay, or Reject.
        Hospital chooses Try if U^{Try} > δ_h^4 β_h V_h (delay payoff).
        """
        U_try = self.try_utility(contract_menu, p_L)
        U_delay = self.delta_h**4 * self.beta_h * continuation_value
        
        if U_try > U_delay:
            return 'Try'
        elif U_try < U_delay:
            return 'Reject'  # Weakly dominated by Delay
        else:
            return 'Delay'  # Indifferent
    
    def decide_adopt(self, contract: Contract, signal: str) -> bool:
        """
        t=3 decision: Adopt or Not after seeing signal and contract.
        Screening: L accepts c_Y, both accept c_S (pure strategy).
        """
        utility = self.adoption_utility(contract)
        return utility > 0
    
    def update_state(self, action: str, signal: Optional[str] = None, contract: Optional[Contract] = None):
        """Update hospital state after action."""
        if action == 'Adopt':
            self.adopted = True
        elif action == 'Try':
            # Signal is drawn, but hospital state unchanged until t=3
            pass
        # Delay/Reject: no state change, retry next period