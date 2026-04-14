# contracts/contract.py
from dataclasses import dataclass
from typing import Dict
from utils import PayoffCalculator

@dataclass
class Contract:
    """Contract objects with fees and parameters."""
    name: str          # 'c_Y' or 'c_S'
    F: float           # Fixed fee
    gamma: float       # Usage coefficient
    tau: float         # Success contingent
    M: float           # Maintenance per period
    kappa: float       # Switching friction
    
    def expected_profit(self, v_sigma: float, C: float) -> float:
        """Firm expected profit under this contract."""
        P_c = self.F + self.gamma * v_sigma * (1 + self.kappa)
        return PayoffCalculator.firm_profit(P_c, C)
    
    def hospital_utility(self, v_sigma: float, beta_h: float, delta_h: float) -> float:
        """Hospital adoption utility h_{\sigma c A}."""
        P_c = self.F + self.gamma * v_sigma * (1 + self.kappa)
        return PayoffCalculator.hospital_adoption