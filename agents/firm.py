# agents/firm.py
from typing import Dict, Optional
from contracts.contract import Contract
from agents.hospital import Hospital
from utils import PayoffCalculator
import numpy as np

class Firm:
    """Firm agent implementing Stage 1 equilibrium strategy."""
    
    def __init__(self, cost_C: float = 0.5):
        self.C = cost_C  # Serving cost C from thesis
        self.menu = None  # Will hold {'c_Y': Contract, 'c_S': Contract}
    
    def set_contract_menu(self, c_Y: Contract, c_S: Contract):
        """Set the signal-contingent contract menu."""
        self.menu