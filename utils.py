# utils.py
import numpy as np

class PayoffCalculator:
    """Static class with payoff functions from your thesis."""
    
    @staticmethod
    def hospital_adoption_utility(v_sigma, P_c, kappa, beta_h, delta_h):
        """Stage 1 hospital payoff h_{\sigma c A}."""
        return beta_h * delta_h * v_sigma - P_c * (1 + kappa)
    
    @staticmethod
    def firm_profit(pi_c, C):
        """Stage 1 firm profit m_{\sigma c A}."""
        return pi_c - C
    
    @staticmethod
    def try_utility(c_trial, beta_h, delta_h, p_L, Delta_v, bar_v, rho):
        """U_h^{Try} from your equilibrium."""
        return -c_trial + delta_h**2 * (beta_h * delta_h - rho) * (bar_v + p_L * Delta_v)
    
    @staticmethod
    def continuation_value(p_T, U_try, beta_h, delta_h):
        """V_h fixed point."""
        return p_T * U_try / (1 - (1 - p_T) * beta_h * delta_h**3)