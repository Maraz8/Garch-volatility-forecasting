"""VaR backtesting utilities."""

import numpy as np
from scipy.stats import binom

class VaRBacktester:
    """Backtest VaR estimates."""
    
    @staticmethod
    def count_violations(returns, var_estimates):
        """Count VaR violations."""
        violations = (returns < var_estimates).sum()
        return violations
    
    @staticmethod
    def kupiec_test(violations, total_obs, confidence=0.95):
        """
        Kupiec test for VaR backtesting.
        H0: VaR model is correct
        """
        expected_violations = (1 - confidence) * total_obs
        violation_rate = violations / total_obs
        expected_rate = 1 - confidence
        
        # Likelihood ratio test statistic
        if violation_rate == 0 or violation_rate == 1:
            return {'test_stat': np.inf, 'p_value': 0, 'result': 'reject'}
        
        lr = -2 * np.log(
            ((1 - expected_rate) ** (total_obs - violations) * expected_rate ** violations) /
            ((1 - violation_rate) ** (total_obs - violations) * violation_rate ** violations)
        )
        
        # Critical value at 5% significance
        critical_value = 3.841  # chi-square(1) at 95%
        p_value = 1 - binom.cdf(violations, total_obs, expected_rate)
        
        result = 'pass' if lr < critical_value else 'reject'
        
        return {
            'test_stat': lr,
            'critical_value': critical_value,
            'p_value': p_value,
            'result': result,
            'violations': violations,
            'expected': expected_violations
        }
