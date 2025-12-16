"""Value-at-Risk calculation methods."""

import numpy as np
import pandas as pd
from scipy import stats

class VaRCalculator:
    """Calculate VaR using multiple methods."""

    @staticmethod
    def historical_var(returns, confidence=0.95):
        """Historical simulation VaR."""
        return np.percentile(returns, (1 - confidence) * 100)

    @staticmethod
    def parametric_var(returns, confidence=0.95):
        """Parametric VaR (normal distribution)."""
        mu = returns.mean()
        sigma = returns.std()
        z_score = stats.norm.ppf(1 - confidence)
        return mu + z_score * sigma

    @staticmethod
    def parametric_var_t(returns, confidence=0.95):
        """Parametric VaR with Student-t distribution."""
        params = stats.t.fit(returns)
        df, loc, scale = params
        t_score = stats.t.ppf(1 - confidence, df)
        return loc + t_score * scale

    @staticmethod
    def cornish_fisher_var(returns, confidence=0.95):
        """Cornish-Fisher VaR (adjusted for skewness/kurtosis)."""
        from scipy.stats import skew, kurtosis

        mu = returns.mean()
        sigma = returns.std()
        s = skew(returns)
        k = kurtosis(returns)

        z = stats.norm.ppf(1 - confidence)
        z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * k / 24 - (2*z**3 - 5*z) * s**2 / 36

        return mu + z_cf * sigma

    @staticmethod
    def filtered_historical_var(returns, volatility_forecast, confidence=0.95, window=250):
        """
        Filtered Historical Simulation VaR using GARCH volatility.

        Args:
            returns: Return series
            volatility_forecast: GARCH volatility forecast
            confidence: Confidence level
            window: Historical window for standardized returns

        Returns:
            VaR estimate
        """
        # Standardize historical returns by their realized volatility
        realized_vol = returns.rolling(window=20).std()
        standardized_returns = returns / realized_vol

        # Get historical standardized returns
        hist_std_returns = standardized_returns.dropna().iloc[-window:]

        # Calculate VaR of standardized returns
        std_var = np.percentile(hist_std_returns, (1 - confidence) * 100)

        # Scale by forecasted volatility
        var = std_var * volatility_forecast

        return var

    @staticmethod
    def expected_shortfall(returns, confidence=0.95):
        """Expected Shortfall (CVaR) - average loss beyond VaR."""
        var = VaRCalculator.historical_var(returns, confidence)
        return returns[returns <= var].mean()

    @staticmethod
    def rolling_var(returns, confidence=0.95, window=250, method='historical'):
        """
        Calculate rolling VaR estimates.

        Args:
            returns: Return series
            confidence: Confidence level
            window: Rolling window size
            method: VaR method ('historical', 'parametric', 'parametric_t', 'cornish_fisher')

        Returns:
            Series of rolling VaR estimates
        """
        var_estimates = []
        dates = []

        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]

            if method == 'historical':
                var = VaRCalculator.historical_var(window_returns, confidence)
            elif method == 'parametric':
                var = VaRCalculator.parametric_var(window_returns, confidence)
            elif method == 'parametric_t':
                var = VaRCalculator.parametric_var_t(window_returns, confidence)
            elif method == 'cornish_fisher':
                var = VaRCalculator.cornish_fisher_var(window_returns, confidence)
            else:
                raise ValueError(f"Unknown method: {method}")

            var_estimates.append(var)
            dates.append(returns.index[i])

        return pd.Series(var_estimates, index=dates, name=f'VaR_{method}')

    @staticmethod
    def portfolio_var(returns_df, weights, confidence=0.95, method='historical'):
        """
        Calculate portfolio VaR.

        Args:
            returns_df: DataFrame of individual asset returns
            weights: Array or Series of portfolio weights
            confidence: Confidence level
            method: VaR calculation method

        Returns:
            Portfolio VaR
        """
        # Calculate portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)

        if method == 'historical':
            return VaRCalculator.historical_var(portfolio_returns, confidence)
        elif method == 'parametric':
            return VaRCalculator.parametric_var(portfolio_returns, confidence)
        elif method == 'parametric_t':
            return VaRCalculator.parametric_var_t(portfolio_returns, confidence)
        elif method == 'cornish_fisher':
            return VaRCalculator.cornish_fisher_var(portfolio_returns, confidence)
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def diversification_benefit(returns_df, weights, confidence=0.95, method='historical'):
        """
        Calculate diversification benefit.

        Diversification Benefit = Sum of Individual VaRs - Portfolio VaR

        Args:
            returns_df: DataFrame of individual asset returns
            weights: Array or Series of portfolio weights
            confidence: Confidence level
            method: VaR calculation method

        Returns:
            Dictionary with diversification metrics
        """
        # Calculate individual VaRs
        individual_vars = {}
        for col in returns_df.columns:
            if method == 'historical':
                var = VaRCalculator.historical_var(returns_df[col], confidence)
            elif method == 'parametric':
                var = VaRCalculator.parametric_var(returns_df[col], confidence)
            elif method == 'parametric_t':
                var = VaRCalculator.parametric_var_t(returns_df[col], confidence)
            elif method == 'cornish_fisher':
                var = VaRCalculator.cornish_fisher_var(returns_df[col], confidence)

            individual_vars[col] = var

        # Weighted sum of individual VaRs
        weighted_sum_var = sum(individual_vars[col] * weights[i]
                               for i, col in enumerate(returns_df.columns))

        # Portfolio VaR
        portfolio_var = VaRCalculator.portfolio_var(returns_df, weights, confidence, method)

        # Diversification benefit
        div_benefit = weighted_sum_var - portfolio_var

        return {
            'individual_vars': individual_vars,
            'weighted_sum_var': weighted_sum_var,
            'portfolio_var': portfolio_var,
            'diversification_benefit': div_benefit,
            'diversification_ratio': div_benefit / abs(weighted_sum_var) if weighted_sum_var != 0 else 0
        }
