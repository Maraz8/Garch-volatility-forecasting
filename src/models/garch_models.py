"""GARCH family models for volatility forecasting."""

from arch import arch_model
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

class GARCHVolatility:
    """GARCH(1,1) volatility model."""

    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.model = None
        self.results = None

    def fit(self, returns):
        """Fit GARCH model to returns."""
        # Scale returns to percentage
        returns_pct = returns * 100

        # Fit GARCH(1,1)
        self.model = arch_model(returns_pct, vol='Garch', p=self.p, q=self.q)
        self.results = self.model.fit(disp='off')
        return self.results

    def forecast(self, horizon=1):
        """Forecast volatility."""
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")

        forecast = self.results.forecast(horizon=horizon)
        return forecast.variance.iloc[-1] / 100  # Convert back from percentage

    def rolling_forecast(self, returns, window=252, refit_freq=20):
        """
        Generate rolling volatility forecasts.

        Args:
            returns: Full return series
            window: Training window size
            refit_freq: Frequency to refit model (in days)

        Returns:
            Series of volatility forecasts
        """
        forecasts = []
        dates = []

        for i in range(window, len(returns)):
            # Refit only at specified frequency
            if (i - window) % refit_freq == 0 or i == window:
                train_data = returns.iloc[i-window:i]
                self.fit(train_data)

            # Forecast next period
            vol_forecast = self.forecast(horizon=1)
            forecasts.append(vol_forecast.iloc[0])
            dates.append(returns.index[i])

        return pd.Series(forecasts, index=dates, name='GARCH_forecast')

    def save(self, filepath):
        """Save model results."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)

    def load(self, filepath):
        """Load model results."""
        with open(filepath, 'rb') as f:
            self.results = pickle.load(f)

class EGARCHVolatility:
    """EGARCH model for asymmetric volatility."""

    def __init__(self, p=1, o=1, q=1):
        self.p = p
        self.o = o
        self.q = q
        self.model = None
        self.results = None

    def fit(self, returns):
        """Fit EGARCH model."""
        returns_pct = returns * 100
        self.model = arch_model(returns_pct, vol='EGARCH', p=self.p, o=self.o, q=self.q)
        self.results = self.model.fit(disp='off')
        return self.results

    def forecast(self, horizon=1):
        """Forecast volatility."""
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")

        forecast = self.results.forecast(horizon=horizon)
        return forecast.variance.iloc[-1] / 100

    def rolling_forecast(self, returns, window=252, refit_freq=20):
        """Generate rolling volatility forecasts."""
        forecasts = []
        dates = []

        for i in range(window, len(returns)):
            if (i - window) % refit_freq == 0 or i == window:
                train_data = returns.iloc[i-window:i]
                self.fit(train_data)

            vol_forecast = self.forecast(horizon=1)
            forecasts.append(vol_forecast.iloc[0])
            dates.append(returns.index[i])

        return pd.Series(forecasts, index=dates, name='EGARCH_forecast')

    def save(self, filepath):
        """Save model results."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)

    def load(self, filepath):
        """Load model results."""
        with open(filepath, 'rb') as f:
            self.results = pickle.load(f)

class GJRGARCHVolatility:
    """GJR-GARCH model for leverage effects."""

    def __init__(self, p=1, o=1, q=1):
        self.p = p
        self.o = o
        self.q = q
        self.model = None
        self.results = None

    def fit(self, returns):
        """Fit GJR-GARCH model."""
        returns_pct = returns * 100
        # GJR-GARCH in arch package uses power=2.0 by default
        self.model = arch_model(returns_pct, vol='GARCH', p=self.p, o=self.o, q=self.q,
                                power=2.0)
        self.results = self.model.fit(disp='off')
        return self.results

    def forecast(self, horizon=1):
        """Forecast volatility."""
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")

        forecast = self.results.forecast(horizon=horizon)
        return forecast.variance.iloc[-1] / 100

    def rolling_forecast(self, returns, window=252, refit_freq=20):
        """Generate rolling volatility forecasts."""
        forecasts = []
        dates = []

        for i in range(window, len(returns)):
            if (i - window) % refit_freq == 0 or i == window:
                train_data = returns.iloc[i-window:i]
                self.fit(train_data)

            vol_forecast = self.forecast(horizon=1)
            forecasts.append(vol_forecast.iloc[0])
            dates.append(returns.index[i])

        return pd.Series(forecasts, index=dates, name='GJRGARCH_forecast')

    def save(self, filepath):
        """Save model results."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)

    def load(self, filepath):
        """Load model results."""
        with open(filepath, 'rb') as f:
            self.results = pickle.load(f)

def compare_garch_models(returns, models=['GARCH', 'EGARCH', 'GJRGARCH']):
    """
    Compare different GARCH family models.

    Args:
        returns: Return series
        models: List of models to compare

    Returns:
        DataFrame with model comparison results
    """
    results = {}

    for model_name in models:
        if model_name == 'GARCH':
            model = GARCHVolatility()
        elif model_name == 'EGARCH':
            model = EGARCHVolatility()
        elif model_name == 'GJRGARCH':
            model = GJRGARCHVolatility()
        else:
            continue

        fit_result = model.fit(returns)

        results[model_name] = {
            'AIC': fit_result.aic,
            'BIC': fit_result.bic,
            'LogLikelihood': fit_result.loglikelihood
        }

    return pd.DataFrame(results).T
