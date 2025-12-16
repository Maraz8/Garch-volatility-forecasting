"""Correlation analysis utilities."""

import pandas as pd
import numpy as np

def rolling_correlation(returns_df, window=30):
    """Calculate rolling correlation matrix."""
    return returns_df.rolling(window=window).corr()

def regime_correlations(returns_df, regime_labels):
    """Calculate correlations by market regime."""
    regime_corr = {}
    for regime in regime_labels.unique():
        regime_mask = regime_labels == regime
        regime_corr[regime] = returns_df[regime_mask].corr()
    return regime_corr

def detect_correlation_regimes(correlation_series, threshold=0.7):
    """Detect high/low correlation regimes."""
    high_corr = correlation_series > threshold
    low_corr = correlation_series < -threshold
    return pd.DataFrame({'high_corr': high_corr, 'low_corr': low_corr})
