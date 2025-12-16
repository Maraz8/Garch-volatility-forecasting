"""Data preprocessing for risk modeling."""

import pandas as pd
import numpy as np
from pathlib import Path

def prepare_returns(price_data):
    """Prepare returns for risk modeling."""
    returns = {}
    for ticker, df in price_data.items():
        # Use close prices
        close_prices = df['Close'] if 'Close' in df.columns else df['close']
        # Calculate log returns
        returns[ticker] = np.log(close_prices / close_prices.shift(1)).dropna()
    return returns

def align_returns(returns_dict):
    """Align returns across assets to common dates."""
    # Convert dict to DataFrame
    returns_df = pd.DataFrame(returns_dict)
    # Drop any rows with NaN
    returns_df = returns_df.dropna()
    return returns_df

def split_data(returns_df, train_end, val_end):
    """
    Split data into train, validation, and test sets.

    Args:
        returns_df: DataFrame of returns
        train_end: End date for training set (str or datetime)
        val_end: End date for validation set (str or datetime)

    Returns:
        Tuple of (train, val, test) DataFrames
    """
    train = returns_df.loc[:train_end]
    val = returns_df.loc[train_end:val_end]
    test = returns_df.loc[val_end:]

    return train, val, test

def calculate_realized_volatility(returns, window=20):
    """
    Calculate realized volatility using rolling standard deviation.

    Args:
        returns: Series or DataFrame of returns
        window: Rolling window size (default: 20 days)

    Returns:
        Realized volatility series/dataframe
    """
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized

def detect_outliers(returns, n_std=3):
    """
    Detect outliers using z-score method.

    Args:
        returns: Series of returns
        n_std: Number of standard deviations for threshold

    Returns:
        Boolean series indicating outliers
    """
    z_scores = np.abs((returns - returns.mean()) / returns.std())
    return z_scores > n_std

def winsorize_returns(returns, lower=0.01, upper=0.99):
    """
    Winsorize returns by capping at percentiles.

    Args:
        returns: Series of returns
        lower: Lower percentile (default: 1%)
        upper: Upper percentile (default: 99%)

    Returns:
        Winsorized returns
    """
    lower_bound = returns.quantile(lower)
    upper_bound = returns.quantile(upper)
    return returns.clip(lower=lower_bound, upper=upper_bound)

def clean_returns(returns_df, method='drop', **kwargs):
    """
    Clean returns data by handling outliers.

    Args:
        returns_df: DataFrame of returns
        method: 'drop', 'winsorize', or 'none'
        **kwargs: Additional arguments for cleaning method

    Returns:
        Cleaned returns DataFrame
    """
    if method == 'drop':
        n_std = kwargs.get('n_std', 3)
        mask = ~returns_df.apply(lambda x: detect_outliers(x, n_std)).any(axis=1)
        return returns_df[mask]
    elif method == 'winsorize':
        lower = kwargs.get('lower', 0.01)
        upper = kwargs.get('upper', 0.99)
        return returns_df.apply(lambda x: winsorize_returns(x, lower, upper))
    else:
        return returns_df

def save_processed_data(returns_df, filename, output_dir='data/processed'):
    """Save processed returns data."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / filename
    returns_df.to_csv(file_path)
    print(f"Saved processed data to {file_path}")

def load_processed_data(filename, data_dir='data/processed'):
    """Load processed returns data."""
    file_path = Path(data_dir) / filename
    if file_path.exists():
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        print(f"File not found: {file_path}")
        return None

def calculate_summary_statistics(returns):
    """
    Calculate summary statistics for returns.

    Returns:
        Dictionary of statistics
    """
    from scipy.stats import skew, kurtosis

    stats = {
        'mean': returns.mean(),
        'std': returns.std(),
        'min': returns.min(),
        'max': returns.max(),
        'skewness': skew(returns.dropna()),
        'kurtosis': kurtosis(returns.dropna()),
        'sharpe': returns.mean() / returns.std() * np.sqrt(252)  # Annualized
    }

    return stats
