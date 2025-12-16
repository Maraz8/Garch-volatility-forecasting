"""Plotting utilities for financial data visualization."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def plot_returns(returns, title='Returns Over Time', save_path=None):
    """
    Plot returns time series.

    Args:
        returns: Series or DataFrame of returns
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    if isinstance(returns, pd.Series):
        ax.plot(returns.index, returns.values, linewidth=0.8, alpha=0.7)
        ax.set_ylabel('Returns')
    else:
        for col in returns.columns:
            ax.plot(returns.index, returns[col], label=col, linewidth=0.8, alpha=0.7)
        ax.legend(loc='best')
        ax.set_ylabel('Returns')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig, ax

def plot_volatility(volatility, title='Volatility Over Time', save_path=None):
    """Plot volatility time series."""
    fig, ax = plt.subplots(figsize=(14, 6))

    if isinstance(volatility, pd.Series):
        ax.plot(volatility.index, volatility.values, color='darkred', linewidth=1.2)
        ax.fill_between(volatility.index, 0, volatility.values, alpha=0.3, color='red')
    else:
        for col in volatility.columns:
            ax.plot(volatility.index, volatility[col], label=col, linewidth=1.2)
        ax.legend(loc='best')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility (Annualized)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax

def plot_var_breaches(returns, var_estimates, confidence=0.95, title='VaR Breaches', save_path=None):
    """
    Plot returns with VaR threshold and highlight breaches.

    Args:
        returns: Series of returns
        var_estimates: Series of VaR estimates
        confidence: Confidence level
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot returns
    ax.plot(returns.index, returns.values, color='blue', linewidth=0.8, alpha=0.6, label='Returns')

    # Plot VaR threshold
    ax.plot(var_estimates.index, var_estimates.values, color='red', linewidth=1.5,
            linestyle='--', label=f'VaR ({confidence*100:.0f}%)')

    # Highlight breaches
    breaches = returns < var_estimates
    breach_dates = returns.index[breaches]
    breach_values = returns[breaches]

    ax.scatter(breach_dates, breach_values, color='red', s=50, zorder=5,
               label=f'Breaches ({breaches.sum()})')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax

def plot_correlation_matrix(returns_df, title='Correlation Matrix', save_path=None):
    """Plot correlation heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    corr_matrix = returns_df.corr()

    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={'label': 'Correlation'}, ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax

def plot_rolling_correlation(corr_series, asset1, asset2, window=30,
                             title=None, save_path=None):
    """Plot rolling correlation between two assets."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(corr_series.index, corr_series.values, color='darkblue', linewidth=1.5)
    ax.fill_between(corr_series.index, 0, corr_series.values, alpha=0.3, color='blue')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axhline(y=0.7, color='green', linestyle='--', linewidth=1, alpha=0.5, label='High Correlation')
    ax.axhline(y=-0.7, color='red', linestyle='--', linewidth=1, alpha=0.5, label='High Negative Correlation')

    if title is None:
        title = f'Rolling {window}-day Correlation: {asset1} vs {asset2}'

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Correlation')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax

def plot_qq(returns, title='Q-Q Plot', save_path=None):
    """Plot Q-Q plot to check normality."""
    from scipy import stats

    fig, ax = plt.subplots(figsize=(8, 8))

    stats.probplot(returns.dropna(), dist="norm", plot=ax)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax

def plot_distribution(returns, title='Return Distribution', save_path=None):
    """Plot return distribution with normal overlay."""
    from scipy import stats

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(returns.dropna(), bins=50, density=True, alpha=0.7, color='skyblue',
            edgecolor='black', label='Empirical')

    # Fit normal distribution
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Returns')
    ax.set_ylabel('Density')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add statistics
    textstr = f'Mean: {mu:.4f}\nStd: {sigma:.4f}\nSkew: {stats.skew(returns.dropna()):.2f}\nKurt: {stats.kurtosis(returns.dropna()):.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax

def plot_model_comparison(actual, predictions_dict, title='Model Comparison', save_path=None):
    """
    Compare multiple model predictions against actual values.

    Args:
        actual: Series of actual values
        predictions_dict: Dictionary of {model_name: predictions}
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot actual
    ax.plot(actual.index, actual.values, color='black', linewidth=2,
            label='Actual', alpha=0.8)

    # Plot predictions
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (model_name, predictions) in enumerate(predictions_dict.items()):
        color = colors[i % len(colors)]
        ax.plot(predictions.index, predictions.values, color=color,
                linewidth=1.5, linestyle='--', label=model_name, alpha=0.7)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax

def ensure_plot_dir(plot_dir='results/plots'):
    """Ensure plot directory exists."""
    path = Path(plot_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
