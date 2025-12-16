"""
Main execution script for FinRisk Analytics pipeline.

This script runs the complete workflow:
1. Fetch data
2. Preprocess returns
3. Fit models
4. Calculate VaR
5. Generate reports
"""

import yaml
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.data.fetch_data import fetch_all_assets, save_data
from src.data.preprocess import (prepare_returns, align_returns, split_data,
                                 save_processed_data, calculate_summary_statistics)
from src.models.garch_models import GARCHVolatility, EGARCHVolatility, GJRGARCHVolatility
from src.risk.var_calculator import VaRCalculator
from src.risk.backtester import VaRBacktester
from src.utils.logger import setup_logger, log_config, log_data_summary

def load_config(config_path='configs/config.yaml'):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def fetch_data_step(config, logger):
    """Step 1: Fetch data for all assets."""
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching Data")
    logger.info("=" * 60)

    data = fetch_all_assets(config)
    save_data(data, output_dir='data/raw')

    log_data_summary(logger, data)

    return data

def preprocess_data_step(data, config, logger):
    """Step 2: Preprocess data and calculate returns."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Preprocessing Data")
    logger.info("=" * 60)

    # Calculate returns
    returns_dict = prepare_returns(data)

    # Align returns
    returns_df = align_returns(returns_dict)

    logger.info(f"\nAligned returns shape: {returns_df.shape}")
    logger.info(f"Date range: {returns_df.index[0]} to {returns_df.index[-1]}")

    # Split data
    train, val, test = split_data(
        returns_df,
        config['data']['train_end'],
        config['data']['val_end']
    )

    logger.info(f"\nTrain set: {train.shape}")
    logger.info(f"Validation set: {val.shape}")
    logger.info(f"Test set: {test.shape}")

    # Save processed data
    save_processed_data(returns_df, 'all_returns.csv')

    return returns_df, train, val, test

def fit_garch_models_step(returns_df, config, logger):
    """Step 3: Fit GARCH models on case study assets."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Fitting GARCH Models")
    logger.info("=" * 60)

    case_studies = config['assets']['case_studies']
    models_fitted = {}

    for ticker in case_studies:
        if ticker not in returns_df.columns:
            logger.warning(f"Skipping {ticker} - not in data")
            continue

        logger.info(f"\nFitting models for {ticker}...")

        returns = returns_df[ticker]

        # Fit GARCH
        garch = GARCHVolatility()
        garch_result = garch.fit(returns)
        logger.info(f"  GARCH AIC: {garch_result.aic:.2f}")

        # Fit EGARCH
        egarch = EGARCHVolatility()
        egarch_result = egarch.fit(returns)
        logger.info(f"  EGARCH AIC: {egarch_result.aic:.2f}")

        # Fit GJR-GARCH
        gjr = GJRGARCHVolatility()
        gjr_result = gjr.fit(returns)
        logger.info(f"  GJR-GARCH AIC: {gjr_result.aic:.2f}")

        models_fitted[ticker] = {
            'garch': garch,
            'egarch': egarch,
            'gjr': gjr
        }

    return models_fitted

def calculate_var_step(returns_df, config, logger):
    """Step 4: Calculate VaR using different methods."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Calculating VaR")
    logger.info("=" * 60)

    case_studies = config['assets']['case_studies']
    var_results = {}

    for ticker in case_studies:
        if ticker not in returns_df.columns:
            continue

        logger.info(f"\nCalculating VaR for {ticker}...")

        returns = returns_df[ticker]

        # Calculate VaR using all methods
        hist_var = VaRCalculator.historical_var(returns, 0.95)
        param_var = VaRCalculator.parametric_var(returns, 0.95)
        param_t_var = VaRCalculator.parametric_var_t(returns, 0.95)
        cf_var = VaRCalculator.cornish_fisher_var(returns, 0.95)
        es = VaRCalculator.expected_shortfall(returns, 0.95)

        logger.info(f"  Historical VaR: {hist_var:.4f}")
        logger.info(f"  Parametric VaR: {param_var:.4f}")
        logger.info(f"  Parametric-t VaR: {param_t_var:.4f}")
        logger.info(f"  Cornish-Fisher VaR: {cf_var:.4f}")
        logger.info(f"  Expected Shortfall: {es:.4f}")

        var_results[ticker] = {
            'historical': hist_var,
            'parametric': param_var,
            'parametric_t': param_t_var,
            'cornish_fisher': cf_var,
            'expected_shortfall': es
        }

    return var_results

def backtest_var_step(returns_df, var_results, config, logger):
    """Step 5: Backtest VaR estimates."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Backtesting VaR")
    logger.info("=" * 60)

    case_studies = config['assets']['case_studies']

    for ticker in case_studies:
        if ticker not in returns_df.columns or ticker not in var_results:
            continue

        logger.info(f"\nBacktesting VaR for {ticker}...")

        returns = returns_df[ticker]

        # Use historical VaR for backtesting
        var_estimates = VaRCalculator.rolling_var(returns, confidence=0.95,
                                                   window=250, method='historical')

        # Align returns with var estimates
        aligned_returns = returns.loc[var_estimates.index]

        # Backtest
        violations = VaRBacktester.count_violations(aligned_returns, var_estimates)
        kupiec_result = VaRBacktester.kupiec_test(violations, len(aligned_returns), 0.95)

        logger.info(f"  Violations: {kupiec_result['violations']} / {len(aligned_returns)}")
        logger.info(f"  Expected: {kupiec_result['expected']:.2f}")
        logger.info(f"  Kupiec Test: {kupiec_result['result']}")
        logger.info(f"  P-value: {kupiec_result['p_value']:.4f}")

def main():
    """Main execution function."""
    # Setup logger
    logger = setup_logger()

    logger.info("Starting FinRisk Analytics Pipeline")
    logger.info("=" * 60)

    # Load configuration
    config = load_config()
    log_config(logger, config)

    # Run pipeline steps
    try:
        # Step 1: Fetch data
        data = fetch_data_step(config, logger)

        # Step 2: Preprocess
        returns_df, train, val, test = preprocess_data_step(data, config, logger)

        # Step 3: Fit GARCH models
        models = fit_garch_models_step(returns_df, config, logger)

        # Step 4: Calculate VaR
        var_results = calculate_var_step(returns_df, config, logger)

        # Step 5: Backtest VaR
        backtest_var_step(returns_df, var_results, config, logger)

        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
