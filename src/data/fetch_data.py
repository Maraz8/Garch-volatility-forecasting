"""Data fetching module for stocks, ETFs, crypto, and VIX."""

import yfinance as yf
import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch OHLCV data for stocks/ETFs."""
    print(f"Fetching {ticker}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            print(f"Warning: No data retrieved for {ticker}")
            return None
        return data
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def fetch_crypto_data(symbol, start_date, end_date, exchange_name='binance'):
    """
    Fetch OHLCV data for cryptocurrencies.

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        start_date: Start date string
        end_date: End date string
        exchange_name: Exchange to use (default: binance)
    """
    print(f"Fetching {symbol} from {exchange_name}...")
    try:
        exchange = getattr(ccxt, exchange_name)()

        # Convert dates to timestamps
        since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        # Fetch OHLCV data
        all_ohlcv = []
        current = since

        while current < end_ts:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=current, limit=1000)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                current = ohlcv[-1][0] + 86400000  # Move to next day
                time.sleep(exchange.rateLimit / 1000)  # Rate limiting
            except Exception as e:
                print(f"Error fetching batch: {e}")
                break

        if not all_ohlcv:
            print(f"Warning: No data retrieved for {symbol}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        return df

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def fetch_vix_data(start_date, end_date):
    """Fetch VIX (volatility index) data."""
    print("Fetching VIX...")
    return fetch_stock_data('^VIX', start_date, end_date)

def fetch_all_assets(config):
    """
    Fetch all assets from config file.

    Args:
        config: Dictionary with asset lists and date ranges

    Returns:
        Dictionary of DataFrames keyed by ticker
    """
    data = {}

    start_date = config['data']['start_date']
    end_date = config['data']['end_date']

    # Fetch equities
    for ticker in config['assets']['equities']:
        df = fetch_stock_data(ticker, start_date, end_date)
        if df is not None:
            data[ticker] = df

    # Fetch fixed income
    for ticker in config['assets']['fixed_income']:
        df = fetch_stock_data(ticker, start_date, end_date)
        if df is not None:
            data[ticker] = df

    # Fetch alternatives (GLD)
    for ticker in config['assets']['alternatives']:
        df = fetch_stock_data(ticker, start_date, end_date)
        if df is not None:
            data[ticker] = df

    # Fetch crypto
    for symbol in config['assets']['crypto']:
        # Convert BTC/USDT format to BTC for storage key
        ticker_key = symbol.split('/')[0]
        df = fetch_crypto_data(symbol, start_date, end_date)
        if df is not None:
            data[ticker_key] = df

    # Fetch VIX
    for ticker in config['assets']['macro']:
        df = fetch_stock_data(ticker, start_date, end_date)
        if df is not None:
            # Store VIX without ^ prefix
            data['VIX'] = df

    return data

def save_data(data, output_dir='data/raw'):
    """Save fetched data to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for ticker, df in data.items():
        file_path = output_path / f"{ticker}.csv"
        df.to_csv(file_path)
        print(f"Saved {ticker} to {file_path}")

def load_data(ticker, data_dir='data/raw'):
    """Load data from CSV file."""
    file_path = Path(data_dir) / f"{ticker}.csv"
    if file_path.exists():
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        print(f"File not found: {file_path}")
        return None

def calculate_returns(prices):
    """Calculate log returns."""
    returns = np.log(prices / prices.shift(1))
    return returns.dropna()
