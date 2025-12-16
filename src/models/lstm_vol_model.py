"""
Simple LSTM-based volatility forecasting model.

This class takes a series of returns, builds a rolling window dataset,
trains an LSTM, and produces forecasts of conditional volatility.

Author: Filippo Tiberi
"""

import numpy as np
import pandas as pd

from typing import Tuple

try:
    # tensorflow / keras might not be installed in all environments
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam
except ImportError as e:
    raise ImportError(
        "TensorFlow/Keras is required for LSTMVolatilityModel. "
        "Install it with: pip install tensorflow"
    ) from e


def _build_supervised_dataset(
    series: np.ndarray, window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) for supervised learning from a 1D time series.
    X: sequences of length `window_size`
    y: next-step conditional volatility (abs return or squared return).
    """
    series = np.asarray(series)
    vol_target = np.abs(series)  # you can also use series**2 as target

    X, y = [], []
    for t in range(len(series) - window_size):
        X.append(series[t : t + window_size])
        y.append(vol_target[t + window_size])

    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y


class LSTMVolatilityModel:
    """
    LSTM model for one-step-ahead volatility forecasting.

    Usage:
        model = LSTMVolatilityModel(window_size=20, epochs=20)
        model.fit(returns)
        vol_pred = model.in_sample_volatility_
        vol_forecast = model.forecast(steps=5)
    """

    def __init__(
        self,
        window_size: int = 20,
        lstm_units: int = 32,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        verbose: int = 0,
    ):
        self.window_size = window_size
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose

        self.model = None
        self.in_sample_volatility_ = None
        self.returns_ = None

    def _build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(self.lstm_units, input_shape=input_shape))
        model.add(Dense(1, activation="relu"))  # volatility must be >= 0
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse")
        return model

    def fit(self, returns: np.ndarray):
        """
        Fit the LSTM model on a series of returns.

        Parameters
        ----------
        returns : np.ndarray
            1D array of log-returns.
        """
        returns = np.asarray(returns, dtype=float)
        self.returns_ = returns

        X, y = _build_supervised_dataset(returns, self.window_size)
        self.model = self._build_model(input_shape=(self.window_size, 1))

        self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

        # in-sample volatility predictions aligned with original series
        vol_pred = self.model.predict(X, verbose=0).flatten()
        # prepend NaNs for the first `window_size` points
        vol_full = np.concatenate([np.full(self.window_size, np.nan), vol_pred])
        self.in_sample_volatility_ = vol_full

        return self

    def get_in_sample_volatility(self) -> np.ndarray:
        if self.in_sample_volatility_ is None:
            raise ValueError("Fit the model before calling get_in_sample_volatility().")
        return self.in_sample_volatility_

    def forecast(self, steps: int = 1) -> np.ndarray:
        """
        Multi-step volatility forecast using recursive strategy.
        """
        if self.model is None or self.returns_ is None:
            raise ValueError("Model must be fitted before forecasting.")

        window = self.returns_[-self.window_size :].copy()
        forecasts = []

        for _ in range(steps):
            x = window.reshape(1, self.window_size, 1)
            vol_pred = self.model.predict(x, verbose=0)[0, 0]
            forecasts.append(vol_pred)

            # here we append 0 as "return" placeholder or you can use random normal
            window = np.roll(window, -1)
            window[-1] = 0.0

        return np.array(forecasts)
