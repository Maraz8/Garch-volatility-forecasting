"""LSTM model for volatility forecasting."""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle

class LSTMVolatility:
    """LSTM model for volatility prediction."""

    def __init__(self, sequence_length=20, units=64, dropout=0.2):
        self.sequence_length = sequence_length
        self.units = units
        self.dropout = dropout
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None

    def build_model(self, input_shape):
        """Build LSTM model architecture."""
        model = keras.Sequential([
            layers.LSTM(self.units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(self.dropout),
            layers.LSTM(self.units // 2, return_sequences=True),
            layers.Dropout(self.dropout),
            layers.LSTM(self.units // 4),
            layers.Dropout(self.dropout),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='relu')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model
        return model

    def prepare_sequences(self, returns, scale=True):
        """Prepare sequences for LSTM."""
        # Calculate realized volatility (rolling std)
        volatility = returns.rolling(window=5).std() * np.sqrt(252)  # Annualized
        volatility = volatility.dropna()

        if scale:
            volatility_scaled = self.scaler.fit_transform(volatility.values.reshape(-1, 1))
            volatility_values = volatility_scaled.flatten()
        else:
            volatility_values = volatility.values

        X, y = [], []
        for i in range(len(volatility_values) - self.sequence_length):
            X.append(volatility_values[i:i+self.sequence_length])
            y.append(volatility_values[i+self.sequence_length])

        X = np.array(X).reshape(-1, self.sequence_length, 1)
        y = np.array(y)

        return X, y, volatility.index[self.sequence_length:]

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=100, batch_size=32, model_path='results/models/lstm_vol.h5'):
        """
        Train LSTM model.

        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            model_path: Path to save best model

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], 1))

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss',
                         patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss' if X_val is not None else 'loss',
                             factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]

        # Add ModelCheckpoint only if model_path is provided
        if model_path:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                ModelCheckpoint(model_path, monitor='val_loss' if X_val is not None else 'loss',
                              save_best_only=True, verbose=1)
            )

        # Train model
        validation_data = (X_val, y_val) if X_val is not None else None

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def predict(self, X, unscale=True):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        predictions = self.model.predict(X, verbose=0)

        if unscale:
            predictions = self.scaler.inverse_transform(predictions)

        return predictions.flatten()

    def rolling_forecast(self, returns, train_size=252, refit_freq=60):
        """
        Generate rolling volatility forecasts.

        Args:
            returns: Full return series
            train_size: Initial training window
            refit_freq: Frequency to refit model (in days)

        Returns:
            Series of volatility forecasts
        """
        forecasts = []
        dates = []

        # Prepare initial data
        X_all, y_all, all_dates = self.prepare_sequences(returns, scale=True)

        for i in range(train_size, len(X_all)):
            # Refit model at specified frequency
            if (i - train_size) % refit_freq == 0:
                X_train = X_all[:i]
                y_train = y_all[:i]

                # Split for validation
                val_split = int(0.9 * len(X_train))
                X_tr, X_val = X_train[:val_split], X_train[val_split:]
                y_tr, y_val = y_train[:val_split], y_train[val_split:]

                # Train model
                self.train(X_tr, y_tr, X_val, y_val, epochs=50, batch_size=32,
                          model_path=None)

            # Forecast next period
            X_test = X_all[i:i+1]
            pred = self.predict(X_test, unscale=True)
            forecasts.append(pred[0])
            dates.append(all_dates[i])

        return pd.Series(forecasts, index=dates, name='LSTM_forecast')

    def save(self, model_path, scaler_path):
        """Save model and scaler."""
        if self.model is not None:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(model_path)

        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

    def load(self, model_path, scaler_path):
        """Load model and scaler."""
        self.model = keras.models.load_model(model_path)

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
