"""
models/lstm_model.py
─────────────────────
Multi-layer LSTM with dropout, early stopping, and optional GRU comparison.
Handles training, evaluation, multi-step forecasting, and model persistence.
"""

import numpy as np
import os
import joblib
import logging
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, BatchNormalization, Input, Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Model Builders ─────────────────────────────────────────────────────────────
def build_lstm_model(
    input_shape: tuple,
    units: list = [128, 64, 32],
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    bidirectional: bool = False,
) -> keras.Model:
    """
    Build a stacked LSTM model.

    Architecture:
      Input → [BiLSTM/LSTM × N layers] → BatchNorm → Dropout → Dense(1)

    Parameters
    ----------
    input_shape   : (window_size, n_features)
    units         : list of unit counts per LSTM layer
    dropout       : fraction of neurons to drop
    learning_rate : Adam LR
    bidirectional : wrap each LSTM in Bidirectional wrapper
    """
    model = Sequential(name="StockVision_LSTM")
    model.add(Input(shape=input_shape))

    for i, u in enumerate(units):
        return_seq = (i < len(units) - 1)
        if bidirectional:
            model.add(Bidirectional(LSTM(u, return_sequences=return_seq)))
        else:
            model.add(LSTM(u, return_sequences=return_seq))

        model.add(BatchNormalization())
        if i < len(units) - 1:
            model.add(Dropout(dropout))

    model.add(Dropout(dropout))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="huber",
        metrics=["mae"],
    )
    return model


def build_gru_model(
    input_shape: tuple,
    units: list = [128, 64],
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Lightweight GRU baseline for comparison."""
    model = Sequential(name="StockVision_GRU")
    model.add(Input(shape=input_shape))

    for i, u in enumerate(units):
        return_seq = (i < len(units) - 1)
        model.add(GRU(u, return_sequences=return_seq))
        model.add(Dropout(dropout))

    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="huber", metrics=["mae"])
    return model


# ── Training Pipeline ──────────────────────────────────────────────────────────
def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    ticker: str = "STOCK",
) -> tuple:
    """
    Train the model with callbacks.

    Returns (history, trained_model)
    """
    checkpoint_path = os.path.join(MODEL_DIR, f"{ticker}_best.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=0,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
        shuffle=False,  # preserve time order
    )

    logger.info(f"Training complete. Best val_loss: {min(history.history['val_loss']):.6f}")
    return history, model


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_target,
) -> dict:
    """
    Compute RMSE, MAE, MAPE, R² on the test set.

    Returns dict with raw and scaled metrics plus predictions.
    """
    preds_scaled = model.predict(X_test, verbose=0).flatten()

    preds = scaler_target.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    actual = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(actual, preds))
    mae  = mean_absolute_error(actual, preds)
    r2   = r2_score(actual, preds)
    mape = np.mean(np.abs((actual - preds) / (actual + 1e-8))) * 100
    acc  = max(0, 100 - mape)

    return {
        "RMSE":        round(rmse, 4),
        "MAE":         round(mae, 4),
        "MAPE":        round(mape, 4),
        "R2":          round(r2, 4),
        "Accuracy":    round(acc, 2),
        "predictions": preds,
        "actual":      actual,
    }


# ── Multi-Step Forecasting ─────────────────────────────────────────────────────
def forecast_future(
    model: keras.Model,
    last_sequence: np.ndarray,
    scaler_target,
    n_steps: int = 30,
) -> np.ndarray:
    """
    Iterative multi-step forecast.

    last_sequence : shape (window, n_features) — last known window (scaled)
    Returns array of n_steps future price predictions.
    """
    seq = last_sequence.copy()
    future_preds = []

    for _ in range(n_steps):
        inp = seq[np.newaxis, :, :]                   # (1, window, features)
        pred_scaled = model.predict(inp, verbose=0)[0, 0]
        future_preds.append(pred_scaled)

        # Shift window: drop oldest, append new prediction as last feature
        new_row = seq[-1].copy()
        new_row[-1] = pred_scaled                     # update close (last feature)
        seq = np.vstack([seq[1:], new_row])

    future_preds = np.array(future_preds)
    return scaler_target.inverse_transform(future_preds.reshape(-1, 1)).flatten()


# ── Model Persistence ──────────────────────────────────────────────────────────
def save_model(model: keras.Model, scaler_f, scaler_t, ticker: str, config: dict):
    """Save model weights + scalers + config to disk."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    base = os.path.join(MODEL_DIR, f"{ticker}_{ts}")

    model.save(f"{base}.keras")
    joblib.dump(scaler_f, f"{base}_scaler_features.pkl")
    joblib.dump(scaler_t, f"{base}_scaler_target.pkl")
    joblib.dump(config,   f"{base}_config.pkl")
    logger.info(f"Model saved: {base}")
    return base


def load_saved_model(base_path: str):
    """Load a previously saved model + scalers."""
    model    = load_model(f"{base_path}.keras")
    scaler_f = joblib.load(f"{base_path}_scaler_features.pkl")
    scaler_t = joblib.load(f"{base_path}_scaler_target.pkl")
    config   = joblib.load(f"{base_path}_config.pkl")
    return model, scaler_f, scaler_t, config


def list_saved_models() -> list:
    """Return list of (display_name, base_path) tuples for saved models."""
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]
    return [(f.replace(".keras", ""), os.path.join(MODEL_DIR, f.replace(".keras", "")))
            for f in sorted(files, reverse=True)]
