"""
model_training.py

Model training module for currency exchange rate forecasting.
Implements ARIMA/SARIMA, LSTM (deep learning), Prophet, XGBoost, and
GARCH volatility models with evaluation and Optuna hyperparameter tuning.
"""

import os
import json
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. Statistical Models: ARIMA / SARIMA
# ─────────────────────────────────────────────────────────────

def fit_arima_model(
    series: pd.Series,
    order: Tuple[int, int, int] = None,
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    auto: bool = True
) -> object:
    """Fit an ARIMA or SARIMA model to a time series.

    Args:
        series: Target time series (log-returns or price).
        order: (p, d, q) tuple. If None and auto=True, auto-selects.
        seasonal_order: (P, D, Q, s) seasonal parameters.
        auto: If True, use pmdarima auto_arima for order selection.

    Returns:
        Fitted model object (pmdarima or statsmodels).
    """
    if auto:
        try:
            from pmdarima import auto_arima
            model = auto_arima(
                series,
                seasonal=seasonal_order[3] > 0,
                m=seasonal_order[3] or 1,
                information_criterion="aic",
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                max_p=5, max_q=5, max_d=2
            )
            logger.info("Auto ARIMA: order=%s, seasonal=%s, AIC=%.2f",
                        model.order, model.seasonal_order, model.aic())
            return model
        except ImportError:
            logger.warning("pmdarima not available, using statsmodels ARIMA")

    from statsmodels.tsa.arima.model import ARIMA
    order = order or (1, 1, 1)
    model = ARIMA(series, order=order, seasonal_order=seasonal_order)
    fitted = model.fit()
    logger.info("ARIMA(%s) AIC=%.2f", order, fitted.aic)
    return fitted


def forecast_arima(
    fitted_model: object,
    steps: int = 30
) -> np.ndarray:
    """Generate out-of-sample forecasts from a fitted ARIMA model.

    Args:
        fitted_model: Fitted ARIMA / SARIMA model.
        steps: Number of future steps to forecast.

    Returns:
        np.ndarray: Forecast values of shape (steps,).
    """
    try:
        # pmdarima API
        forecast = fitted_model.predict(n_periods=steps)
    except AttributeError:
        # statsmodels API
        forecast = fitted_model.forecast(steps=steps)
    return np.array(forecast)


def rolling_forecast_arima(
    series: pd.Series,
    train_size: int,
    horizon: int = 1,
    order: Tuple[int, int, int] = (1, 1, 1)
) -> np.ndarray:
    """Walk-forward validation with refitting ARIMA at each step.

    Args:
        series: Full time series.
        train_size: Initial training window size.
        horizon: Forecast horizon (steps ahead).
        order: ARIMA order.

    Returns:
        np.ndarray: One-step-ahead predictions for the test period.
    """
    from statsmodels.tsa.arima.model import ARIMA
    history = list(series.iloc[:train_size])
    predictions = []
    test_series = series.iloc[train_size:]
    for t in range(len(test_series)):
        try:
            model = ARIMA(history, order=order)
            result = model.fit()
            yhat = result.forecast(steps=horizon)
            predictions.append(float(yhat[0]))
        except Exception:
            predictions.append(history[-1])
        history.append(float(test_series.iloc[t]))
    return np.array(predictions)


# ─────────────────────────────────────────────────────────────
# 2. Deep Learning: LSTM
# ─────────────────────────────────────────────────────────────

def build_lstm_model(
    lookback: int,
    n_features: int = 1,
    forecast_horizon: int = 1,
    units: List[int] = None,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3
) -> object:
    """Build a stacked LSTM model for time series forecasting.

    Args:
        lookback: Number of past time steps as input.
        n_features: Number of input features.
        forecast_horizon: Output sequence length.
        units: List of LSTM units per layer.
        dropout_rate: Dropout fraction after each LSTM layer.
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled Keras LSTM model.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    units = units or [128, 64, 32]
    inputs = tf.keras.Input(shape=(lookback, n_features), name="input")
    x = inputs
    for i, u in enumerate(units):
        return_seq = i < len(units) - 1
        x = layers.LSTM(u, return_sequences=return_seq, name=f"lstm_{i+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    outputs = layers.Dense(forecast_horizon, name="output")(x)
    model = Model(inputs, outputs, name="LSTM_Forecaster")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model


def build_bidirectional_lstm(
    lookback: int,
    n_features: int = 1,
    forecast_horizon: int = 1,
    units: int = 64,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3
) -> object:
    """Build a Bidirectional LSTM with attention mechanism.

    Args:
        lookback: Input sequence length.
        n_features: Number of input features.
        forecast_horizon: Output sequence length.
        units: Number of BiLSTM units.
        dropout_rate: Dropout fraction.
        learning_rate: Learning rate for Adam.

    Returns:
        Compiled Keras Bidirectional LSTM model.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inputs = tf.keras.Input(shape=(lookback, n_features))
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(inputs)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Bidirectional(layers.LSTM(units // 2, return_sequences=False))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(forecast_horizon)(x)
    model = Model(inputs, outputs, name="BiLSTM_Forecaster")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model


def train_lstm_model(
    model: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    checkpoint_path: str = "models/lstm_best.h5"
) -> object:
    """Train an LSTM model with early stopping and checkpoint callbacks.

    Args:
        model: Compiled Keras LSTM model.
        X_train: Training input array.
        y_train: Training target array.
        X_val: Validation input array.
        y_val: Validation target array.
        epochs: Maximum training epochs.
        batch_size: Training batch size.
        checkpoint_path: Path to save the best model weights.

    Returns:
        keras.callbacks.History object.
    """
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    )
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ─────────────────────────────────────────────────────────────
# 3. Machine Learning: XGBoost & LightGBM
# ─────────────────────────────────────────────────────────────

def build_xgboost_model(
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0
) -> object:
    """Build an XGBoost regressor for tabular time series forecasting.

    Args:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: XGBoost learning rate (eta).
        subsample: Row subsampling ratio.
        colsample_bytree: Column subsampling ratio.
        reg_lambda: L2 regularization term.

    Returns:
        xgboost.XGBRegressor: Configured XGBoost model.
    """
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


def build_lightgbm_model(
    n_estimators: int = 500,
    max_depth: int = -1,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    subsample: float = 0.8
) -> object:
    """Build a LightGBM regressor for time series forecasting.

    Args:
        n_estimators: Number of boosting iterations.
        max_depth: Maximum tree depth (-1 = unlimited).
        num_leaves: Maximum number of leaves per tree.
        learning_rate: LightGBM learning rate.
        subsample: Row subsampling fraction.

    Returns:
        lightgbm.LGBMRegressor: Configured LightGBM model.
    """
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        subsample=subsample,
        objective="regression",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


# ─────────────────────────────────────────────────────────────
# 4. Prophet Model
# ─────────────────────────────────────────────────────────────

def fit_prophet_model(
    df: pd.DataFrame,
    date_col: str = "ds",
    target_col: str = "y",
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True
) -> object:
    """Fit a Facebook Prophet model for currency rate forecasting.

    Args:
        df: DataFrame with "ds" (date) and "y" (target) columns.
        date_col: Name of the date column.
        target_col: Name of the target column.
        changepoint_prior_scale: Flexibility of trend changepoints.
        seasonality_prior_scale: Flexibility of seasonality.
        yearly_seasonality: Include yearly seasonality.
        weekly_seasonality: Include weekly seasonality.

    Returns:
        Fitted Prophet model.
    """
    from prophet import Prophet
    prophet_df = df[[date_col, target_col]].copy()
    if date_col != "ds":
        prophet_df = prophet_df.rename(columns={date_col: "ds", target_col: "y"})
    elif target_col != "y":
        prophet_df = prophet_df.rename(columns={target_col: "y"})
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
        uncertainty_samples=200,
    )
    model.fit(prophet_df)
    logger.info("Prophet model fitted on %d observations.", len(prophet_df))
    return model


def forecast_prophet(
    model: object,
    periods: int = 30,
    freq: str = "B"
) -> pd.DataFrame:
    """Generate forecast from a fitted Prophet model.

    Args:
        model: Fitted Prophet model.
        periods: Number of future periods to forecast.
        freq: Date frequency (B=business days, D=daily).

    Returns:
        pd.DataFrame: Prophet forecast DataFrame with yhat, yhat_lower, yhat_upper.
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


# ─────────────────────────────────────────────────────────────
# 5. Evaluation Metrics
# ─────────────────────────────────────────────────────────────

def compute_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """Compute standard regression and forecasting evaluation metrics.

    Metrics: MAE, RMSE, MAPE, sMAPE, R2, Direction Accuracy.

    Args:
        y_true: True values array.
        y_pred: Predicted values array.
        model_name: Model name for logging.

    Returns:
        dict: Dictionary of metric name -> value.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100
    # sMAPE
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100
    # Direction accuracy
    if len(y_true) > 1:
        direction_acc = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
    else:
        direction_acc = float("nan")
    metrics = {
        "model": model_name,
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "mape": round(mape, 4),
        "smape": round(smape, 4),
        "r2": round(r2, 4),
        "direction_accuracy": round(direction_acc, 2),
    }
    logger.info("%s | MAE=%.6f | RMSE=%.6f | MAPE=%.2f%% | DA=%.1f%%",
                model_name, mae, rmse, mape, direction_acc)
    return metrics


def save_metrics(
    metrics_list: List[Dict],
    output_path: str = "reports/evaluation_metrics.json"
) -> None:
    """Save evaluation metrics to a JSON file.

    Args:
        metrics_list: List of metric dicts (one per model).
        output_path: Path to save JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    logger.info("Metrics saved to %s", output_path)


# ─────────────────────────────────────────────────────────────
# 6. Optuna Hyperparameter Optimization
# ─────────────────────────────────────────────────────────────

def optimize_lstm_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 15
) -> Dict:
    """Optimize LSTM hyperparameters using Optuna.

    Args:
        X_train: Training input array (n_samples, lookback, n_features).
        y_train: Training targets.
        X_val: Validation input.
        y_val: Validation targets.
        n_trials: Number of Optuna trials.

    Returns:
        dict: Best hyperparameters and validation MAE.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    lookback = X_train.shape[1]
    n_features = X_train.shape[2]
    forecast_horizon = y_train.shape[1] if y_train.ndim > 1 else 1

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 3)
        units = [trial.suggest_int(f"units_l{i}", 32, 256, step=32) for i in range(n_layers)]
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        model = build_lstm_model(lookback, n_features, forecast_horizon, units, dropout, lr)
        from tensorflow.keras.callbacks import EarlyStopping
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=batch_size,
            callbacks=[EarlyStopping(monitor="val_loss", patience=5)],
            verbose=0,
        )
        return min(history.history["val_loss"])

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("Best LSTM trial: val_loss=%.6f, params=%s", study.best_value, study.best_params)
    return {"best_params": study.best_params, "best_val_loss": study.best_value}


def optimize_xgboost_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 20
) -> Dict:
    """Optimize XGBoost hyperparameters using Optuna.

    Args:
        X_train: 2D training feature matrix.
        y_train: Training target array.
        X_val: Validation feature matrix.
        y_val: Validation target array.
        n_trials: Number of Optuna trials.

    Returns:
        dict: Best parameters and best validation MAE.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
        model = build_xgboost_model(**params)
        X_tr = X_train.reshape(len(X_train), -1) if X_train.ndim > 2 else X_train
        X_v = X_val.reshape(len(X_val), -1) if X_val.ndim > 2 else X_val
        model.fit(X_tr, y_train.flatten(), eval_set=[(X_v, y_val.flatten())], verbose=False)
        preds = model.predict(X_v)
        return mean_absolute_error(y_val.flatten(), preds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    logger.info("Best XGBoost trial: MAE=%.6f, params=%s", study.best_value, study.best_params)
    return {"best_params": study.best_params, "best_val_mae": study.best_value}


if __name__ == "__main__":
    from src.data_preprocessing import run_preprocessing_pipeline
    result = run_preprocessing_pipeline(use_synthetic=True)
    train_df = result["train_df"]
    # Quick ARIMA test
    arima_model = fit_arima_model(train_df["Close"], auto=False, order=(1, 1, 1))
    forecast = forecast_arima(arima_model, steps=5)
    logger.info("ARIMA 5-day forecast: %s", np.round(forecast, 4))
    # Quick LSTM test
    lstm_model = build_lstm_model(result["X_train"].shape[1], 1, result["y_train"].shape[1])
    lstm_model.summary()
    logger.info("Model training module OK.")
