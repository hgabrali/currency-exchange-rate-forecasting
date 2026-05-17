"""
Currency Exchange Rate Forecasting Package

A complete pipeline for forecasting currency exchange rates for banking operations.
Supports ARIMA/SARIMA, LSTM deep learning, Prophet, and XGBoost ensemble models
with comprehensive evaluation metrics and visualization utilities.

Modules:
    data_preprocessing: Data loading, cleaning, feature engineering, LSTM windows.
    model_training: ARIMA, LSTM, Prophet, XGBoost, and Optuna tuning.
    visualization: Time series plots, forecast charts, residual analysis.
"""

from .data_preprocessing import (
    fetch_forex_data,
    generate_synthetic_forex_data,
    clean_forex_data,
    add_technical_indicators,
    add_lag_features,
    add_calendar_features,
    create_supervised_dataset,
    train_test_split_time_series,
    scale_data,
    run_preprocessing_pipeline,
    CURRENCY_PAIRS,
)

from .model_training import (
    fit_arima_model,
    forecast_arima,
    rolling_forecast_arima,
    build_lstm_model,
    build_bidirectional_lstm,
    train_lstm_model,
    build_xgboost_model,
    build_lightgbm_model,
    fit_prophet_model,
    forecast_prophet,
    compute_forecast_metrics,
    save_metrics,
    optimize_lstm_hyperparams,
    optimize_xgboost_hyperparams,
)

from .visualization import (
    plot_exchange_rate_history,
    plot_forecast_vs_actual,
    plot_forecast_interactive,
    plot_residuals,
    plot_model_comparison,
    plot_correlation_matrix,
    plot_future_forecast,
)

__version__ = "1.0.0"
__author__ = "Hande Gabrali-Knobloch"
__description__ = "Currency Exchange Rate Forecasting for Banking Operations"

__all__ = [
    # Data preprocessing
    "fetch_forex_data",
    "generate_synthetic_forex_data",
    "clean_forex_data",
    "add_technical_indicators",
    "add_lag_features",
    "add_calendar_features",
    "create_supervised_dataset",
    "train_test_split_time_series",
    "scale_data",
    "run_preprocessing_pipeline",
    "CURRENCY_PAIRS",
    # Model training
    "fit_arima_model",
    "forecast_arima",
    "rolling_forecast_arima",
    "build_lstm_model",
    "build_bidirectional_lstm",
    "train_lstm_model",
    "build_xgboost_model",
    "build_lightgbm_model",
    "fit_prophet_model",
    "forecast_prophet",
    "compute_forecast_metrics",
    "save_metrics",
    "optimize_lstm_hyperparams",
    "optimize_xgboost_hyperparams",
    # Visualization
    "plot_exchange_rate_history",
    "plot_forecast_vs_actual",
    "plot_forecast_interactive",
    "plot_residuals",
    "plot_model_comparison",
    "plot_correlation_matrix",
    "plot_future_forecast",
]
