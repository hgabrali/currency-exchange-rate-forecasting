"""
visualization.py

Visualization utilities for currency exchange rate forecasting.
Creates time series plots, forecast charts, residual analysis,
model comparison dashboards, and interactive Plotly charts.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110


def plot_exchange_rate_history(
    df: pd.DataFrame,
    pair: str = "EUR/USD",
    price_col: str = "Close",
    save_path: Optional[str] = "reports/exchange_rate_history.png"
) -> None:
    """Plot historical exchange rate with volume and rolling statistics.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        pair: Currency pair label for title.
        price_col: Column to plot as price.
        save_path: Optional path to save figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Price
    axes[0].plot(df.index, df[price_col], color="royalblue", linewidth=1.2, label="Close")
    if "sma_20" in df.columns:
        axes[0].plot(df.index, df["sma_20"], color="orange", linewidth=1, linestyle="--", label="SMA-20")
    if "sma_50" in df.columns:
        axes[0].plot(df.index, df["sma_50"], color="red", linewidth=1, linestyle="--", label="SMA-50")
    if "bb_upper" in df.columns:
        axes[0].fill_between(df.index, df["bb_lower"], df["bb_upper"], alpha=0.15, color="gray", label="Bollinger Bands")
    axes[0].set_title(f"{pair} Exchange Rate History", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Rate")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Returns
    if "daily_return" in df.columns:
        returns = df["daily_return"].fillna(0)
        colors = ["green" if r > 0 else "red" for r in returns]
        axes[1].bar(df.index, returns * 100, color=colors, alpha=0.7, width=1)
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_ylabel("Daily Return (%)")
        axes[1].set_title("Daily Returns")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_visible(False)

    # Volatility
    if "volatility_20" in df.columns:
        axes[2].plot(df.index, df["volatility_20"] * 100, color="purple", linewidth=1)
        axes[2].fill_between(df.index, 0, df["volatility_20"] * 100, alpha=0.3, color="purple")
        axes[2].set_ylabel("20-Day Volatility (%)")
        axes[2].set_title("Rolling Volatility")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].set_visible(False)

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Exchange rate history saved to %s", save_path)
    plt.show()
    plt.close()


def plot_forecast_vs_actual(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    forecasts: Dict[str, np.ndarray],
    pair: str = "EUR/USD",
    save_path: Optional[str] = "reports/forecast_comparison.png"
) -> None:
    """Plot actual vs forecasted exchange rates for multiple models.

    Args:
        dates: DatetimeIndex for the test period.
        y_true: True exchange rate values.
        forecasts: Dict mapping model name to prediction array.
        pair: Currency pair label.
        save_path: Optional path to save figure.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(dates, y_true, color="black", linewidth=2, label="Actual", zorder=5)
    colors = ["royalblue", "orange", "green", "red", "purple", "brown"]
    for (model_name, preds), color in zip(forecasts.items(), colors):
        plt.plot(dates[:len(preds)], preds, color=color, linewidth=1.5,
                 linestyle="--", label=model_name, alpha=0.85)
    plt.title(f"{pair} — Forecast vs Actual (Test Period)", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Exchange Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Forecast comparison saved to %s", save_path)
    plt.show()
    plt.close()


def plot_forecast_interactive(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    forecasts: Dict[str, np.ndarray],
    pair: str = "EUR/USD",
    confidence_intervals: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    save_path: Optional[str] = "reports/forecast_interactive.html"
) -> go.Figure:
    """Create interactive Plotly chart of actual vs forecast.

    Args:
        dates: DatetimeIndex for the test period.
        y_true: True values.
        forecasts: Dict of model_name -> predictions.
        pair: Currency pair label.
        confidence_intervals: Optional dict of model_name -> (lower, upper) bands.
        save_path: Optional path to save HTML.

    Returns:
        go.Figure: Plotly figure.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=y_true,
        mode="lines",
        name="Actual",
        line=dict(color="black", width=2)
    ))
    colors = ["royalblue", "orange", "green", "red", "purple"]
    for (model_name, preds), color in zip(forecasts.items(), colors):
        n = min(len(preds), len(dates))
        fig.add_trace(go.Scatter(
            x=dates[:n], y=preds[:n],
            mode="lines",
            name=model_name,
            line=dict(color=color, width=1.5, dash="dash")
        ))
        if confidence_intervals and model_name in confidence_intervals:
            lower, upper = confidence_intervals[model_name]
            fig.add_trace(go.Scatter(
                x=list(dates[:n]) + list(dates[:n][::-1]),
                y=list(upper[:n]) + list(lower[:n][::-1]),
                fill="toself",
                fillcolor=f"rgba(0,0,255,0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                name=f"{model_name} CI",
                showlegend=True
            ))
    fig.update_layout(
        title=f"{pair} — Forecast vs Actual",
        xaxis_title="Date",
        yaxis_title="Exchange Rate",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        logger.info("Interactive forecast saved to %s", save_path)
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = "reports/residuals.png"
) -> None:
    """Plot residual analysis: residuals over time, histogram, QQ-plot, ACF.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        model_name: Model name for titles.
        save_path: Path to save figure.
    """
    residuals = np.array(y_true).flatten() - np.array(y_pred).flatten()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Residuals over time
    axes[0, 0].plot(residuals, color="steelblue", linewidth=0.8)
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Residuals Over Time")
    axes[0, 0].set_xlabel("Sample Index")
    axes[0, 0].set_ylabel("Residual")
    axes[0, 0].grid(True, alpha=0.3)

    # Histogram
    axes[0, 1].hist(residuals, bins=40, color="steelblue", edgecolor="black", alpha=0.8)
    axes[0, 1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Residuals Distribution")
    axes[0, 1].set_xlabel("Residual")
    axes[0, 1].set_ylabel("Frequency")

    # Actual vs Predicted
    axes[1, 0].scatter(y_pred, y_true, alpha=0.4, color="steelblue", s=10)
    min_v = min(np.min(y_true), np.min(y_pred))
    max_v = max(np.max(y_true), np.max(y_pred))
    axes[1, 0].plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5, label="Perfect Fit")
    axes[1, 0].set_title("Actual vs Predicted")
    axes[1, 0].set_xlabel("Predicted")
    axes[1, 0].set_ylabel("Actual")
    axes[1, 0].legend()

    # Residual autocorrelation
    try:
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(residuals, lags=40, ax=axes[1, 1], alpha=0.05)
        axes[1, 1].set_title("Residuals ACF")
    except Exception:
        pd.Series(residuals).autocorr_plot = None
        lags = range(1, min(41, len(residuals)))
        acf_vals = [pd.Series(residuals).autocorr(lag) for lag in lags]
        axes[1, 1].bar(list(lags), acf_vals, color="steelblue")
        axes[1, 1].axhline(0, color="black", linewidth=0.8)
        axes[1, 1].set_title("Residuals Autocorrelation")

    plt.suptitle(f"{model_name} — Residual Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Residuals plot saved to %s", save_path)
    plt.show()
    plt.close()


def plot_model_comparison(
    metrics_list: List[Dict],
    save_path: Optional[str] = "reports/model_comparison.html"
) -> go.Figure:
    """Create interactive bar chart comparing model performance metrics.

    Args:
        metrics_list: List of metric dicts, each with keys: model, mae, rmse, mape, r2.
        save_path: Optional path to save HTML.

    Returns:
        go.Figure: Plotly figure.
    """
    df = pd.DataFrame(metrics_list)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("MAE (lower is better)", "RMSE (lower is better)",
                        "MAPE % (lower is better)", "R2 Score (higher is better)")
    )
    colors = px.colors.qualitative.Set2[:len(df)]
    for metric, (row, col) in [("mae", (1, 1)), ("rmse", (1, 2)), ("mape", (2, 1)), ("r2", (2, 2))]:
        if metric in df.columns:
            fig.add_trace(go.Bar(
                x=df["model"], y=df[metric],
                marker_color=colors,
                showlegend=False
            ), row=row, col=col)
    fig.update_layout(
        title="Currency Forecasting Model Comparison",
        height=600,
    )
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        logger.info("Model comparison saved to %s", save_path)
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    features: List[str] = None,
    top_n: int = 20,
    save_path: Optional[str] = "reports/correlation_matrix.png"
) -> None:
    """Plot a heatmap of feature correlation matrix.

    Args:
        df: DataFrame with numeric features.
        features: List of column names to include (default: all numeric).
        top_n: Maximum number of features to show.
        save_path: Path to save figure.
    """
    numeric_df = df.select_dtypes(include=np.number)
    if features:
        numeric_df = numeric_df[[c for c in features if c in numeric_df.columns]]
    if len(numeric_df.columns) > top_n:
        # Select top_n most correlated features to Close
        if "Close" in numeric_df.columns:
            corrs = numeric_df.corr()["Close"].abs().sort_values(ascending=False)
            top_cols = corrs.head(top_n).index.tolist()
            numeric_df = numeric_df[top_cols]
        else:
            numeric_df = numeric_df.iloc[:, :top_n]
    corr = numeric_df.corr()
    fig_size = max(10, len(corr) // 2)
    plt.figure(figsize=(fig_size, fig_size))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=len(corr) <= 15,
        fmt=".2f", cmap="RdYlGn",
        center=0, square=True,
        linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    plt.title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Correlation matrix saved to %s", save_path)
    plt.show()
    plt.close()


def plot_future_forecast(
    historical_df: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    forecast_values: np.ndarray,
    pair: str = "EUR/USD",
    model_name: str = "Model",
    conf_lower: Optional[np.ndarray] = None,
    conf_upper: Optional[np.ndarray] = None,
    save_path: Optional[str] = "reports/future_forecast.html"
) -> go.Figure:
    """Plot historical data + future forecast with optional confidence interval.

    Args:
        historical_df: Historical DataFrame with DatetimeIndex and Close column.
        forecast_dates: DatetimeIndex for the forecast period.
        forecast_values: Predicted values for the forecast period.
        pair: Currency pair name.
        model_name: Model name.
        conf_lower: Lower confidence bound (optional).
        conf_upper: Upper confidence bound (optional).
        save_path: Optional path to save interactive HTML.

    Returns:
        go.Figure: Plotly figure.
    """
    fig = go.Figure()
    n_hist = min(len(historical_df), 500)
    hist = historical_df.tail(n_hist)
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"],
        mode="lines", name="Historical",
        line=dict(color="royalblue", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_values,
        mode="lines+markers", name=f"Forecast ({model_name})",
        line=dict(color="orange", width=2, dash="dash"),
        marker=dict(size=5)
    ))
    if conf_lower is not None and conf_upper is not None:
        fig.add_trace(go.Scatter(
            x=list(forecast_dates) + list(forecast_dates[::-1]),
            y=list(conf_upper) + list(conf_lower[::-1]),
            fill="toself",
            fillcolor="rgba(255, 165, 0, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI"
        ))
    # Vertical line at forecast start
    fig.add_vline(
        x=str(forecast_dates[0]),
        line_dash="dot", line_color="gray",
        annotation_text="Forecast Start"
    )
    fig.update_layout(
        title=f"{pair} — {model_name} Future Forecast",
        xaxis_title="Date",
        yaxis_title="Exchange Rate",
        height=500,
        hovermode="x",
    )
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        logger.info("Future forecast saved to %s", save_path)
    return fig


if __name__ == "__main__":
    logger.info("Running visualization demo with synthetic data...")
    dates = pd.bdate_range("2023-01-01", periods=100)
    y_true = 1.08 + np.cumsum(np.random.normal(0, 0.002, 100))
    forecasts = {
        "ARIMA": y_true + np.random.normal(0, 0.003, 100),
        "LSTM": y_true + np.random.normal(0, 0.005, 100),
        "XGBoost": y_true + np.random.normal(0, 0.004, 100),
    }
    fig = plot_forecast_interactive(dates, y_true, forecasts, pair="EUR/USD", save_path=None)
    logger.info("Visualization demo complete.")
