"""
data_preprocessing.py

Data preprocessing pipeline for currency exchange rate forecasting.
Handles data loading (yfinance / CSV), cleaning, feature engineering,
lag features, technical indicators, and train/test splitting.
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Major currency pairs for banking operations
CURRENCY_PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X",
    "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "EURGBP=X",
    "EURJPY=X", "GBPJPY=X",
]


def fetch_forex_data(
    pairs: List[str] = None,
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    interval: str = "1d"
) -> Dict[str, pd.DataFrame]:
    """Fetch historical forex data from Yahoo Finance.

    Args:
        pairs: List of currency pair tickers (e.g., ["EURUSD=X"]).
        start_date: Start date string YYYY-MM-DD.
        end_date: End date string YYYY-MM-DD.
        interval: Data frequency (1d, 1wk, 1mo).

    Returns:
        dict: Mapping from pair ticker to OHLCV DataFrame.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    pairs = pairs or CURRENCY_PAIRS
    data = {}
    for pair in pairs:
        try:
            ticker = yf.Ticker(pair)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            if df.empty:
                logger.warning("No data returned for %s", pair)
                continue
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"
            data[pair] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            logger.info("Fetched %d rows for %s", len(df), pair)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", pair, exc)
    return data


def generate_synthetic_forex_data(
    pair: str = "EURUSD=X",
    n_days: int = 2000,
    start_date: str = "2018-01-01",
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic forex OHLCV data for testing without internet.

    Uses a geometric Brownian motion model to simulate realistic price paths.

    Args:
        pair: Currency pair name for labeling.
        n_days: Number of trading days to generate.
        start_date: Starting date string.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame: Synthetic OHLCV DataFrame with DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    # Geometric Brownian motion parameters
    mu = 0.00002    # daily drift
    sigma = 0.006   # daily volatility
    dt = 1
    returns = rng.normal(mu * dt, sigma * np.sqrt(dt), n_days)
    prices = 1.1000 * np.exp(np.cumsum(returns))
    # Build OHLCV
    daily_vol = rng.uniform(0.002, 0.008, n_days)
    opens = prices * (1 + rng.normal(0, 0.001, n_days))
    highs = prices * (1 + np.abs(rng.normal(0, daily_vol, n_days)))
    lows = prices * (1 - np.abs(rng.normal(0, daily_vol, n_days)))
    volumes = rng.integers(50_000, 500_000, n_days).astype(float)
    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows,
        "Close": prices, "Volume": volumes
    }, index=pd.DatetimeIndex(dates, name="Date"))
    logger.info("Generated %d synthetic rows for %s", len(df), pair)
    return df


def clean_forex_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw forex OHLCV data.

    Steps: sort index, remove duplicates, fill weekend gaps with forward fill,
    remove outliers (Close > 5 stdevs from rolling mean).

    Args:
        df: Raw OHLCV DataFrame with DatetimeIndex.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = df.copy()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(subset=["Close"])
    # Forward-fill up to 3 consecutive missing days
    df = df.ffill(limit=3)
    # Remove statistical outliers
    rolling_mean = df["Close"].rolling(30, min_periods=5).mean()
    rolling_std = df["Close"].rolling(30, min_periods=5).std()
    outlier_mask = np.abs(df["Close"] - rolling_mean) > 5 * rolling_std
    n_outliers = outlier_mask.sum()
    if n_outliers > 0:
        logger.warning("Removed %d outlier rows.", n_outliers)
        df = df[~outlier_mask]
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical analysis features to OHLCV data.

    Features added:
    - Returns: daily, log, weekly
    - Moving averages: SMA 5/10/20/50/200, EMA 12/26
    - Momentum: RSI, MACD, ROC
    - Volatility: ATR, Bollinger Bands, rolling std
    - Volume: VWAP, OBV

    Args:
        df: Cleaned OHLCV DataFrame.

    Returns:
        pd.DataFrame: DataFrame with added technical features.
    """
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Returns
    df["daily_return"] = close.pct_change()
    df["log_return"] = np.log(close / close.shift(1))
    df["weekly_return"] = close.pct_change(5)
    df["monthly_return"] = close.pct_change(21)

    # Simple Moving Averages
    for window in [5, 10, 20, 50, 200]:
        df[f"sma_{window}"] = close.rolling(window).mean()
        df[f"close_vs_sma{window}"] = close / df[f"sma_{window}"] - 1

    # EMA and MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["ema_12"] = ema12
    df["ema_26"] = ema26
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    sma20 = df["sma_20"]
    std20 = close.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
    df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

    # ATR (Average True Range)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / close

    # Rate of Change
    df["roc_5"] = close.pct_change(5)
    df["roc_10"] = close.pct_change(10)

    # Rolling volatility
    df["volatility_10"] = close.pct_change().rolling(10).std()
    df["volatility_20"] = close.pct_change().rolling(20).std()
    df["volatility_60"] = close.pct_change().rolling(60).std()

    # Volume indicators
    df["volume_sma_20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / (df["volume_sma_20"] + 1e-10)
    df["obv"] = (np.sign(close.diff()) * volume).fillna(0).cumsum()

    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "Close",
    lags: List[int] = None
) -> pd.DataFrame:
    """Add lag features of the target variable.

    Args:
        df: DataFrame with time series data.
        target_col: Column name for the target variable.
        lags: List of lag periods to create.

    Returns:
        pd.DataFrame: DataFrame with added lag features.
    """
    lags = lags or [1, 2, 3, 5, 7, 10, 14, 21]
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features from the DatetimeIndex.

    Features: day of week, month, quarter, day of year, is month-end, is quarter-end.

    Args:
        df: DataFrame with DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame with calendar features.
    """
    df = df.copy()
    idx = df.index
    df["day_of_week"] = idx.dayofweek
    df["month"] = idx.month
    df["quarter"] = idx.quarter
    df["day_of_year"] = idx.dayofyear
    df["week_of_year"] = idx.isocalendar().week.values.astype(int)
    df["is_month_end"] = idx.is_month_end.astype(int)
    df["is_quarter_end"] = idx.is_quarter_end.astype(int)
    df["is_year_end"] = idx.is_year_end.astype(int)
    # Sine/Cosine encoding for cyclical features
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 5)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 5)
    return df


def create_supervised_dataset(
    series: np.ndarray,
    lookback: int = 60,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Create supervised (X, y) dataset from a time series for LSTM / ML models.

    Args:
        series: 1D or 2D array of shape (n_timesteps,) or (n_timesteps, n_features).
        lookback: Number of past time steps to use as input.
        forecast_horizon: Number of future steps to predict.

    Returns:
        Tuple of (X, y) arrays:
            X: shape (n_samples, lookback, n_features) for LSTM
            y: shape (n_samples, forecast_horizon)
    """
    if series.ndim == 1:
        series = series.reshape(-1, 1)
    X, y = [], []
    n = len(series)
    for i in range(lookback, n - forecast_horizon + 1):
        X.append(series[i - lookback:i])
        y.append(series[i:i + forecast_horizon, 0])
    return np.array(X), np.array(y)


def train_test_split_time_series(
    df: pd.DataFrame,
    test_ratio: float = 0.15,
    val_ratio: float = 0.10
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time series data into train / val / test sets chronologically.

    Args:
        df: Full DataFrame with DatetimeIndex.
        test_ratio: Proportion for test set.
        val_ratio: Proportion for validation set.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    n = len(df)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_df = df.iloc[:n - test_size - val_size]
    val_df = df.iloc[n - test_size - val_size:n - test_size]
    test_df = df.iloc[n - test_size:]
    logger.info("Split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


def scale_data(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    method: str = "minmax"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Scale data using MinMax or Standard scaler fitted on training set only.

    Args:
        train: Training array.
        val: Validation array.
        test: Test array.
        method: "minmax" or "standard".

    Returns:
        Tuple of (train_scaled, val_scaled, test_scaled, scaler).
    """
    scaler = MinMaxScaler(feature_range=(0, 1)) if method == "minmax" else StandardScaler()
    if train.ndim == 1:
        train = train.reshape(-1, 1)
        val = val.reshape(-1, 1)
        test = test.reshape(-1, 1)
    train_s = scaler.fit_transform(train)
    val_s = scaler.transform(val)
    test_s = scaler.transform(test)
    return train_s, val_s, test_s, scaler


def run_preprocessing_pipeline(
    pair: str = "EURUSD=X",
    start_date: str = "2018-01-01",
    end_date: str = "2024-12-31",
    lookback: int = 60,
    forecast_horizon: int = 5,
    use_synthetic: bool = True,
    output_dir: str = "data/processed"
) -> Dict:
    """End-to-end preprocessing pipeline for a single currency pair.

    Args:
        pair: Currency pair ticker.
        start_date: Data start date.
        end_date: Data end date.
        lookback: LSTM lookback window.
        forecast_horizon: Number of future days to forecast.
        use_synthetic: If True, use synthetic data instead of downloading.
        output_dir: Directory to save processed CSVs.

    Returns:
        dict: Contains raw_df, feature_df, splits, scaled arrays, scaler.
    """
    logger.info("Starting preprocessing pipeline for %s", pair)
    if use_synthetic:
        raw_df = generate_synthetic_forex_data(pair=pair)
    else:
        data = fetch_forex_data(pairs=[pair], start_date=start_date, end_date=end_date)
        if pair not in data:
            raise ValueError(f"Failed to fetch data for {pair}")
        raw_df = data[pair]

    df = clean_forex_data(raw_df)
    df = add_technical_indicators(df)
    df = add_lag_features(df, target_col="Close")
    df = add_calendar_features(df)
    df = df.dropna()

    train_df, val_df, test_df = train_test_split_time_series(df)
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, f"{pair.replace("=", "")}_train.csv"))
    val_df.to_csv(os.path.join(output_dir, f"{pair.replace("=", "")}_val.csv"))
    test_df.to_csv(os.path.join(output_dir, f"{pair.replace("=", "")}_test.csv"))

    # Scale Close price for LSTM
    train_close = train_df["Close"].values
    val_close = val_df["Close"].values
    test_close = test_df["Close"].values
    train_s, val_s, test_s, scaler = scale_data(train_close, val_close, test_close)

    # Create supervised windows
    X_train, y_train = create_supervised_dataset(train_s, lookback, forecast_horizon)
    X_val, y_val = create_supervised_dataset(val_s, lookback, forecast_horizon)
    X_test, y_test = create_supervised_dataset(test_s, lookback, forecast_horizon)

    logger.info("Pipeline complete: X_train=%s, X_val=%s, X_test=%s",
                X_train.shape, X_val.shape, X_test.shape)

    return {
        "raw_df": raw_df,
        "feature_df": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val,   "y_val": y_val,
        "X_test": X_test,  "y_test": y_test,
        "scaler": scaler,
        "lookback": lookback,
        "forecast_horizon": forecast_horizon,
    }


if __name__ == "__main__":
    result = run_preprocessing_pipeline(use_synthetic=True)
    logger.info("Sample feature_df shape: %s", result["feature_df"].shape)
    logger.info("Sample feature_df columns: %s", result["feature_df"].columns.tolist())
    logger.info("X_train shape: %s", result["X_train"].shape)
    logger.info("y_train shape: %s", result["y_train"].shape)
