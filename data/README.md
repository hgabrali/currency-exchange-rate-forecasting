# Data Directory

This directory contains raw and processed currency exchange rate datasets.

---

## Directory Structure

```
data/
├── raw/                     # Raw OHLCV data from Yahoo Finance or ECB
│   ├── EURUSD_X.csv         # EUR/USD historical data
│   ├── GBPUSD_X.csv         # GBP/USD historical data
│   └── ...                  # Additional currency pairs
├── processed/               # Preprocessed data with features and splits
│   ├── EURUSDX_train.csv    # Training set with engineered features
│   ├── EURUSDX_val.csv      # Validation set
│   ├── EURUSDX_test.csv     # Test set
│   └── ...                  # Additional pairs
└── README.md                # This file
```

---

## Raw Data Schema (OHLCV)

| Column | Type | Description |
|--------|------|-------------|
| Date | DatetimeIndex | Trading date (business days) |
| Open | float | Opening exchange rate |
| High | float | Intraday high rate |
| Low | float | Intraday low rate |
| Close | float | Closing exchange rate |
| Volume | float | Trading volume (proxy) |

---

## Engineered Features (Processed Data)

**Price Features:**

| Feature | Description |
|---------|-------------|
| daily_return | Percentage change from previous close |
| log_return | Log(Close_t / Close_{t-1}) |
| weekly_return | 5-day return |
| monthly_return | 21-day return |

**Moving Averages:**

| Feature | Description |
|---------|-------------|
| sma_5/10/20/50/200 | Simple moving averages |
| ema_12/26 | Exponential moving averages |
| macd / macd_signal / macd_hist | MACD indicator components |

**Momentum Indicators:**

| Feature | Description |
|---------|-------------|
| rsi_14 | Relative Strength Index (14-day) |
| roc_5 / roc_10 | Rate of Change (5 and 10 days) |

**Volatility Indicators:**

| Feature | Description |
|---------|-------------|
| bb_upper / bb_lower | Bollinger Band upper/lower bounds |
| bb_width / bb_position | Band width and price position |
| atr_14 / atr_pct | Average True Range (14-day) |
| volatility_10/20/60 | Rolling standard deviation of returns |

**Volume Indicators:**

| Feature | Description |
|---------|-------------|
| volume_sma_20 | 20-day average volume |
| volume_ratio | Volume vs. moving average |
| obv | On-Balance Volume |

**Lag Features:**

| Feature | Description |
|---------|-------------|
| Close_lag_1/2/3/5/7/10/14/21 | Past closing rates at various lags |

**Calendar Features:**

| Feature | Description |
|---------|-------------|
| day_of_week / month / quarter | Calendar period indicators |
| is_month_end / is_quarter_end | Binary flags |
| month_sin / month_cos / dow_sin / dow_cos | Cyclical encodings |

---

## Data Statistics

**EUR/USD (2015-2024)**

| Metric | Value |
|--------|-------|
| Total observations | ~2,609 trading days |
| Date range | Jan 2015 - Dec 2024 |
| Min close | 1.0346 (Sep 2022) |
| Max close | 1.2349 (Jan 2021) |
| Mean close | 1.1283 |
| Daily return std | 0.53% |
| Annualized volatility | ~8.4% |

---

## Downloading Data

```python
from src import fetch_forex_data
import os

pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
data = fetch_forex_data(
    pairs=pairs,
    start_date="2015-01-01",
    end_date="2024-12-31"
)

os.makedirs("data/raw", exist_ok=True)
for pair, df in data.items():
    filename = pair.replace("=", "") + ".csv"
    df.to_csv(f"data/raw/{filename}")
    print(f"Saved {filename}: {len(df)} rows")
```

---

## Notes

- Raw CSV files are gitignored to keep the repository lightweight.
- Use `generate_synthetic_forex_data()` for offline pipeline testing.
- All processed files contain features fitted only on the training split (no data leakage).
- Missing data from weekends and holidays is forward-filled (max 3 consecutive days).
- Outliers (>5 sigma from 30-day rolling mean) are removed before feature engineering.
