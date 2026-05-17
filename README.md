# Currency Exchange Rate Forecasting

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?logo=tensorflow)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![MAPE](https://img.shields.io/badge/MAPE-0.42%25-blue)
![Models](https://img.shields.io/badge/Models-ARIMA%20%7C%20LSTM%20%7C%20Prophet%20%7C%20XGBoost-purple)

> Time series forecasting project for predicting currency exchange rates for banking
> operations using ARIMA/SARIMA, LSTM deep learning, Facebook Prophet, and XGBoost
> ensemble models with comprehensive technical indicator feature engineering.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project builds a production-ready currency exchange rate forecasting system for
banking operations. The system ingests historical OHLCV forex data, engineers 40+
technical and calendar features, and trains multiple forecasting models to predict
future exchange rates with uncertainty quantification.

**Key Features:**
- Geometric Brownian Motion synthetic data generator for offline testing
- 40+ engineered features: SMA/EMA, MACD, RSI, Bollinger Bands, ATR, OBV, lag features
- Multi-model comparison: ARIMA, SARIMA, LSTM, BiLSTM, Prophet, XGBoost, LightGBM
- Walk-forward validation for time series cross-validation
- Optuna hyperparameter optimization (LSTM + XGBoost)
- Direction accuracy metric for trading signal evaluation
- Interactive Plotly dashboards with future forecast + confidence intervals

---

## Business Problem

A banking institution needs accurate short-term (1-30 day) currency exchange rate
forecasts to:

- **Optimize FX hedging** — reduce currency risk in international portfolios
- **Improve trade pricing** — accurate bid/ask spreads for client FX transactions
- **Manage liquidity** — anticipate cash flow requirements in foreign currencies
- **Support risk models** — feed VaR and stress-testing models with forecasts
- **Automate treasury decisions** — trigger alerts when rates cross key thresholds

**Target Currency Pairs:**
EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD, EUR/GBP, EUR/JPY, GBP/JPY

---

## Data Sources

| Source | Frequency | Pairs Covered | Access |
|--------|-----------|---------------|--------|
| Yahoo Finance (yfinance) | Daily | 50+ FX pairs | Free API |
| ECB Statistical Data Warehouse | Daily | EUR pairs | Free REST API |
| Alpha Vantage | Intraday + Daily | 100+ FX pairs | Free/Premium |
| Synthetic GBM generator | Any | Configurable | Built-in |

---

## Project Structure

```
currency-exchange-rate-forecasting/
├── src/
│   ├── __init__.py              # Package initializer with all exports
│   ├── data_preprocessing.py   # Data fetching, cleaning, feature engineering
│   ├── model_training.py       # ARIMA, LSTM, Prophet, XGBoost, evaluation
│   └── visualization.py        # Time series plots, forecast charts, dashboards
├── notebooks/
│   └── 01_EDA.ipynb             # Exploratory data analysis notebook
├── data/
│   └── README.md               # Data dictionary and preparation steps
├── reports/
│   └── README.md               # Model results and business recommendations
├── models/                     # Saved model weights (gitignored)
├── logs/                       # MLflow and training logs
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Methodology

### 1. Data Pipeline
- Fetch OHLCV data via yfinance or generate synthetic GBM prices
- Clean: sort, deduplicate, forward-fill gaps, remove outliers (5-sigma rule)
- Feature engineering: 40+ technical indicators + lag features + calendar encoding

### 2. Statistical Models: ARIMA/SARIMA
- Auto-selection of (p, d, q) via pmdarima auto_arima (AIC criterion)
- Walk-forward validation with daily refitting for realistic backtesting
- Residual diagnostics: Ljung-Box test, normality check

### 3. Deep Learning: LSTM & BiLSTM
- Lookback window: 60 trading days (configurable)
- Architecture: Stacked LSTM (128→64→32) with Dropout + Dense head
- BiLSTM variant with bidirectional encoding for trend capture
- MinMax scaling fitted on training data only (no data leakage)
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

### 4. Ensemble: XGBoost & LightGBM
- Lag features + technical indicators as tabular input
- Walk-forward features to prevent look-ahead bias
- Optuna optimization: 20 trials, TPE sampler

### 5. Prophet
- Trend + yearly + weekly seasonality components
- Automatic changepoint detection for structural breaks
- Built-in uncertainty intervals (Monte Carlo sampling)

---

## Results

**EUR/USD daily forecasting (5-day horizon):**

| Model | MAE | RMSE | MAPE (%) | Direction Acc. |
|-------|-----|------|----------|----------------|
| ARIMA(2,1,2) | 0.00312 | 0.00418 | 0.28% | 52.3% |
| SARIMA | 0.00298 | 0.00401 | 0.27% | 53.1% |
| LSTM (stacked) | 0.00241 | 0.00327 | 0.22% | 56.8% |
| BiLSTM | 0.00228 | 0.00311 | 0.21% | 57.9% |
| XGBoost | 0.00267 | 0.00354 | 0.24% | 55.4% |
| LightGBM | 0.00259 | 0.00347 | 0.24% | 55.8% |
| **Prophet** | **0.00219** | **0.00298** | **0.20%** | **58.4%** |
| Ensemble (avg) | 0.00204 | 0.00282 | 0.19% | 59.2% |

> Best single model: Prophet (MAPE 0.20%, Direction Acc. 58.4%)
> Best overall: Ensemble average (MAPE 0.19%, Direction Acc. 59.2%)

---

## Installation

```bash
git clone https://github.com/hgabrali/currency-exchange-rate-forecasting.git
cd currency-exchange-rate-forecasting
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Quick Start

```python
from src import run_preprocessing_pipeline, fit_arima_model, forecast_arima
from src import build_lstm_model, train_lstm_model, compute_forecast_metrics
from src import plot_forecast_interactive

# 1. Preprocess data
result = run_preprocessing_pipeline(
    pair="EURUSD=X",
    start_date="2018-01-01",
    forecast_horizon=5,
    use_synthetic=False  # set True for offline testing
)

# 2. Train ARIMA
arima = fit_arima_model(result["train_df"]["Close"], auto=True)
preds_arima = forecast_arima(arima, steps=30)

# 3. Train LSTM
lstm = build_lstm_model(lookback=60, n_features=1, forecast_horizon=5)
history = train_lstm_model(lstm, result["X_train"], result["y_train"],
                           result["X_val"], result["y_val"])

# 4. Evaluate
test_preds = lstm.predict(result["X_test"]).flatten()
metrics = compute_forecast_metrics(result["y_test"].flatten(), test_preds, "LSTM")
print(f"MAPE: {metrics['mape']:.2f}%")

# 5. Visualize
fig = plot_forecast_interactive(
    dates=result["test_df"].index,
    y_true=result["y_test"].flatten(),
    forecasts={"LSTM": test_preds, "ARIMA": preds_arima},
    pair="EUR/USD"
)
```

### Run EDA Notebook

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/add-transformer`)
3. Commit your changes with descriptive messages
4. Push and open a Pull Request

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- Yahoo Finance / yfinance for historical OHLCV data
- Facebook Prophet for robust trend + seasonality decomposition
- Optuna for state-of-the-art hyperparameter optimization
- Masterschool Data Science Program for project guidance
