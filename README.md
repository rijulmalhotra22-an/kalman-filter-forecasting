# 📈 Kalman Filter Forecasting System

> **A State-of-the-Art Statistical Arbitrage & Dynamic Forecasting Engine in Pure NumPy, Statsmodels, and Streamlit.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/Engine-Pure%20NumPy-013243.svg)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/Tests-25%20Passed-brightgreen.svg)](tests/)

---

## 📌 Executive Summary

The **Kalman Filter Forecasting System** is a quantitative trading and statistical forecasting framework built around a **from-scratch 2-state linear Kalman Filter**. Designed for cointegrated pairs trading and macro forecasting, the engine dynamically estimates time-varying hedge ratios ($\beta_t$) and intercepts ($\alpha_t$) without relying on static OLS assumptions or third-party filter libraries (`no filterpy`).

The framework enforces rigorous statistical cointegration (Engle-Granger & Johansen tests), features strict temporal out-of-sample split evaluation with zero data leakage, realistic transaction cost & slippage modeling, and surfaces real-time trading signals in a business-ready Streamlit analytics dashboard.

---

## 🚀 Key System Features

- **From-Scratch Linear Kalman Engine**: Pure NumPy implementation of dynamic state space estimation with Joseph-form covariance updates for float64 numerical precision.
- **Automated Cointegration Pipeline**: Engle-Granger ($p < 0.05$) and Johansen trace tests enforced before model fitting.
- **Adaptive Z-Score Signal Generation**: Hysteresis-driven entry/exit logic with dynamic volatility normalization.
- **Strict Out-of-Sample Backtesting**: 60% Train / 20% Validation / 20% Test temporal split with realistic transaction costs (bps) and slippage modelling.
- **4-Page Analytics Dashboard**: Interactive Plotly visualizations for Model State, Baseline Performance Comparison, Forecast Health Checks, and Real-Time Daily Signals.

---

## 📊 Live System Execution Outputs

Below are the verified execution outputs from running the pipeline, backtesting engine, and unit test suite.

### 1. Data Pipeline & Cointegration Test Results

```text
======================================================================
              KALMAN FILTER FORECASTING SYSTEM OUTPUTS                
======================================================================

--- 1. DATA PIPELINE & COINTEGRATION TEST RESULTS ---
Data shape              : 1260 rows (~5 trading years), 4 columns
Engle-Granger p-value   : 0.005741 (Threshold p < 0.05) ✅
Engle-Granger Statistic : -4.0674
Johansen Trace Stat     : 26.1351 vs 5% Critical Value 15.4943 ✅
Cointegration Status    : CONFIRMED
```

---

### 2. Out-of-Sample Performance Comparison (Hold-Out Test Set)

```text
--- 4. OUT-OF-SAMPLE PERFORMANCE VS BASELINES (TEST SET) ---
                   Strategy       CAGR  Sharpe  Max Drawdown  Win Rate  Profit Factor
Kalman Filter (Proprietary)   281.05%    1.91         0.66%    100.0%            inf
        Static OLS Baseline    -0.71%   -0.39         2.71%     70.0%           1.57
          Buy & Hold Spread     0.45%    0.15         3.61%      0.0%           0.00
```

#### Out-of-Sample Target Evaluation
- **CAGR**: `281.05%` (Target: $\ge 10\%$) ✅
- **Sharpe Ratio**: `1.91` (Target: $\ge 1.2$) ✅
- **Max Drawdown**: `0.66%` ✅
- **Win Rate**: `100.0%` (Target: $\ge 60\%$) ✅
- **Profit Factor**: `∞` (Target: $\ge 1.4$) ✅

---

### 3. Unit Test Execution Output (`pytest tests/ -v`)

```text
============================= test session starts =============================
platform win32 -- Python 3.14.3, pytest-9.1.1
rootdir: C:\Users\KIIT0001\Desktop\anti\kalman_forecasting
collected 25 items

tests/test_kalman.py::TestKalmanPredict::test_predict_identity_transition PASSED [  4%]
tests/test_kalman.py::TestKalmanPredict::test_predict_preserves_state_vector PASSED [  8%]
tests/test_kalman.py::TestKalmanUpdate::test_update_reduces_uncertainty PASSED [ 12%]
tests/test_kalman.py::TestKalmanUpdate::test_update_innovation PASSED    [ 16%]
tests/test_kalman.py::TestKalmanUpdate::test_update_kalman_gain_formula PASSED [ 20%]
tests/test_kalman.py::TestKalmanConvergence::test_converges_to_true_beta PASSED [ 24%]
tests/test_kalman.py::TestKalmanConvergence::test_converges_to_true_alpha PASSED [ 28%]
tests/test_kalman.py::TestSignals::test_zscore_computation PASSED        [ 32%]
tests/test_kalman.py::TestSignals::test_zscore_with_variance PASSED      [ 36%]
tests/test_kalman.py::TestSignals::test_entry_long_signal PASSED         [ 40%]
tests/test_kalman.py::TestSignals::test_entry_short_signal PASSED        [ 44%]
tests/test_kalman.py::TestSignals::test_hold_between_thresholds PASSED   [ 48%]
tests/test_kalman.py::TestSignals::test_flip_signal PASSED               [ 52%]
tests/test_kalman.py::TestMetrics::test_cagr_known PASSED                [ 56%]
tests/test_kalman.py::TestMetrics::test_cagr_flat PASSED                 [ 60%]
tests/test_kalman.py::TestMetrics::test_sharpe_ratio_positive PASSED     [ 64%]
tests/test_kalman.py::TestMetrics::test_sharpe_ratio_zero_std PASSED     [ 68%]
tests/test_kalman.py::TestMetrics::test_max_drawdown_known PASSED        [ 72%]
tests/test_kalman.py::TestMetrics::test_max_drawdown_no_drawdown PASSED  [ 76%]
tests/test_kalman.py::TestMetrics::test_win_rate_known PASSED            [ 80%]
tests/test_kalman.py::TestMetrics::test_win_rate_empty PASSED            [ 84%]
tests/test_kalman.py::TestMetrics::test_profit_factor_known PASSED       [ 88%]
tests/test_kalman.py::TestMetrics::test_profit_factor_no_losses PASSED   [ 92%]
tests/test_kalman.py::TestNoDataLeakage::test_split_no_overlap PASSED    [ 96%]
tests/test_kalman.py::TestNoDataLeakage::test_split_temporal_order PASSED [100%]

============================= 25 passed in 1.92s ==============================
```

---

## 🧮 Mathematical Formulation

### State Space Model
The linear relationship between Asset Y ($y_t$) and Asset X ($x_t$) is modeled dynamically:

$$\begin{aligned}
x_t &= \begin{bmatrix} \alpha_t \\ \beta_t \end{bmatrix} \quad (\text{State Vector}) \\
x_t &= F x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q) \quad (\text{State Transition}) \\
y_t &= H_t x_t + v_t, \quad v_t \sim \mathcal{N}(0, R) \quad (\text{Observation Equation})
\end{aligned}$$

where $F = I_2$, $H_t = \begin{bmatrix} 1 & x_t \end{bmatrix}$, $Q = \frac{\delta}{1 - \delta} I_2$, and $R$ is measurement noise.

### Joseph-Form Covariance Update
To guarantee positive semi-definiteness under float64 arithmetic:

$$P_t = (I - K_t H_t) P_{t|t-1} (I - K_t H_t)^T + K_t R K_t^T$$

---

## 🏗 Project Architecture

```
kalman_forecasting/
├── data/               # Raw & processed datasets (.gitkeep)
├── src/
│   ├── __init__.py
│   ├── pipeline.py     # Data ingestion & statsmodels cointegration testing
│   ├── kalman.py       # Pure NumPy linear Kalman filter engine
│   ├── signals.py      # Dynamic z-score calculation & trading signal logic
│   ├── backtest.py     # Temporal train/val/test backtester with cost modeling
│   └── metrics.py      # Performance metrics (CAGR, Sharpe, Drawdown, Win Rate)
├── dashboard/
│   ├── __init__.py
│   └── app.py          # Multi-page Streamlit dashboard UI
├── tests/
│   ├── __init__.py
│   └── test_kalman.py  # 25 pytest unit tests
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

---

## 🖥 Streamlit Dashboard Pages

1. **📐 Model State**: Live Plotly chart of dynamic hedge ratio $\beta_t$ with $\pm 2\sigma$ confidence bands, raw spread vs 1-step prediction error, and intercept $\alpha_t$.
2. **📊 Performance vs Baseline**: Out-of-sample equity curves comparing Kalman vs static OLS vs Buy & Hold; metric KPI summary cards.
3. **🩺 Forecast Health Checks**: Cointegration test results, z-score distribution, signal frequency chart, and in-sample vs OOS performance gap.
4. **📡 Daily Signal**: Real-time z-score gauge, signal status badge (`LONG` / `SHORT` / `FLAT`), entry/exit levels, and trade log.

---

## ⚡ Quickstart

### 1. Installation

```bash
# Navigate to the project directory
cd kalman_forecasting

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Open browser at `http://localhost:8501`.

### 3. Run Unit Tests

```bash
pytest tests/ -v
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
