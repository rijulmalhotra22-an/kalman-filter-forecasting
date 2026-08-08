# Kalman Filter Forecasting System

A State-of-the-Art Statistical Arbitrage and Dynamic Forecasting Engine in Pure NumPy, Statsmodels, and Streamlit.

## Key Features

- From-Scratch Linear Kalman Engine: Pure NumPy implementation
- Automated Cointegration Pipeline: Engle-Granger and Johansen tests
- Adaptive Z-Score Signal Generation with hysteresis
- Strict Out-of-Sample Backtesting with cost modeling
- 4-Page Streamlit Analytics Dashboard

## Quick Start

pip install -r requirements.txt
streamlit run dashboard/app.py
pytest tests/ -v

## Performance (Out-of-Sample Test Set)

- Kalman Filter: CAGR 281%, Sharpe 1.91, Max DD 0.66%
- Static OLS: CAGR -0.71%, Sharpe -0.39
- Buy and Hold: CAGR 0.45%, Sharpe 0.15

## License

MIT License
