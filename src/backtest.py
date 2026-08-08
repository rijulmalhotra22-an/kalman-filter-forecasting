"""
backtest.py — Backtesting & evaluation engine with cost modelling.

Implements:
- Strict train / validation / test split (no data leakage)
- Kalman strategy vs OLS baseline comparison
- Transaction cost and slippage modelling
- Equity curve computation with per-trade PnL tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from .kalman import KalmanFilter
from .signals import compute_zscore, generate_signals
from .metrics import compute_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class SplitData:
    """Train / validation / test split."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_frac: float
    val_frac: float
    test_frac: float


@dataclass
class BacktestResult:
    """Full backtest output for one strategy."""

    name: str
    equity_curve: pd.Series
    daily_returns: pd.Series
    trade_pnls: pd.Series
    signals_df: pd.DataFrame
    metrics: dict[str, float]


def split_data(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> SplitData:
    """Temporal train / validation / test split.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset, assumed sorted by date.
    train_frac : float
        Fraction for training (default 0.60).
    val_frac : float
        Fraction for validation (default 0.20).
        Test = 1 - train - val.

    Returns
    -------
    SplitData
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    split = SplitData(
        train=df.iloc[:train_end].copy().reset_index(drop=True),
        validation=df.iloc[train_end:val_end].copy().reset_index(drop=True),
        test=df.iloc[val_end:].copy().reset_index(drop=True),
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=round(1.0 - train_frac - val_frac, 4),
    )
    logger.info(
        "Data split: train=%d, val=%d, test=%d",
        len(split.train), len(split.validation), len(split.test),
    )
    return split


def compute_equity_curve(
    spread: np.ndarray,
    signals: np.ndarray,
    txn_cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    capital_base: float = 100.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute equity curve from spread returns and signals.

    Parameters
    ----------
    spread : np.ndarray
        Spread series (Kalman prediction error or OLS residual).
    signals : np.ndarray
        Position signals (1, -1, 0).
    txn_cost_bps : float
        One-way transaction cost in basis points.
    slippage_bps : float
        Slippage in basis points per trade.
    capital_base : float
        Capital allocated to position for percentage return scaling.

    Returns
    -------
    equity_curve : pd.Series
    daily_returns : pd.Series
    trade_pnls : pd.Series
    """
    T = len(spread)
    total_cost_frac = (txn_cost_bps + slippage_bps) / 10_000

    daily_pnl = np.zeros(T)
    for t in range(1, T):
        raw_pnl = signals[t - 1] * (spread[t] - spread[t - 1])
        # Charge cost on position changes
        cost = abs(signals[t] - signals[t - 1]) * total_cost_frac * abs(spread[t])
        daily_pnl[t] = raw_pnl - cost

    cap = max(capital_base, 1.0)
    daily_returns = pd.Series(daily_pnl / cap, name="returns")
    equity = pd.Series((1.0 + daily_returns).cumprod() * 100.0, name="equity")

    # Extract per-trade PnLs
    trade_pnls = _extract_trade_pnls(spread, signals, total_cost_frac)

    return equity, daily_returns, trade_pnls


def _extract_trade_pnls(
    spread: np.ndarray,
    signals: np.ndarray,
    cost_frac: float,
) -> pd.Series:
    """Extract PnL for each completed round-trip trade."""
    trades: list[float] = []
    in_trade = False
    entry_price = 0.0
    entry_signal = 0
    total_cost = 0.0

    for t in range(len(signals)):
        if not in_trade and signals[t] != 0:
            # Enter trade
            in_trade = True
            entry_price = spread[t]
            entry_signal = signals[t]
            total_cost = cost_frac * abs(spread[t])
        elif in_trade:
            if signals[t] != entry_signal:
                # Exit (or flip)
                exit_price = spread[t]
                pnl = entry_signal * (exit_price - entry_price)
                total_cost += cost_frac * abs(spread[t])
                trades.append(pnl - total_cost)

                if signals[t] != 0:
                    # Flip — enter new trade
                    entry_price = spread[t]
                    entry_signal = signals[t]
                    total_cost = cost_frac * abs(spread[t])
                else:
                    in_trade = False
                    total_cost = 0.0

    return pd.Series(trades, name="trade_pnl", dtype=np.float64)


# ──────────────────────────────────────────────────────────────────
# Kalman Strategy
# ──────────────────────────────────────────────────────────────────

def run_kalman_backtest(
    df: pd.DataFrame,
    delta: float = 1e-4,
    R: float = 1e-3,
    entry_threshold: float = 1.5,
    exit_threshold: float = 0.5,
    txn_cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> tuple[BacktestResult, dict]:
    """Run the Kalman strategy on a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns [date, series_x, series_y].

    Returns
    -------
    result : BacktestResult
    kalman_output : dict
        Raw Kalman filter output (alpha, beta, spread, etc.).
    """
    kf = KalmanFilter(delta=delta, R=R)
    kf_out = kf.filter_series(df["series_y"].values, df["series_x"].values)

    zscore = compute_zscore(kf_out["spread"], window=30)
    signals = generate_signals(zscore, entry_threshold, exit_threshold)

    # Use the 1-step prediction residual spread for PnL
    spread = kf_out["spread"]

    # Compute capital base allocated to pair position (~20% margin requirement)
    capital_base = float(np.mean(df["series_y"].values + np.abs(kf_out["beta"]) * df["series_x"].values)) / 5.0

    equity, daily_ret, trade_pnls = compute_equity_curve(
        spread, signals, txn_cost_bps, slippage_bps, capital_base=capital_base,
    )

    metrics = compute_all_metrics(equity, daily_ret, trade_pnls)

    signals_df = pd.DataFrame({
        "date": df["date"].values,
        "spread": kf_out["spread"],
        "raw_spread": spread,
        "zscore": zscore,
        "signal": signals,
    })

    return BacktestResult(
        name="Kalman",
        equity_curve=equity,
        daily_returns=daily_ret,
        trade_pnls=trade_pnls,
        signals_df=signals_df,
        metrics=metrics,
    ), kf_out


# ──────────────────────────────────────────────────────────────────
# OLS Baseline
# ──────────────────────────────────────────────────────────────────

def run_ols_baseline(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    entry_threshold: float = 1.5,
    exit_threshold: float = 0.5,
    txn_cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> BacktestResult:
    """Run the static OLS baseline strategy.

    Fits OLS on train set, applies static hedge ratio to eval set.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data for OLS fit.
    df_eval : pd.DataFrame
        Evaluation data to generate signals on.
    """
    # Fit OLS on training data
    X_train = add_constant(df_train["series_x"].values)
    model = OLS(df_train["series_y"].values, X_train).fit()
    alpha_ols, beta_ols = model.params[0], model.params[1]

    logger.info("OLS baseline: alpha=%.4f, beta=%.4f", alpha_ols, beta_ols)

    # Compute static spread on eval data
    spread = df_eval["series_y"].values - (alpha_ols + beta_ols * df_eval["series_x"].values)

    # Z-score using rolling window (30-day)
    spread_series = pd.Series(spread)
    roll_mean = spread_series.rolling(window=30, min_periods=10).mean().bfill().values
    roll_std = spread_series.rolling(window=30, min_periods=10).std().bfill().values
    roll_std = np.maximum(roll_std, 1e-10)
    zscore = (spread - roll_mean) / roll_std

    signals = generate_signals(zscore, entry_threshold, exit_threshold)

    capital_base = float(np.mean(df_eval["series_y"].values + abs(beta_ols) * df_eval["series_x"].values)) / 5.0

    equity, daily_ret, trade_pnls = compute_equity_curve(
        spread, signals, txn_cost_bps, slippage_bps, capital_base=capital_base,
    )

    metrics = compute_all_metrics(equity, daily_ret, trade_pnls)

    signals_df = pd.DataFrame({
        "date": df_eval["date"].values,
        "spread": spread,
        "raw_spread": spread,
        "zscore": zscore,
        "signal": signals,
    })

    return BacktestResult(
        name="OLS Baseline",
        equity_curve=equity,
        daily_returns=daily_ret,
        trade_pnls=trade_pnls,
        signals_df=signals_df,
        metrics=metrics,
    )


# ──────────────────────────────────────────────────────────────────
# Buy-and-hold baseline
# ──────────────────────────────────────────────────────────────────

def run_buy_and_hold(df: pd.DataFrame) -> BacktestResult:
    """Simple buy-and-hold on the spread for comparison."""
    spread = df["series_y"].values - df["series_x"].values
    equity = pd.Series(spread - spread[0] + 100.0, name="equity")
    daily_ret = pd.Series(np.diff(spread, prepend=spread[0]) / 100.0, name="returns")

    metrics = compute_all_metrics(equity, daily_ret, pd.Series(dtype=np.float64))

    signals_df = pd.DataFrame({
        "date": df["date"].values,
        "spread": spread,
        "raw_spread": spread,
        "zscore": np.zeros(len(spread)),
        "signal": np.ones(len(spread), dtype=np.int32),
    })

    return BacktestResult(
        name="Buy & Hold",
        equity_curve=equity,
        daily_returns=daily_ret,
        trade_pnls=pd.Series(dtype=np.float64),
        signals_df=signals_df,
        metrics=metrics,
    )


# ──────────────────────────────────────────────────────────────────
# Full backtest orchestrator
# ──────────────────────────────────────────────────────────────────

def run_full_backtest(
    df: pd.DataFrame,
    delta: float = 1e-4,
    R: float = 1e-3,
    entry_threshold: float = 1.5,
    exit_threshold: float = 0.5,
    txn_cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> dict:
    """Run complete backtest with all strategies and splits.

    Returns
    -------
    dict with keys:
        split         — SplitData
        kalman_train  — BacktestResult (in-sample)
        kalman_val    — BacktestResult (validation)
        kalman_test   — BacktestResult (out-of-sample)
        kalman_full   — BacktestResult (full series for dashboard viz)
        kalman_output — dict (full-series Kalman filter output)
        ols_test      — BacktestResult
        bh_test       — BacktestResult
    """
    split = split_data(df, train_frac, val_frac)

    params = dict(
        delta=delta, R=R,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        txn_cost_bps=txn_cost_bps,
        slippage_bps=slippage_bps,
    )

    # --- Kalman on each split ---
    logger.info("Running Kalman backtest on TRAIN set …")
    kalman_train, _ = run_kalman_backtest(split.train, **params)
    kalman_train.name = "Kalman (Train)"

    logger.info("Running Kalman backtest on VALIDATION set …")
    kalman_val, _ = run_kalman_backtest(split.validation, **params)
    kalman_val.name = "Kalman (Validation)"

    logger.info("Running Kalman backtest on TEST set …")
    kalman_test, _ = run_kalman_backtest(split.test, **params)
    kalman_test.name = "Kalman (Test)"

    # --- Kalman on full series (for dashboard visualisation) ---
    logger.info("Running Kalman on FULL series for visualisation …")
    kalman_full, kalman_output = run_kalman_backtest(df, **params)
    kalman_full.name = "Kalman"

    # --- OLS baseline on test set ---
    logger.info("Running OLS baseline on TEST set …")
    ols_test = run_ols_baseline(
        split.train, split.test,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        txn_cost_bps=txn_cost_bps,
        slippage_bps=slippage_bps,
    )

    # --- Buy-and-hold on test set ---
    logger.info("Running buy-and-hold on TEST set …")
    bh_test = run_buy_and_hold(split.test)

    return {
        "split": split,
        "kalman_train": kalman_train,
        "kalman_val": kalman_val,
        "kalman_test": kalman_test,
        "kalman_full": kalman_full,
        "kalman_output": kalman_output,
        "ols_test": ols_test,
        "bh_test": bh_test,
    }
