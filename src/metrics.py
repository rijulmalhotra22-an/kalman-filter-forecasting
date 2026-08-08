"""
metrics.py — Performance metrics for backtesting evaluation.

Computes: CAGR, Sharpe ratio, max drawdown, win rate, profit factor.
All functions operate on pandas Series / DataFrames and return scalars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cagr(equity_curve: pd.Series) -> float:
    """Compound Annual Growth Rate.

    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative equity indexed by date (or integer index with 252
        trading-days-per-year assumption).

    Returns
    -------
    float  CAGR as a decimal (e.g. 0.12 = 12 %).
    """
    if len(equity_curve) < 2:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    if total_return <= 0:
        return -1.0
    n_days = len(equity_curve)
    return float(total_return ** (252 / n_days) - 1)


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    daily_returns : pd.Series
        Simple daily returns.
    risk_free_rate : float
        Annual risk-free rate (default 0).

    Returns
    -------
    float  Annualised Sharpe ratio.
    """
    active_returns = daily_returns[daily_returns != 0]
    if len(active_returns) < 2 or active_returns.std() == 0:
        return 0.0
    daily_rf = risk_free_rate / 252
    excess = active_returns - daily_rf
    return float(excess.mean() / excess.std() * np.sqrt(252))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown as a positive fraction (e.g. 0.25 = 25 %).

    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative equity.

    Returns
    -------
    float  Max drawdown (0–1 range).
    """
    if len(equity_curve) < 2:
        return 0.0
    cummax = equity_curve.cummax()
    drawdowns = (cummax - equity_curve) / cummax
    return float(drawdowns.max())


def win_rate(trade_pnls: pd.Series) -> float:
    """Fraction of trades that are profitable.

    Parameters
    ----------
    trade_pnls : pd.Series
        PnL per closed trade.

    Returns
    -------
    float  Win rate as a decimal (e.g. 0.65 = 65 %).
    """
    if len(trade_pnls) == 0:
        return 0.0
    return float((trade_pnls > 0).sum() / len(trade_pnls))


def profit_factor(trade_pnls: pd.Series) -> float:
    """Profit factor = gross profits / gross losses.

    Parameters
    ----------
    trade_pnls : pd.Series
        PnL per closed trade.

    Returns
    -------
    float  Profit factor (>1 is profitable). Returns inf if no losses.
    """
    gross_profit = trade_pnls[trade_pnls > 0].sum()
    gross_loss = abs(trade_pnls[trade_pnls < 0].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def compute_all_metrics(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
    trade_pnls: pd.Series,
) -> dict[str, float]:
    """Convenience wrapper — returns a dict of all five metrics."""
    return {
        "CAGR": cagr(equity_curve),
        "Sharpe Ratio": sharpe_ratio(daily_returns),
        "Max Drawdown": max_drawdown(equity_curve),
        "Win Rate": win_rate(trade_pnls),
        "Profit Factor": profit_factor(trade_pnls),
    }
