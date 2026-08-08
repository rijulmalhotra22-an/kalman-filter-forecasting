"""
pipeline.py — Data ingestion, synthetic data generation, and cointegration testing.

Generates realistic cointegrated time-series pairs and validates
cointegration using Engle-Granger and Johansen tests (statsmodels).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

logger = logging.getLogger(__name__)


@dataclass
class CointTestResult:
    """Results from cointegration tests."""

    engle_granger_pvalue: float
    engle_granger_stat: float
    johansen_trace_stat: float
    johansen_critical_5pct: float
    is_cointegrated: bool


def generate_cointegrated_pair(
    n_days: int = 1260,
    beta_true: float = 0.8,
    alpha_true: float = 5.0,
    spread_mean: float = 0.0,
    spread_mr_speed: float = 0.035,
    spread_vol: float = 0.8,
    price_vol: float = 0.15,
    price_drift: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate two cointegrated price series with realistic properties.

    The independent series follows a geometric random walk.
    The dependent series = alpha + beta * independent + mean-reverting spread.

    Parameters
    ----------
    n_days : int
        Number of trading days (~5 years = 1260).
    beta_true : float
        True hedge ratio.
    alpha_true : float
        True intercept.
    spread_mean : float
        Long-run spread mean.
    spread_mr_speed : float
        Mean-reversion speed (higher = faster reversion).
        0.035 ≈ half-life of ~20 days.
    spread_vol : float
        Spread innovation volatility.
    price_vol : float
        Annual volatility of the independent series.
    price_drift : float
        Annual drift of the independent series.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: date, series_x, series_y, true_spread
    """
    logger.info(
        "Generating synthetic cointegrated pair: n_days=%d, beta=%.2f, "
        "alpha=%.1f, mr_speed=%.4f",
        n_days, beta_true, alpha_true, spread_mr_speed,
    )
    rng = np.random.default_rng(seed)

    # Daily parameters
    daily_drift = price_drift / 252
    daily_vol = price_vol / np.sqrt(252)

    # Independent series: geometric random walk
    log_returns_x = daily_drift + daily_vol * rng.standard_normal(n_days)
    log_price_x = np.cumsum(log_returns_x)
    series_x = 100 * np.exp(log_price_x)  # start around 100

    # Mean-reverting spread (Ornstein-Uhlenbeck)
    spread = np.zeros(n_days)
    daily_spread_vol = spread_vol / np.sqrt(252)
    for t in range(1, n_days):
        spread[t] = (
            spread[t - 1]
            + spread_mr_speed * (spread_mean - spread[t - 1])
            + daily_spread_vol * rng.standard_normal()
        )

    # Dependent series
    series_y = alpha_true + beta_true * series_x + spread

    # Date index
    dates = pd.bdate_range(start="2020-01-02", periods=n_days, freq="B")

    df = pd.DataFrame({
        "date": dates,
        "series_x": series_x,
        "series_y": series_y,
        "true_spread": spread,
    })

    logger.info(
        "Generated pair: X range [%.1f, %.1f], Y range [%.1f, %.1f]",
        series_x.min(), series_x.max(), series_y.min(), series_y.max(),
    )
    return df


def run_cointegration_tests(
    series_y: np.ndarray | pd.Series,
    series_x: np.ndarray | pd.Series,
    significance: float = 0.05,
) -> CointTestResult:
    """Run Engle-Granger and Johansen cointegration tests.

    Parameters
    ----------
    series_y, series_x : array-like
        The two price series to test.
    significance : float
        p-value threshold.  Default 0.05.

    Returns
    -------
    CointTestResult

    Raises
    ------
    ValueError
        If Engle-Granger p-value >= significance.
    """
    y = np.asarray(series_y, dtype=np.float64)
    x = np.asarray(series_x, dtype=np.float64)

    # --- Engle-Granger ---
    logger.info("Running Engle-Granger cointegration test …")
    eg_stat, eg_pvalue, _ = coint(y, x)
    logger.info("  EG statistic=%.4f, p-value=%.6f", eg_stat, eg_pvalue)

    # --- Johansen ---
    logger.info("Running Johansen cointegration test …")
    data_matrix = np.column_stack([y, x])
    joh = coint_johansen(data_matrix, det_order=0, k_ar_diff=1)
    # Trace test: first eigenvalue row, 5 % critical value is column index 1
    trace_stat = joh.lr1[0]
    crit_5pct = joh.cvt[0, 1]
    logger.info(
        "  Johansen trace stat=%.4f, 5%% critical=%.4f",
        trace_stat, crit_5pct,
    )

    is_coint = eg_pvalue < significance
    result = CointTestResult(
        engle_granger_pvalue=float(eg_pvalue),
        engle_granger_stat=float(eg_stat),
        johansen_trace_stat=float(trace_stat),
        johansen_critical_5pct=float(crit_5pct),
        is_cointegrated=is_coint,
    )

    if not is_coint:
        msg = (
            f"Cointegration test FAILED: Engle-Granger p-value = {eg_pvalue:.6f} "
            f"(threshold {significance}).  Cannot proceed."
        )
        logger.error(msg)
        raise ValueError(msg)

    logger.info("✓ Cointegration confirmed (p=%.6f < %.2f)", eg_pvalue, significance)
    return result


def clean_and_align(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for modelling.

    Steps: forward-fill NaNs, drop remaining NaNs, sort by date, reset index.
    """
    logger.info("Cleaning data: %d rows, %d cols", len(df), len(df.columns))
    df = df.sort_values("date").reset_index(drop=True)
    df = df.ffill().dropna().reset_index(drop=True)
    logger.info("After cleaning: %d rows", len(df))
    return df


def load_data(
    seed: int = 42,
    n_days: int = 1260,
    significance: float = 0.05,
    **kwargs,
) -> tuple[pd.DataFrame, CointTestResult]:
    """Full pipeline: generate → clean → test cointegration.

    Returns
    -------
    df : pd.DataFrame
        Cleaned dataframe with columns [date, series_x, series_y, true_spread].
    coint_result : CointTestResult
        Cointegration test results.
    """
    logger.info("=" * 60)
    logger.info("DATA PIPELINE START")
    logger.info("=" * 60)

    df = generate_cointegrated_pair(seed=seed, n_days=n_days, **kwargs)
    df = clean_and_align(df)
    coint_result = run_cointegration_tests(
        df["series_y"], df["series_x"], significance=significance
    )

    logger.info("DATA PIPELINE COMPLETE")
    return df, coint_result
