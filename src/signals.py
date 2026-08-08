"""
signals.py — Z-score computation and trading signal generation.

Consumes Kalman filter output and produces daily signals:
LONG / SHORT / FLAT based on z-score thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_zscore(
    spread: np.ndarray,
    spread_var: np.ndarray | None = None,
    window: int = 30,
) -> np.ndarray:
    """Compute the z-score of the spread.

    Parameters
    ----------
    spread : np.ndarray
        Kalman prediction error or spread residual at each timestep.
    spread_var : np.ndarray, optional
        Innovation variance (S) at each timestep from the Kalman filter.
    window : int, default 30
        Rolling window size for standard deviation normalization.
        If window > 0, rolling std is used.

    Returns
    -------
    np.ndarray
        Z-score at each timestep.
    """
    if spread_var is not None:
        spread_std = np.sqrt(np.maximum(spread_var, 1e-10))
        return spread / spread_std
    elif window is not None and window > 0:
        s_series = pd.Series(spread)
        min_p = min(len(spread), max(2, window // 3))
        roll_mean = s_series.rolling(window=window, min_periods=min_p).mean().bfill().fillna(0).values
        roll_std = s_series.rolling(window=window, min_periods=min_p).std().bfill().fillna(1.0).values
        spread_std = np.maximum(roll_std, 1e-10)
        return (spread - roll_mean) / spread_std
    else:
        std = np.std(spread)
        return spread / max(std, 1e-10)


def generate_signals(
    zscore: np.ndarray,
    entry_threshold: float = 1.5,
    exit_threshold: float = 0.5,
) -> np.ndarray:
    """Generate trading signals from z-scores.

    Signal rules (with hysteresis):
    - Enter LONG  (1)  when z-score < -entry_threshold
    - Enter SHORT (-1) when z-score > +entry_threshold
    - Exit to FLAT (0) when |z-score| < exit_threshold
    - Hold current position between entry and exit thresholds

    Parameters
    ----------
    zscore : np.ndarray
        Z-score series.
    entry_threshold : float
        Absolute z-score to enter a position (default 1.5).
    exit_threshold : float
        Absolute z-score to exit a position (default 0.5).

    Returns
    -------
    np.ndarray
        Signal array: 1 = long, -1 = short, 0 = flat.
    """
    T = len(zscore)
    signals = np.zeros(T, dtype=np.int32)
    position = 0

    for t in range(T):
        z = zscore[t]
        if position == 0:
            # No position — check for entry
            if z < -entry_threshold:
                position = 1   # spread too low → long spread
            elif z > entry_threshold:
                position = -1  # spread too high → short spread
        elif position == 1:
            # Long position — check for exit
            if abs(z) < exit_threshold:
                position = 0
            elif z > entry_threshold:
                position = -1  # flip to short
        elif position == -1:
            # Short position — check for exit
            if abs(z) < exit_threshold:
                position = 0
            elif z < -entry_threshold:
                position = 1  # flip to long

        signals[t] = position

    return signals


def build_signal_dataframe(
    dates: pd.Series | np.ndarray,
    spread: np.ndarray,
    spread_var: np.ndarray | None = None,
    entry_threshold: float = 1.5,
    exit_threshold: float = 0.5,
    window: int = 30,
) -> pd.DataFrame:
    """Full signal pipeline: z-score → signals → DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: date, spread, zscore, signal, position_label
    """
    zscore = compute_zscore(spread, spread_var=spread_var, window=window)
    signals = generate_signals(zscore, entry_threshold, exit_threshold)

    labels = {1: "LONG", -1: "SHORT", 0: "FLAT"}
    position_labels = [labels[s] for s in signals]

    return pd.DataFrame({
        "date": np.asarray(dates),
        "spread": spread,
        "zscore": zscore,
        "signal": signals,
        "position_label": position_labels,
    })
