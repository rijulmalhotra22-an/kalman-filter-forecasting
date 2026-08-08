"""
test_kalman.py — Unit tests for the Kalman filter, signal logic, and metrics.

Tests:
  - Kalman predict step correctness
  - Kalman update step correctness (hand-computed values)
  - Filter convergence on perfectly cointegrated data
  - Signal generation with known z-score sequences
  - Metric computations against known values
  - No data leakage in train/test split
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.kalman import KalmanFilter
from src.signals import compute_zscore, generate_signals
from src.metrics import cagr, sharpe_ratio, max_drawdown, win_rate, profit_factor
from src.backtest import split_data


# ──────────────────────────────────────────────────────────────────
# Kalman Filter Tests
# ──────────────────────────────────────────────────────────────────


class TestKalmanPredict:
    """Verify prediction step: x_pred = F @ x, P_pred = F @ P @ F.T + Q."""

    def test_predict_identity_transition(self):
        kf = KalmanFilter(delta=1e-4, R=1e-3)
        kf.x = np.array([2.0, 0.5])
        kf.P = np.array([[0.1, 0.0], [0.0, 0.2]])

        x_pred, P_pred = kf.predict()

        # With F = I, x_pred should equal x
        np.testing.assert_array_almost_equal(x_pred, kf.x)

        # P_pred = P + Q
        expected_P = kf.P + kf.Q
        np.testing.assert_array_almost_equal(P_pred, expected_P)

    def test_predict_preserves_state_vector(self):
        kf = KalmanFilter(delta=1e-3, R=1e-2)
        kf.x = np.array([10.0, -3.0])
        x_pred, _ = kf.predict()
        np.testing.assert_array_almost_equal(x_pred, np.array([10.0, -3.0]))


class TestKalmanUpdate:
    """Verify update step: Kalman gain, state, and covariance updates."""

    def test_update_reduces_uncertainty(self):
        kf = KalmanFilter(delta=1e-4, R=1.0)
        kf.x = np.array([0.0, 1.0])
        kf.P = np.eye(2) * 10.0

        x_pred, P_pred = kf.predict()

        H = np.array([[1.0, 50.0]])
        z = 50.0  # observation: y = 0 + 1*50

        x_upd, P_upd, y_innov, S = kf.update(z, H, x_pred, P_pred)

        # Covariance should decrease after update
        assert P_upd[0, 0] < P_pred[0, 0]
        assert P_upd[1, 1] < P_pred[1, 1]

    def test_update_innovation(self):
        kf = KalmanFilter(delta=1e-4, R=1.0)
        kf.x = np.array([5.0, 2.0])  # predict y = 5 + 2*x
        kf.P = np.eye(2) * 0.01

        x_pred, P_pred = kf.predict()
        H = np.array([[1.0, 10.0]])
        z_expected = 5.0 + 2.0 * 10.0  # = 25.0
        z_actual = 27.0

        _, _, y_innov, S = kf.update(z_actual, H, x_pred, P_pred)

        # Innovation should be close to 27 - 25 = 2.0
        assert abs(y_innov - 2.0) < 0.01

    def test_update_kalman_gain_formula(self):
        """Hand-verify Kalman gain K = P_pred @ H.T / S."""
        kf = KalmanFilter(delta=0.0, R=2.0)
        kf.x = np.array([0.0, 0.0])
        P = np.array([[4.0, 0.0], [0.0, 1.0]])
        kf.P = P.copy()

        x_pred, P_pred = kf.predict()  # With delta=0, Q=0, so P_pred = P
        H = np.array([[1.0, 3.0]])

        # Manual computation
        S_manual = (H @ P @ H.T + 2.0).item()  # = 4 + 9 + 2 = 15
        K_manual = (P @ H.T) / S_manual  # = [[4],[3]] / 15

        x_upd, P_upd, y_innov, S = kf.update(0.0, H, x_pred, P_pred)

        assert abs(S - S_manual) < 1e-10
        expected_K = np.array([[4.0 / 15], [3.0 / 15]])
        # Verify by checking the update reduces P correctly
        I_KH = np.eye(2) - K_manual @ H
        expected_P_upd = I_KH @ P @ I_KH.T + (K_manual * 2.0) @ K_manual.T
        np.testing.assert_array_almost_equal(P_upd, expected_P_upd, decimal=10)


class TestKalmanConvergence:
    """Filter should converge to true hedge ratio on clean data."""

    def test_converges_to_true_beta(self):
        rng = np.random.default_rng(123)
        T = 1000
        beta_true = 0.75
        alpha_true = 0.0

        x = np.cumsum(rng.standard_normal(T) * 0.5)
        x = x - np.mean(x)  # zero-mean
        noise = rng.standard_normal(T) * 0.1
        y = alpha_true + beta_true * x + noise

        kf = KalmanFilter(delta=1e-4, R=0.01)
        result = kf.filter_series(y, x)

        # After 200 steps, beta should be within 2% of true
        beta_final = result["beta"][200:]
        mean_beta = np.mean(beta_final)
        assert abs(mean_beta - beta_true) / beta_true < 0.02, (
            f"Beta did not converge: mean={mean_beta:.4f}, true={beta_true}"
        )

    def test_converges_to_true_alpha(self):
        rng = np.random.default_rng(456)
        T = 1000
        beta_true = 1.2
        alpha_true = -5.0

        x = np.cumsum(rng.standard_normal(T) * 0.3)
        x = x - np.mean(x)  # zero-mean
        noise = rng.standard_normal(T) * 0.05
        y = alpha_true + beta_true * x + noise

        kf = KalmanFilter(delta=1e-4, R=0.005)
        result = kf.filter_series(y, x)

        alpha_tail = result["alpha"][200:]
        mean_alpha = np.mean(alpha_tail)
        assert abs(mean_alpha - alpha_true) < 0.5, (
            f"Alpha did not converge: mean={mean_alpha:.4f}, true={alpha_true}"
        )


# ──────────────────────────────────────────────────────────────────
# Signal Logic Tests
# ──────────────────────────────────────────────────────────────────


class TestSignals:
    """Test z-score computation and signal generation."""

    def test_zscore_computation(self):
        spread = np.array([0.0, 1.0, -1.0, 2.0])
        spread_var = np.array([1.0, 1.0, 1.0, 1.0])
        zs = compute_zscore(spread, spread_var)
        np.testing.assert_array_almost_equal(zs, spread)  # var=1 → zscore=spread

    def test_zscore_with_variance(self):
        spread = np.array([2.0, -4.0])
        spread_var = np.array([4.0, 16.0])
        zs = compute_zscore(spread, spread_var)
        np.testing.assert_array_almost_equal(zs, [1.0, -1.0])

    def test_entry_long_signal(self):
        """Z-score < -1.5 should trigger LONG."""
        zscore = np.array([0.0, -1.6, -1.7, -0.4, 0.0])
        signals = generate_signals(zscore, entry_threshold=1.5, exit_threshold=0.5)
        assert signals[0] == 0   # flat
        assert signals[1] == 1   # long (z < -1.5)
        assert signals[2] == 1   # hold long
        assert signals[3] == 0   # exit (|z| < 0.5)
        assert signals[4] == 0   # flat

    def test_entry_short_signal(self):
        """Z-score > +1.5 should trigger SHORT."""
        zscore = np.array([0.0, 1.6, 1.8, 0.3, 0.0])
        signals = generate_signals(zscore, entry_threshold=1.5, exit_threshold=0.5)
        assert signals[0] == 0
        assert signals[1] == -1  # short
        assert signals[2] == -1  # hold short
        assert signals[3] == 0   # exit
        assert signals[4] == 0

    def test_hold_between_thresholds(self):
        """Position should hold between entry and exit thresholds."""
        zscore = np.array([0.0, -2.0, -1.0, -0.8, -0.6, -0.4])
        signals = generate_signals(zscore, entry_threshold=1.5, exit_threshold=0.5)
        assert signals[1] == 1   # enter long
        assert signals[2] == 1   # hold (z between -1.5 and -0.5)
        assert signals[3] == 1   # hold
        assert signals[4] == 1   # hold
        assert signals[5] == 0   # exit (|z| < 0.5)

    def test_flip_signal(self):
        """Should flip from long to short if z crosses +1.5."""
        zscore = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        signals = generate_signals(zscore, entry_threshold=1.5, exit_threshold=0.5)
        assert signals[0] == 1   # long
        assert signals[4] == -1  # flipped to short


# ──────────────────────────────────────────────────────────────────
# Metrics Tests
# ──────────────────────────────────────────────────────────────────


class TestMetrics:
    """Verify metric computations against hand-calculated values."""

    def test_cagr_known(self):
        # 100 → 200 over 252 days = 100% CAGR
        equity = pd.Series([100.0] + [200.0] * 251)
        result = cagr(equity)
        assert abs(result - 1.0) < 0.01  # ≈ 100%

    def test_cagr_flat(self):
        equity = pd.Series([100.0] * 100)
        assert cagr(equity) == 0.0

    def test_sharpe_ratio_positive(self):
        rng = np.random.default_rng(42)
        # Consistent positive returns
        returns = pd.Series(rng.normal(0.001, 0.01, 252))
        sr = sharpe_ratio(returns)
        assert sr > 0

    def test_sharpe_ratio_zero_std(self):
        returns = pd.Series([0.0] * 100)
        assert sharpe_ratio(returns) == 0.0

    def test_max_drawdown_known(self):
        equity = pd.Series([100, 110, 90, 95, 80, 100])
        # Peak 110 → trough 80 → DD = 30/110 ≈ 0.2727
        md = max_drawdown(equity)
        assert abs(md - 30 / 110) < 0.001

    def test_max_drawdown_no_drawdown(self):
        equity = pd.Series([100, 101, 102, 103])
        assert max_drawdown(equity) == 0.0

    def test_win_rate_known(self):
        trades = pd.Series([10, -5, 20, -3, 15, -2, 8])
        # 4 wins out of 7 → 4/7 ≈ 0.571
        assert abs(win_rate(trades) - 4 / 7) < 0.001

    def test_win_rate_empty(self):
        assert win_rate(pd.Series(dtype=float)) == 0.0

    def test_profit_factor_known(self):
        trades = pd.Series([10, -5, 20, -3])
        # profit = 30, loss = 8 → pf = 30/8 = 3.75
        assert abs(profit_factor(trades) - 3.75) < 0.001

    def test_profit_factor_no_losses(self):
        trades = pd.Series([10, 20, 30])
        assert profit_factor(trades) == float("inf")


# ──────────────────────────────────────────────────────────────────
# Data Leakage Test
# ──────────────────────────────────────────────────────────────────


class TestNoDataLeakage:
    """Verify strict temporal split with no overlap."""

    def test_split_no_overlap(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        df = pd.DataFrame({
            "date": dates,
            "series_x": range(100),
            "series_y": range(100),
        })

        split = split_data(df, train_frac=0.6, val_frac=0.2)

        # Verify sizes
        assert len(split.train) == 60
        assert len(split.validation) == 20
        assert len(split.test) == 20

        # Verify no overlap
        train_dates = set(split.train["date"])
        val_dates = set(split.validation["date"])
        test_dates = set(split.test["date"])

        assert len(train_dates & val_dates) == 0
        assert len(train_dates & test_dates) == 0
        assert len(val_dates & test_dates) == 0

    def test_split_temporal_order(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        df = pd.DataFrame({
            "date": dates,
            "series_x": range(100),
            "series_y": range(100),
        })

        split = split_data(df, train_frac=0.6, val_frac=0.2)

        # All train dates < all val dates < all test dates
        assert split.train["date"].max() < split.validation["date"].min()
        assert split.validation["date"].max() < split.test["date"].min()
