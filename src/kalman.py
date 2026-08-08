"""
kalman.py — From-scratch linear Kalman filter for pairs trading.

Implements the two-state (alpha, beta) Kalman filter to dynamically
estimate the hedge ratio between two cointegrated time series.

**No filterpy** — pure NumPy implementation.

State model
-----------
State vector:  x = [alpha, beta]'  (intercept and hedge ratio)
Transition:    x_{t} = F * x_{t-1} + w,  w ~ N(0, Q)
Observation:   z_{t} = H_{t} * x_{t} + v,  v ~ N(0, R)

Where F = I(2), H_t = [1, series2_t], and Q = (delta/(1-delta)) * I(2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class KalmanFilter:
    """Linear Kalman filter for dynamic hedge-ratio estimation.

    Parameters
    ----------
    delta : float
        Controls process noise magnitude.  Smaller = smoother hedge ratio.
        Typical range: 1e-5 to 1e-3.  Default 1e-4.
    R : float
        Observation noise variance.  Default 1e-3.
    """

    def __init__(self, delta: float = 1e-4, R: float = 1e-3) -> None:
        self.delta = delta
        self.R = R

        # State transition matrix (random walk)
        self.F = np.eye(2)

        # Process noise covariance
        self.Q = (delta / (1 - delta)) * np.eye(2)

        # --- mutable state (reset per run) ---
        self.x: np.ndarray = np.zeros(2)          # state [alpha, beta]
        self.P: np.ndarray = np.eye(2) * 1.0      # state covariance

    # ------------------------------------------------------------------
    # Core Kalman steps
    # ------------------------------------------------------------------

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Prediction step.

        Returns
        -------
        x_pred : np.ndarray  shape (2,)
            Predicted state.
        P_pred : np.ndarray  shape (2, 2)
            Predicted state covariance.
        """
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(
        self,
        z: float,
        H: np.ndarray,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Update step.

        Parameters
        ----------
        z : float
            Observation (dependent series value at time t).
        H : np.ndarray  shape (1, 2)
            Observation matrix [1, x_t] where x_t is independent series value.
        x_pred : np.ndarray
            Predicted state from ``predict()``.
        P_pred : np.ndarray
            Predicted covariance from ``predict()``.

        Returns
        -------
        x_upd : np.ndarray  shape (2,)
            Updated state estimate.
        P_upd : np.ndarray  shape (2, 2)
            Updated state covariance.
        y_innov : float
            Innovation (measurement residual).
        S : float
            Innovation variance.
        """
        # Innovation
        y_innov = z - (H @ x_pred).item()

        # Innovation covariance  S = H P H' + R
        S = (H @ P_pred @ H.T + self.R).item()

        # Kalman gain  K = P_pred H' / S
        K = (P_pred @ H.T) / S  # shape (2, 1)

        # State update
        x_upd = x_pred + K.flatten() * y_innov

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(2) - K @ H
        P_upd = I_KH @ P_pred @ I_KH.T + (K * self.R) @ K.T

        # Store
        self.x = x_upd
        self.P = P_upd

        return x_upd, P_upd, y_innov, S

    # ------------------------------------------------------------------
    # Full-series filtering
    # ------------------------------------------------------------------

    def filter_series(
        self,
        y: np.ndarray | pd.Series,
        x: np.ndarray | pd.Series,
    ) -> dict[str, np.ndarray]:
        """Run the Kalman filter over two aligned series.

        Parameters
        ----------
        y : array-like  shape (T,)
            Dependent series (e.g. asset A prices).
        x : array-like  shape (T,)
            Independent series (e.g. asset B prices).

        Returns
        -------
        dict with keys:
            alpha       — filtered intercept        (T,)
            beta        — filtered hedge ratio       (T,)
            spread      — y - (alpha + beta * x)     (T,)
            spread_var  — innovation variance S      (T,)
            alpha_std   — uncertainty on alpha       (T,)
            beta_std    — uncertainty on beta        (T,)
        """
        y = np.asarray(y, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        T = len(y)
        assert len(x) == T, "Series must be the same length."

        # Reset state
        self.x = np.zeros(2)
        self.P = np.eye(2) * 1.0

        # Pre-allocate output arrays
        alphas = np.empty(T)
        betas = np.empty(T)
        spreads = np.empty(T)
        spread_vars = np.empty(T)
        alpha_stds = np.empty(T)
        beta_stds = np.empty(T)

        for t in range(T):
            H = np.array([[1.0, x[t]]])

            # Predict
            x_pred, P_pred = self.predict()

            # Update
            x_upd, P_upd, y_innov, S = self.update(y[t], H, x_pred, P_pred)

            alphas[t] = x_upd[0]
            betas[t] = x_upd[1]
            spreads[t] = y_innov          # prediction error = spread residual
            spread_vars[t] = S
            alpha_stds[t] = np.sqrt(P_upd[0, 0])
            beta_stds[t] = np.sqrt(P_upd[1, 1])

        return {
            "alpha": alphas,
            "beta": betas,
            "spread": spreads,
            "spread_var": spread_vars,
            "alpha_std": alpha_stds,
            "beta_std": beta_stds,
        }
