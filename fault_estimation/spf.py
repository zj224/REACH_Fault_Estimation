"""Sigma-point (unscented) filter that jointly estimates the fish state and actuator health."""

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from scipy.stats import chi2

from .config import (
    GPS_FILTER_R_SCALE,
    P0_EK_STD,
    P0_POS_PER_NODE,
    P0_RIGID,
    P0_VEL_PER_NODE,
    PROCESS_NOISE,
    SIGMA_SCALE,
    VALIDATION_GATE,
    VALIDATION_NZ,
    VALIDATION_WINDOW,
    Sensor,
)
from .measurements import predict_measurements
from .model import get_controls, step_dynamics


@dataclass
class SPFResult:
    xhat: np.ndarray  # state estimates (nx, nt)
    P_diag: np.ndarray  # covariance diagonals (nx, nt)
    lam: np.ndarray  # normalized innovation squared per step (nt,)
    lam_window: np.ndarray  # moving average of lam over VALIDATION_WINDOW steps (nt,)
    b_lower: float  # chi-square acceptance bounds for lam_window
    b_upper: float
    nin: int  # number of actuators

    @property
    def ek(self):
        """Actuator health estimates (nin, nt)."""
        return self.xhat[-self.nin :]


@dataclass
class _Filter:
    """Constant filter quantities shared by every step."""

    system: object
    sensors: object
    Q: np.ndarray
    R: np.ndarray
    WM: np.ndarray  # mean weights
    WC: np.ndarray  # covariance weights
    nsig: float


def _make_filter(p, system, sensors):
    nx, nsig = p.nx, SIGMA_SCALE
    Wi = 0.5 / nsig**2
    W0M = (nsig**2 - nx) / nsig**2
    W0C = W0M + 3 - nsig**2 / nx
    WM = np.concatenate([[W0M], np.full(2 * nx, Wi)])
    WC = np.concatenate([[W0C], np.full(2 * nx, Wi)])

    q_state, q_ek = PROCESS_NOISE[p.msmt_choice]
    Q = np.diag(np.concatenate([q_state * np.tile([1, 1, 0.1], 2 * p.nn + 1), np.full(p.nin, q_ek)]))
    R = sensors.R * (GPS_FILTER_R_SCALE if p.msmt_choice == Sensor.GPS else 1)
    return _Filter(system, sensors, Q, R, WM, WC, nsig)


def _initial_covariance(p):
    return np.diag(
        np.concatenate(
            [
                np.tile(P0_POS_PER_NODE, p.nn),
                np.tile(P0_VEL_PER_NODE, p.nn),
                P0_RIGID,
                np.square(P0_EK_STD),
            ]
        )
    )


def _sigma_offsets(P, nsig):
    """[0, -S, +S] with S the scaled lower Cholesky factor of P."""
    S = nsig * scipy.linalg.cholesky(P).T  # upper-triangle factor, as MATLAB's chol
    return np.hstack([np.zeros((P.shape[0], 1)), -S, S])


def _predict_state(X, p, f, k):
    """Propagate sigma points (nx, nsp) through the dynamics; ek is held constant."""
    ek = X[p.sl_ek]
    uk = ek * get_controls(p, k)
    Xfem, Xrigid = step_dynamics(X[: p.ns], X[p.sl_rigid], uk, p, f.system)
    X_new = X.copy()
    X_new[: p.ns] = Xfem
    X_new[p.sl_rigid] = Xrigid
    return X_new


def spf_step(p, f, x_prior, P_prior, z, k):
    """One predict + update step. Returns the posterior mean, covariance and NIS value."""
    # predict
    X = _predict_state(x_prior[:, None] + _sigma_offsets(P_prior, f.nsig), p, f, k)
    x_pred = X @ f.WM
    dX = X - x_pred[:, None]
    P_pred = (dX * f.WC) @ dX.T + f.Q / np.sqrt(p.dt)

    # update (sigma points redrawn around the prediction to include Q)
    dX = _sigma_offsets(P_pred, f.nsig)
    Z = predict_measurements(f.sensors, p, x_pred[:, None] + dX, x_prior)
    z_pred = Z @ f.WM
    dZ = Z - z_pred[:, None]
    Pxz = (dX * f.WC) @ dZ.T
    Pzz = (dZ * f.WC) @ dZ.T + f.R
    Pzz_chol = scipy.linalg.cho_factor(Pzz)
    K = scipy.linalg.cho_solve(Pzz_chol, Pxz.T).T

    innovation = z - z_pred
    x_post = x_pred + K @ innovation
    P_post = P_pred - K @ Pxz.T
    nis = innovation @ scipy.linalg.cho_solve(Pzz_chol, innovation)
    return x_post, P_post, nis


def run_spf_estimation(p, system, sensors, Z):
    """Run the filter over measurements Z (nz, nt)."""
    f = _make_filter(p, system, sensors)

    x = np.zeros(p.nx)
    x[p.sl_ek] = p.ek_predict
    P = _initial_covariance(p)

    xhat = np.zeros((p.nx, p.nt))
    P_diag = np.zeros((p.nx, p.nt))
    xhat[:, 0], P_diag[:, 0] = x, np.diag(P)

    alpha = 1 - VALIDATION_GATE
    N = VALIDATION_WINDOW
    b_lower = chi2.ppf(alpha / 2, N * VALIDATION_NZ) / N
    b_upper = chi2.ppf(1 - alpha / 2, N * VALIDATION_NZ) / N
    lam = np.full(p.nt, np.nan)
    lam_window = np.full(p.nt, np.nan)

    for k in range(1, p.nt):
        x, P, lam[k] = spf_step(p, f, x, P, Z[:, k], k)
        xhat[:, k], P_diag[:, k] = x, np.diag(P)
        if k >= N:
            lam_window[k] = lam[k - N + 1 : k + 1].mean()

    return SPFResult(xhat, P_diag, lam, lam_window, b_lower, b_upper, p.nin)
