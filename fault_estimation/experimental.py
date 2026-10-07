"""Experimental validation: run REACH on bend sensor data recorded from the UCSD robot fish.

Each trial is a CSV in ``data/experimental/`` named by the actuator health of A1 (head),
A2 and A3 in percent, e.g. ``health_080_050_100.csv``. Columns (no header, 12 Hz):

    time [s], pump inputs A1-A3 (servo angle), raw bend sensor outputs A1-A3
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .config import (
    BEND_CALIBRATION,
    BEND_RAW_OFFSET,
    EXP_BEND_MARKER_NODES,
    EXP_BEND_VAR,
    EXP_N_ROWS,
    EXP_P0_EK,
    EXP_P0_POS_PER_NODE,
    EXP_P0_RIGID,
    EXP_P0_VEL_PER_NODE,
    EXP_Q_EK,
    EXP_Q_STATE,
    ExperimentParams,
)
from .experiments import _single_thread_blas
from .measurements import bend_sensor_setup
from .model import setup_system
from .spf import initial_covariance, process_noise, run_spf_estimation

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "experimental"

# Trials shown in paper Fig. 9, as health of [A1, A2, A3]
PAPER_TRIALS = [
    (1, 1, 1),
    (0.5, 1, 1),
    (0, 1, 1),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0.8, 1, 1),
    (0.8, 0.5, 1),
    (0.5, 0.8, 1),
]


def trial_path(health, data_dir=DATA_DIR):
    """CSV for actuator health [A1, A2, A3]."""
    return Path(data_dir) / ("health_" + "_".join(f"{round(100 * h):03d}" for h in health) + ".csv")


def bend_angles_from_raw(raw):
    """Calibrated bend angles [rad] from raw sensor outputs (N, 3)."""
    x = (raw - BEND_RAW_OFFSET) / BEND_RAW_OFFSET
    return np.column_stack([np.polyval(c, x[:, i]) for i, c in enumerate(BEND_CALIBRATION)])


def load_trial(path, n_rows=EXP_N_ROWS):
    """Measurement times (N,) and zero-mean bend angles (N, 3) for one trial; the first row is dropped."""
    data = np.loadtxt(path, delimiter=",")[:n_rows]
    angles = bend_angles_from_raw(data[:, 4:7])
    angles -= angles.mean(axis=0)
    return data[1:, 0], angles[1:]


def _round2(x):
    """Round to 0.01 with halves away from zero (as MATLAB's round)."""
    return np.floor(np.asarray(x) * 100 + 0.5) / 100


def measurements_on_grid(p, times, angles):
    """Place measurements on the filter time grid, (3, nt); steps without one are NaN.

    A measurement is used at the step whose time matches it to 0.01 s; measurements are
    consumed in order, and any after ``p.tf`` are dropped.
    """
    Z = np.full((angles.shape[1], p.nt), np.nan)
    t_meas, t_grid = _round2(times), _round2(p.t)
    j = 0
    for k in range(1, p.nt):
        if j < len(times) and t_meas[j] == t_grid[k]:
            Z[:, k] = angles[j]
            j += 1
    return Z


def run_trial(health, p=None, system=None, data_dir=DATA_DIR):
    """Estimate actuator health from one recorded trial; returns ek_hat (3, nt) ordered A3, A2, A1."""
    p = p or ExperimentParams()
    system = system or setup_system(p)
    with _single_thread_blas():
        times, angles = load_trial(trial_path(health, data_dir))
        Z = measurements_on_grid(p, times, angles)
        sensors = bend_sensor_setup(EXP_BEND_MARKER_NODES, EXP_BEND_VAR)
        Q = process_noise(p, EXP_Q_STATE, EXP_Q_EK)
        P0 = initial_covariance(p, EXP_P0_POS_PER_NODE, EXP_P0_VEL_PER_NODE, EXP_P0_RIGID, EXP_P0_EK)
        return run_spf_estimation(p, system, sensors, Z, Q=Q, P0=P0).ek


def _run_trial_job(args):
    return run_trial(*args)


def experimental_validation(trials=PAPER_TRIALS, p=None, n_workers=None, data_dir=DATA_DIR):
    """Run every trial in parallel.

    Returns health (trials, 3) as [A1, A2, A3] and ek_hat (trials, 3, nt) as A3, A2, A1.
    """
    p = p or ExperimentParams()
    system = setup_system(p)
    jobs = [(h, p, system, data_dir) for h in trials]
    if n_workers == 1:
        ek_hat = [_run_trial_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            ek_hat = list(pool.map(_run_trial_job, jobs))
    return {"health": np.array(trials, dtype=float), "ek_hat": np.array(ek_hat)}
