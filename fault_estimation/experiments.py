"""Experiments: batches of simulate-then-estimate runs, run in parallel.

Every experiment takes ``(p, system, seed, n_workers)`` and returns a dict of numpy
arrays, so results can be saved with :func:`save_results` and re-plotted later.
"""

import contextlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .config import (
    FAULT_TIME,
    PLACEMENT_SETS,
    RISE_THRESHOLD,
    RMS_WINDOW,
    SUCCESS_LEVELS,
    Sensor,
)
from .measurements import generate_measurements, sensor_setup
from .model import simulate
from .spf import run_spf_estimation

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # optional; the entry scripts also set the env variables
    threadpool_limits = None


def _single_thread_blas():
    """Multithreaded BLAS is ~10x slower on these small matrices; use one thread per run."""
    return threadpool_limits(limits=1) if threadpool_limits else contextlib.nullcontext()


# --- single runs ---------------------------------------------------------------


def run_case(p, system, seed):
    """Simulate the true fish, generate noisy measurements and run the filter."""
    with _single_thread_blas():
        rng = np.random.default_rng(seed)
        sensors = sensor_setup(p)
        Xfem, Xrigid = simulate(p, system)
        Z = generate_measurements(sensors, p, Xfem, Xrigid, rng)
        return run_spf_estimation(p, system, sensors, Z), (Xfem, Xrigid)


def _run_case_ek(args):
    p, system, seed = args
    result, _ = run_case(p, system, seed)
    return result.ek


def run_cases(cases, system, seed=0, n_workers=None):
    """Run many independent cases (a list of Params); returns ek estimates (n_cases, nin, nt).

    Each case gets its own random stream derived from ``seed``, so results do not
    depend on ``n_workers``. ``n_workers=1`` runs serially (handy for debugging).
    """
    seeds = np.random.SeedSequence(seed).spawn(len(cases))
    jobs = [(p, system, s) for p, s in zip(cases, seeds, strict=True)]
    n_workers = n_workers or os.cpu_count()
    if n_workers == 1:
        return np.array([_run_case_ek(job) for job in jobs])
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        return np.array(list(pool.map(_run_case_ek, jobs)))


# --- scoring -------------------------------------------------------------------


def estimator_properties(p, ek_hat, failed):
    """Rise time and RMS error for a run where actuator ``failed`` goes to zero.

    Rise time: time after the fault until the failed actuator's estimate first drops
    below RISE_THRESHOLD (NaN if it never does). RMS: error over the last RMS_WINDOW steps.
    """
    ek_true = np.ones(ek_hat.shape[0])
    ek_true[failed] = 0
    tail = ek_hat[:, -RMS_WINDOW - 1 : -1]
    rms = np.sqrt(np.mean((tail - ek_true[:, None]) ** 2))
    below = np.flatnonzero(ek_hat[failed] < RISE_THRESHOLD)
    rise = p.t[below[0]] - FAULT_TIME if below.size else np.nan
    return rise, rms


def success_score(rise, rms):
    """3 = best ... 0 = fail, using SUCCESS_LEVELS."""
    for max_rise, max_rms, score in SUCCESS_LEVELS:
        if rise < max_rise and rms < max_rms:
            return score
    return 0


def failure_health(p, actuator):
    """True health after the fault when only ``actuator`` fails."""
    ek = np.ones(p.nin)
    ek[actuator] = 0
    return ek


# --- experiments -------------------------------------------------------------------


def single_run(p, system, seed=0, n_workers=None):
    """One run with ``p`` as configured (e.g. for the time history and sigma bounds)."""
    result, (Xfem, Xrigid) = run_case(p, system, seed)
    n_avg = int(round(0.1 * p.tf / p.dt)) + 1
    return {
        "xhat": result.xhat,
        "P_diag": result.P_diag,
        "ek_est_mean": result.ek[:, -n_avg:].mean(axis=1),
        "Xfem_true": Xfem,
        "Xrigid_true": Xrigid,
    }


def filter_validation(p, system, seed=0, n_workers=None):
    """Normalized-innovation check for each sensor type."""
    out = {"lam": [], "lam_window": [], "b_lower": [], "b_upper": []}
    seeds = np.random.SeedSequence(seed).spawn(len(Sensor))
    for sensor, s in zip(Sensor, seeds, strict=True):
        result, _ = run_case(p.with_(msmt_choice=sensor), system, s)
        for key in out:
            out[key].append(getattr(result, key))
    return {key: np.array(val) for key, val in out.items()}


def actuator_failure(p, system, seed=0, n_workers=None):
    """Each sensor type x each single-actuator failure.

    Returns ek_hat (sensor, actuator, nin, nt) and rise_time, rms (sensor, actuator).
    """
    sensors = list(Sensor)
    cases = [p.with_(msmt_choice=s, ek_true2=failure_health(p, a)) for s in sensors for a in range(p.nin)]
    ek_hat = run_cases(cases, system, seed, n_workers).reshape(len(sensors), p.nin, p.nin, p.nt)
    metrics = np.array(
        [[estimator_properties(p, ek_hat[i, a], a) for a in range(p.nin)] for i in range(len(sensors))]
    )
    return {"ek_hat": ek_hat, "rise_time": metrics[..., 0], "rms": metrics[..., 1]}


SENSOR_FAILURE_PLACEMENTS = [12345, 1245, 1234, 135]
SENSOR_FAILURE_LABELS = ["Full Health", "S3 Failure", "S5 Failure", "S2 & S4 Failure"]


def sensor_failure(p, system, seed=0, n_workers=None, failed_actuator=2):
    """Each sensor type x a set of degraded placements, with one actuator failed.

    Returns ek_hat (sensor, placement, nin, nt) and rise_time, rms (sensor, placement).
    """
    sensors, places = list(Sensor), SENSOR_FAILURE_PLACEMENTS
    ek2 = failure_health(p, failed_actuator)
    cases = [p.with_(msmt_choice=s, sensor_placement=pl, ek_true2=ek2) for s in sensors for pl in places]
    ek_hat = run_cases(cases, system, seed, n_workers).reshape(len(sensors), len(places), p.nin, p.nt)
    metrics = np.array(
        [
            [estimator_properties(p, ek_hat[i, j], failed_actuator) for j in range(len(places))]
            for i in range(len(sensors))
        ]
    )
    return {
        "ek_hat": ek_hat,
        "rise_time": metrics[..., 0],
        "rms": metrics[..., 1],
        "failed_actuator": failed_actuator,
    }


CONTROL_TYPES = ["straight", "low angle turn", "high angle turn"]


def control_comparison(p, system, seed=0, n_workers=None, repeats=1):
    """Each control type x each single-actuator failure, optionally repeated.

    Returns ek_hat (control, actuator, nin, nt) from the first repeat and
    rise_time, rms (control, actuator, repeat).
    """
    cases = [
        p.with_(control_type=c, ek_true2=failure_health(p, a))
        for _ in range(repeats)
        for c in CONTROL_TYPES
        for a in range(p.nin)
    ]
    ek_hat = run_cases(cases, system, seed, n_workers).reshape(
        repeats, len(CONTROL_TYPES), p.nin, p.nin, p.nt
    )
    metrics = np.array(
        [
            [[estimator_properties(p, ek_hat[r, c, a], a) for r in range(repeats)] for a in range(p.nin)]
            for c in range(len(CONTROL_TYPES))
        ]
    )
    return {"ek_hat": ek_hat[0], "rise_time": metrics[..., 0], "rms": metrics[..., 1]}


def control_repeats(p, system, seed=0, n_workers=None):
    """control_comparison repeated 5 times to show run-to-run spread."""
    return control_comparison(p, system, seed, n_workers, repeats=5)


ACCURACY_HEALTH_LEVELS = [1, 0.9, 0.8, 0.5, 0.25, 0]


def accuracy_vs_health(p, system, seed=0, n_workers=None, actuator=3, repeats=3):
    """Estimation accuracy when one actuator drops to partial health.

    Returns health (levels,) and accuracy (repeat, level) = 1 - mean |ek_est - ek_true|.
    """
    levels = ACCURACY_HEALTH_LEVELS
    ek2 = [np.where(np.arange(p.nin) == actuator, h, 1.0) for h in levels]
    cases = [p.with_(ek_true2=e) for _ in range(repeats) for e in ek2]
    ek_hat = run_cases(cases, system, seed, n_workers).reshape(repeats, len(levels), p.nin, p.nt)
    n_avg = int(round(0.1 * p.tf / p.dt)) + 1
    ek_mean = ek_hat[..., -n_avg:].mean(axis=-1)
    accuracy = 1 - np.mean(np.abs(ek_mean - np.array(ek2)), axis=-1)
    return {"health": np.array(levels), "accuracy": accuracy}


def sensor_placement(p, system, seed=0, n_workers=None):
    """IMU and bend sensors at every placement in PLACEMENT_SETS x each actuator failure.

    Arrays are indexed (sensor, placement set, placement, actuator); unused placement
    slots are NaN. ``score`` uses :func:`success_score`.
    """
    sensors = [Sensor.IMU, Sensor.BEND]
    max_places = max(len(s) for s in PLACEMENT_SETS)
    shape = (len(sensors), len(PLACEMENT_SETS), max_places, p.nin)
    rise, rms, score = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)

    keys, cases = [], []
    for i, sensor in enumerate(sensors):
        for j, places in enumerate(PLACEMENT_SETS):
            for k, place in enumerate(places):
                for a in range(p.nin):
                    keys.append((i, j, k, a))
                    cases.append(
                        p.with_(msmt_choice=sensor, sensor_placement=place, ek_true2=failure_health(p, a))
                    )

    for key, ek_hat in zip(keys, run_cases(cases, system, seed, n_workers), strict=True):
        rise[key], rms[key] = estimator_properties(p, ek_hat, key[3])
        score[key] = success_score(rise[key], rms[key])
    return {"rise_time": rise, "rms": rms, "score": score}


def animation(p, system, seed=0, n_workers=None):
    """Same as single_run; plotted as an animation of the estimated and true fish."""
    return single_run(p, system, seed, n_workers)


# --- saving ----------------------------------------------------------------------------


def save_results(path, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **results)


def load_results(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
