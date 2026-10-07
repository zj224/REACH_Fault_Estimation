"""Sensor models: which states each sensor sees, simulated readings and predictions."""

from dataclasses import dataclass

import numpy as np

from .config import (
    BEND_MARKER_NODES,
    BEND_VAR,
    GPS_NODES,
    GPS_VAR,
    IMU_ACCEL_VAR,
    IMU_GYRO_VAR,
    IMU_NODES,
    Sensor,
)
from .model import fem_to_inertial


@dataclass
class SensorSetup:
    """Indices into the state for one sensor type and placement."""

    sensor: Sensor
    R: np.ndarray  # measurement noise covariance
    pos_idx: np.ndarray = None  # GPS: inertial x, y rows
    vel_idx: np.ndarray = None  # IMU: local x, y velocity rows (differenced to acceleration)
    gyro_idx: np.ndarray = None  # IMU: angular rate rows
    marker_idx: np.ndarray = None  # bend: x rows of the 4 marker nodes per sensor


def _x_row(node):
    """0-based row of a node's x position in the FEM state (nodes are 1-based)."""
    return 3 * (node - 1)


def sensor_setup(p):
    """Build the sensor indices and noise covariance for ``p.msmt_choice``/``p.sensor_placement``."""
    positions = [int(c) - 1 for c in str(p.sensor_placement)]

    if p.msmt_choice == Sensor.GPS:
        rows = [r for i in positions for r in (_x_row(GPS_NODES[i]), _x_row(GPS_NODES[i]) + 1)]
        return SensorSetup(p.msmt_choice, np.diag(np.full(len(rows), GPS_VAR)), pos_idx=np.array(rows))

    if p.msmt_choice == Sensor.IMU:
        vel = [p.ndof + r for i in positions for r in (_x_row(IMU_NODES[i]), _x_row(IMU_NODES[i]) + 1)]
        gyro = [p.ndof + _x_row(IMU_NODES[i]) + 2 for i in positions]
        R = np.diag([IMU_ACCEL_VAR] * len(vel) + [IMU_GYRO_VAR] * len(gyro))
        return SensorSetup(p.msmt_choice, R, vel_idx=np.array(vel), gyro_idx=np.array(gyro))

    if p.msmt_choice == Sensor.BEND:
        return bend_sensor_setup([BEND_MARKER_NODES[i] for i in positions], BEND_VAR)

    raise ValueError(f"Unknown sensor {p.msmt_choice!r}")


def bend_sensor_setup(marker_nodes, var):
    """Bend sensors from a list of 4 marker node numbers per sensor, each with noise variance ``var``."""
    markers = [_x_row(node) for nodes in marker_nodes for node in nodes]
    return SensorSetup(Sensor.BEND, np.diag(np.full(len(marker_nodes), var)), marker_idx=np.array(markers))


def bend_angles(marker_idx, Xinertial):
    """Angle between the two marker segments of each bend sensor."""
    x = Xinertial[marker_idx].reshape(-1, 4, Xinertial.shape[1])
    y = Xinertial[marker_idx + 1].reshape(-1, 4, Xinertial.shape[1])
    ax, ay = x[:, 0] - x[:, 1], y[:, 0] - y[:, 1]
    bx, by = x[:, 2] - x[:, 3], y[:, 2] - y[:, 3]
    return np.arctan2(ax * by - bx * ay, ax * bx + ay * by)


def measure(sensors, p, Xfem, Xrigid, v_prev):
    """Noise-free readings for FEM states (ns, N) and rigid poses (3, N).

    ``v_prev`` holds the IMU velocity rows one step earlier (used to form acceleration).
    """
    if sensors.sensor == Sensor.GPS:
        return fem_to_inertial(Xfem, p, Xrigid)[sensors.pos_idx]
    if sensors.sensor == Sensor.IMU:
        acc = (Xfem[sensors.vel_idx] - v_prev) / p.dt
        return np.vstack([acc, Xfem[sensors.gyro_idx]])
    return bend_angles(sensors.marker_idx, fem_to_inertial(Xfem, p, Xrigid))


def generate_measurements(sensors, p, Xfem, Xrigid, rng):
    """Noisy readings over a whole simulated trajectory, shape (nz, nt)."""
    v_prev = None
    if sensors.sensor == Sensor.IMU:
        v = Xfem[sensors.vel_idx]
        v_prev = np.zeros_like(v)
        v_prev[:, 1:] = v[:, :-1]
    z = measure(sensors, p, Xfem, Xrigid, v_prev)
    noise_std = np.sqrt(np.diag(sensors.R))[:, None]
    return z + noise_std * rng.standard_normal(z.shape)


def predict_measurements(sensors, p, X, x_prior):
    """Predicted readings for filter sigma points X (nx, nsp); IMU uses the prior estimate's velocity."""
    v_prev = x_prior[sensors.vel_idx][:, None] if sensors.sensor == Sensor.IMU else None
    return measure(sensors, p, X[: p.ns], X[p.sl_rigid], v_prev)
