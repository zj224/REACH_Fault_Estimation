"""Parameters, sensor definitions and noise settings.

State layout
------------
* FEM state, length ``ns`` (126): node positions ``[x, y, th]`` for nodes 1..21,
  then node velocities in the same order. Node 1 is the tail, node 21 the head.
* Filter state, length ``nx`` (134): FEM state, rigid-body ``[x, y, th]`` (3),
  then actuator health ``ek`` (5).
* Actuator health index 0 is the tail section and index 4 the head. Plots label
  actuators A1 (head) to A5 (tail), i.e. state index ``i`` is shown as ``A{5 - i}``.
"""

from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np


class Sensor(str, Enum):
    GPS = "GPS"
    IMU = "IMU"
    BEND = "actBendAngs"


SENSOR_LABELS = {Sensor.GPS: "GPS", Sensor.IMU: "IMU", Sensor.BEND: "Bend Sensor"}

# --- Sensor locations -------------------------------------------------------
# A sensor placement such as 135 means "sensors at positions 1, 3 and 5".
# Position 1 is nearest the head. Values are node numbers (1 = tail, 21 = head).
GPS_NODES = [21, 15, 11, 7, 3]
IMU_NODES = [21, 15, 11, 7, 1]
# Each bend sensor measures the angle between two marker segments (4 nodes).
BEND_MARKER_NODES = [
    [17, 18, 20, 21],
    [13, 14, 16, 17],
    [9, 10, 12, 13],
    [5, 6, 8, 9],
    [1, 2, 4, 5],
]

# --- Measurement noise (variances) -------------------------------------------
GPS_VAR = 1.0  # per axis
IMU_ACCEL_VAR = 0.0061  # per axis
IMU_GYRO_VAR = 0.002
BEND_VAR = 0.001
GPS_FILTER_R_SCALE = 1.65  # filter uses an inflated R for GPS

# --- Process noise (variances) ------------------------------------------------
# FEM + rigid-body states use the pattern [a, a, 0.1 a] per node; ek states use b.
PROCESS_NOISE = {
    Sensor.BEND: (0.001 * 10 ** (-8), 0.1 * 10 ** (-3)),
    Sensor.IMU: (100 * 0.01 * 10 ** (-8), 100 * 0.001 * 10 ** (-3)),
    Sensor.GPS: (1000 * 10 ** (-7), 0.02 * 10 ** (-3)),
}

# --- Initial filter covariance (variances) -------------------------------------
P0_POS_PER_NODE = [10 ** (-8), 10 ** (-8), 0.0005]
P0_VEL_PER_NODE = [0.0001, 0.002, 0.05]
P0_RIGID = [0.0007, 0.01, 0.01]
P0_EK_STD = [0.0705, 0.069, 0.0387, 0.0423, 0.0276]

# --- Sigma-point filter / validation -----------------------------------------
SIGMA_SCALE = 0.5
VALIDATION_GATE = 0.95
VALIDATION_WINDOW = 10
VALIDATION_NZ = 5  # measurements per step assumed by the chi-square bounds

# --- Fault injection and success scoring -------------------------------------
FAULT_TIME = 1.0  # [s] actuators use ek_true1 up to here, ek_true2 after
RISE_THRESHOLD = 0.1  # failed actuator estimate must drop below this
RMS_WINDOW = 100  # steps at the end of the run used for the RMS error
# (max rise time [s], max RMS error, score); first row that passes wins, else 0
SUCCESS_LEVELS = [(1.0, 0.15, 3), (1.5, 0.2, 2), (2.0, 0.25, 1)]

PLACEMENT_SETS = [
    [1, 2, 3, 4, 5],
    [12, 13, 14, 15, 23, 24, 25, 34, 35, 45],
    [123, 124, 125, 134, 135, 145, 234, 235, 245, 345],
    [1234, 1235, 1245, 1345, 2345],
]

# --- Control presets: (amplitude scale C_A, frequency scale C_w, offset scale C_off)
CONTROL_PRESETS = {
    "straight": (1.0, 1.0, 0.0),
    "turning": (0.8, 1.0, 0.5),
    "low angle turn": (0.8, 1.0, 0.3),
    "high angle turn": (0.8, 1.0, 0.8),
}


def _ones5():
    return np.ones(5)


@dataclass
class Params:
    """Run settings. Derived quantities are recomputed whenever a copy is made
    with :meth:`with_`."""

    tf: float = 1.0  # final time [s]
    t_pause: float = 0.01  # animation frame pause [s]
    dt: float = 0.01
    n: int = 20  # number of finite elements
    nin: int = 5  # number of actuators
    control_type: str = "straight"
    msmt_choice: Sensor = Sensor.IMU
    sensor_placement: int = 12345
    ek_predict: np.ndarray = field(default_factory=_ones5)  # filter initial guess
    ek_true1: np.ndarray = field(default_factory=_ones5)  # true health before FAULT_TIME
    ek_true2: np.ndarray = field(default_factory=_ones5)  # true health after FAULT_TIME

    def __post_init__(self):
        self.msmt_choice = Sensor(self.msmt_choice)
        if self.control_type not in CONTROL_PRESETS:
            raise ValueError(f"Unknown control_type {self.control_type!r}")

        # time
        self.nt = int(round(self.tf / self.dt)) + 1
        self.t = np.linspace(0, self.tf, self.nt)
        self.k_fault = int(round(FAULT_TIME / self.dt))

        # sizes
        self.nn = self.n + 1  # nodes
        self.ndof = 3 * self.nn  # x, y, th per node
        self.ns = 2 * self.ndof  # positions + velocities
        self.nx = self.ns + 3 + self.nin  # filter state
        self.sl_rigid = slice(self.ns, self.ns + 3)
        self.sl_ek = slice(self.ns + 3, self.nx)

        # fish geometry and hydrodynamic drag
        self.L = 1.0  # length [m]
        self.rvec = np.full(self.n, 0.05)  # element radius [m]
        rho_water = 10**3
        c_drag = 0.167
        area = 2 * np.pi * self.rvec * self.L / self.n
        uwvec = 0.5 * c_drag * area * rho_water
        self.uwvec = np.append(uwvec, uwvec[-1])
        self.xnode = np.linspace(-self.L / 2, self.L / 2, self.nn)

        # control
        self.A = 2
        self.w = 1.5  # oscillation frequency (baseline 1.5)
        self.amp_co = np.array([2, 2, 8, 10, 10], dtype=float)
        self.offset_co = np.array([3.5, 6.5, 6.5, 5.5, 1.5])

        # state indices and selection matrices
        self.ix = np.arange(self.nn) * 3
        self.iy = self.ix + 1
        self.ith = self.ix + 2
        eye = np.eye(self.nn)
        self.Hx = np.zeros((self.nn, self.ns))
        self.Hx[:, self.ix] = eye
        self.Hy = np.zeros((self.nn, self.ns))
        self.Hy[:, self.iy] = eye

    def with_(self, **changes):
        """Return a copy with some settings changed."""
        return replace(self, **changes)

    def failed_actuator(self):
        """Index of the actuator with the lowest true health after the fault."""
        return int(np.argmin(self.ek_true2))
