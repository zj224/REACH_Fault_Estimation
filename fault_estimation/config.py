"""Parameters, sensor definitions and noise settings.

State layout
------------
* FEM state, length ``ns`` (126): node positions ``[x, y, th]`` for nodes 1..21,
  then node velocities in the same order. Node 1 is the tail, node 21 the head.
* Filter state, length ``nx`` (134): FEM state, rigid-body ``[x, y, th]`` (3),
  then actuator health ``ek`` (5).
* Actuator health index 0 is the tail section and index 4 the head. Plots label
  actuators A1 (head) to A5 (tail), i.e. state index ``i`` is shown as ``A{5 - i}``.

``Params`` is the 1 m, five-actuator simulated fish (paper Sec. V). ``ExperimentParams``
is the three-actuator UCSD robot fish used with the recorded data (paper Sec. VI).
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

        self._set_body()
        self._set_control()

        # sizes
        self.nn = self.n + 1  # nodes
        self.ndof = 3 * self.nn  # x, y, th per node
        self.ns = 2 * self.ndof  # positions + velocities
        self.n_health = len(self.actuated)  # actuator health states
        self.nx = self.ns + 3 + self.n_health  # filter state
        self.sl_rigid = slice(self.ns, self.ns + 3)
        self.sl_ek = slice(self.ns + 3, self.nx)

        # hydrodynamic drag
        rho_water = 10**3
        c_drag = 0.167
        area = 2 * np.pi * self.rvec * self.L / self.n
        uwvec = 0.5 * c_drag * area * rho_water
        self.uwvec = np.append(uwvec, uwvec[-1])

        # state indices and selection matrices
        self.ix = np.arange(self.nn) * 3
        self.iy = self.ix + 1
        self.ith = self.ix + 2
        eye = np.eye(self.nn)
        self.Hx = np.zeros((self.nn, self.ns))
        self.Hx[:, self.ix] = eye
        self.Hy = np.zeros((self.nn, self.ns))
        self.Hy[:, self.iy] = eye

    def _set_body(self):
        """Fish geometry: 1 m uniform beam, every input section is an actuator."""
        self.L = 1.0  # length [m]
        self.l_elem = np.full(self.n, self.L / self.n)  # element lengths [m], tail first
        self.E_elem = np.full(self.n, 1e6)  # element Young's modulus [N/m^2]
        self.rvec = np.full(self.n, 0.05)  # element radius [m]
        self.xnode = np.linspace(-self.L / 2, self.L / 2, self.n + 1)
        self.actuated = np.arange(self.nin)  # inputs that have a health state

    def _set_control(self):
        self.A = 2
        self.w = 1.5  # oscillation frequency (baseline 1.5)
        self.amp_co = np.array([2, 2, 8, 10, 10], dtype=float)
        self.offset_co = np.array([3.5, 6.5, 6.5, 5.5, 1.5])

    def input_gains(self, ek):
        """Torque gain per input for health ``ek`` (n_health,) or (n_health, N); unactuated inputs get 0."""
        ek = np.asarray(ek, dtype=float)
        if ek.ndim == 1:
            ek = ek[:, None]
        gains = np.zeros((self.nin, ek.shape[1]))
        gains[self.actuated] = ek
        return gains

    def with_(self, **changes):
        """Return a copy with some settings changed."""
        return replace(self, **changes)

    def failed_actuator(self):
        """Index of the actuator with the lowest true health after the fault."""
        return int(np.argmin(self.ek_true2))


# =============================================================================
# Experimental validation: three-actuator UCSD robot fish (paper Sec. VI)
# =============================================================================

# Sections from head to tail; each is split into 4 elements.
EXP_SECTIONS = ["head", "A1", "coupling", "A2", "coupling", "A3", "tail"]
EXP_SECTION_LENGTHS = {"head": 0.04, "A1": 0.092, "A2": 0.092, "A3": 0.092, "coupling": 0.02, "tail": 0.15}
EXP_ELEMS_PER_SECTION = 4
# Young's modulus [N/m^2]: actuators are tuned to the full-health trial; head and couplings are rigid
# plastic; the tail is soft silicone.
EXP_SECTION_E = {
    "head": 2e9,
    "A1": 1.4096e6,
    "coupling": 2e9,
    "A2": 0.9121e6,
    "A3": 0.9934e6,
    "tail": 1e6,
}
EXP_RADIUS = 0.04  # [m]

# Bend sensors: angle between the marker segments on either side of each actuator
# (node numbers, 1 = tail, 29 = head). Rows are A1 (head), A2, A3, matching CSV columns 5-7.
EXP_BEND_MARKER_NODES = [
    [28, 26, 20, 18],
    [20, 18, 12, 10],
    [12, 10, 5, 4],
]
EXP_BEND_VAR = 0.01  # measurement noise variance [rad^2]

# Bend sensor calibration: angle = p1 x^2 + p2 x + p3, with x = (raw - 20000) / 20000.
# Rows are A1, A2, A3 (originally f1.mat, f2.mat, f3.mat from MATLAB's fit()).
BEND_RAW_OFFSET = 20000.0
BEND_CALIBRATION = [
    (-67.827702864982044, 15.635658323261929, 0.50427732971526806),
    (-111.22238665793809, 21.99178255271174, -0.51433897033056331),
    (-18.627077794844084, 5.8589162078156756, 1.0225723105527189),
]
EXP_N_ROWS = 500  # CSV rows used from each trial

# Filter tuning for the experiment (process noise is not scaled by dt here)
EXP_Q_STATE = 1e-8  # FEM + rigid-body states, pattern [a, a, 0.1 a] per node
EXP_Q_EK = 1e-4
EXP_P0_POS_PER_NODE = [0.0001, 0.0042, 0.025]
EXP_P0_VEL_PER_NODE = [0.0031, 0.0149, 0.1632]
EXP_P0_RIGID = [0.0740, 0.0209, 0.1567]
EXP_P0_EK = [0.03123, 0.01785, 0.01654]  # tail-end actuator (A3) first


@dataclass
class ExperimentParams(Params):
    """Model of the three-actuator UCSD robot fish used with the recorded bend sensor data.

    The 28 elements form 7 input sections (tail, A3, coupling, A2, coupling, A1, head);
    only the three actuator sections are driven. Health vectors are ordered A3, A2, A1
    (tail end first), like the simulation.
    """

    tf: float = 33.26
    n: int = 28
    nin: int = 7
    msmt_choice: Sensor = Sensor.BEND
    sensor_placement: int = 123
    ek_predict: np.ndarray = field(default_factory=lambda: np.ones(3))
    ek_true1: np.ndarray = field(default_factory=lambda: np.ones(3))
    ek_true2: np.ndarray = field(default_factory=lambda: np.ones(3))

    def _set_body(self):
        sections = EXP_SECTIONS[::-1]  # tail first
        k = EXP_ELEMS_PER_SECTION
        self.l_elem = np.repeat([EXP_SECTION_LENGTHS[s] / k for s in sections], k)
        self.E_elem = np.repeat([EXP_SECTION_E[s] for s in sections], k)
        self.L = self.l_elem.sum()
        self.rvec = np.full(self.n, EXP_RADIUS)
        self.xnode = -self.L / 2 + np.concatenate([[0], np.cumsum(self.l_elem)])
        self.actuated = np.array([i for i, s in enumerate(sections) if s.startswith("A")])

    def _set_control(self):
        self.A = 0.2
        self.w = 0.5
        self.amp_co = np.array([0, 10, 0, 30, 0, 50, 0], dtype=float)  # pump input ratio 1:3:5
        self.offset_co = np.zeros(self.nin)
