# REACH: Real-time Estimator of Actuator Control and Health

Simulation code for the paper **"Real-time Estimator of Actuator Control and Health (REACH) on an Eel-Inspired Soft Robot"**
by Zhangjingyi Jiang, Myungsun Park, Michael T. Tolley and Mark Campbell.

Published at IEEE RoboSoft 2025: [doi:10.1109/RoboSoft63089.2025.11020927](https://doi.org/10.1109/RoboSoft63089.2025.11020927)
· arXiv: <https://arxiv.org/html/2608.14865v1>

REACH estimates actuator health for a soft robot that swims like an eel (anguilliform swimming). Actuator health
is the ratio of the torque an actuator actually produces to the torque it was asked to produce:

- `0` means the actuator has failed completely
- `1` means it works fully
- values above `1` mean it is over-actuating

REACH appends the health of each actuator to the state of a finite-element (FEM) model of the swimming robot. A
sigma-point Kalman filter then estimates the robot state and the actuator health together, using GPS, IMU or bend
sensor measurements. A normalized-innovation chi-square test checks that the filter is statistically consistent.

## How it works

1. **Robot model (ASSRSimP).** The fish is a 1 m long elastic beam with no fixed ends, split into 20 finite
   elements, with five torque actuators spread evenly along its length. The model includes quadratic hydrodynamic
   drag, and the rigid-body pose is tracked apart from the body's deformation (`model.py`).
2. **Control.** Each actuator receives a sinusoidal torque
   `u_i = C_A·A_i·sin(C_ω·ω·t + φ_i) + C_τ^O·τ_i^O`. Three presets set the gait: linear swimming, wide turning
   and tight turning (`config.CONTROL_PRESETS`).
3. **Fault injection.** The true actuator health is `ek_true1` until `FAULT_TIME = 1 s` and `ek_true2` after it.
   Each actuator's torque is multiplied by its health: `u_i^REACH = H_i·u_i`.
4. **Sensors.** The sensor models are GPS (2D position), IMU (acceleration and angular rate) and bend sensors
   (the angle between two marker segments on each actuator). A sensor placement such as `135` means sensors at
   positions 1, 3 and 5, where position 1 is nearest the head (`measurements.py`).
5. **Estimator.** A sigma-point (unscented) Kalman filter estimates a 134-element state: 126 FEM states, the
   3-element rigid-body pose and the 5 actuator health values. It also records the windowed normalized innovation
   squared, which is compared against chi-square bounds for filter validation (`spf.py`).
6. **Scoring.** Each run is scored on two metrics:
   - **Rise time:** how long after the fault the failed actuator's estimate takes to drop below 0.1.
   - **RMS error:** the error in the health estimates over the last 100 steps.

   The sensor placement study combines the two into a success score from 0 to 3 (`experiments.py`).

## Repository layout

```
REACH_Fault_Estimation/
├── main.py                  # run one experiment (choose it with EXPERIMENT) and plot it
├── main_sensor_choice.py    # sensor quantity / placement study (paper Fig. 7)
├── fault_estimation/
│   ├── config.py            # Params, sensor locations, noise, filter and scoring settings
│   ├── model.py             # FEM fish model, controls, hydrodynamics, simulation
│   ├── measurements.py      # GPS / IMU / bend sensor models
│   ├── spf.py               # sigma-point filter for state + actuator health
│   ├── experiments.py       # batches of runs (parallel), metrics, save/load
│   └── plotting.py          # one plot_<experiment> function per experiment
├── requirements.txt
└── pyproject.toml           # ruff settings
```

## Installation

You need Python 3.10 or newer.

```bash
pip install -r requirements.txt   # numpy, scipy, matplotlib, threadpoolctl
```

## Usage

The scripts import the package as `REACH_Fault_Estimation.fault_estimation`. Run them as modules from the folder
that **contains** this repository:

```bash
cd ..                                            # parent of REACH_Fault_Estimation/
python -m REACH_Fault_Estimation.main
python -m REACH_Fault_Estimation.main_sensor_choice
```

Running `python main.py` from inside the repository fails with `ModuleNotFoundError`.

### Choosing an experiment

To choose an experiment, edit the settings block at the top of `main.py`:

| Setting | Meaning |
|---|---|
| `EXPERIMENT` | Which experiment to run (see the table below) |
| `TF` | Final simulation time in seconds. The fault starts at 1 s, so set `TF > 1` to see it (the paper uses 3–10 s). |
| `SEED` | Random seed. Results do not depend on the number of workers. |
| `N_WORKERS` | Number of parallel processes. `None` uses all cores and `1` runs serially, which helps with debugging. |
| `LOAD_SAVED` | Set to `True` to re-plot `results/<experiment>.npz` without re-simulating. |
| `MSMT_CHOICE`, `SENSOR_PLACEMENT`, `CONTROL_TYPE`, `EK_TRUE2` | Used by `single_run`, `filter_validation`, `animation` and `accuracy_vs_health` |

| `EXPERIMENT` | What it does | Paper |
|---|---|---|
| `actuator_failure` | Each sensor type (GPS, IMU, bend) × each single-actuator failure, with RMS error and rise time | Fig. 5, Fig. 6 |
| `filter_validation` | Windowed normalized innovation squared against the chi-square bounds `b_L` and `b_U`, for each sensor type | Fig. 4 |
| `control_comparison` | Each gait × each single-actuator failure | Sec. V-C |
| `control_repeats` | `control_comparison` repeated 5 times | Fig. 8 |
| `sensor_failure` | Each sensor type with degraded placements, with actuator A3 failed | — |
| `accuracy_vs_health` | Estimation accuracy when one actuator drops to partial health | — |
| `single_run` | One run with the settings above, with 2σ bounds | — |
| `animation` | Animates the estimated fish and the true fish | — |

`main_sensor_choice.py` runs the sensor placement study. It scores 300 filter runs at `TF = 5 s`: IMU and bend
sensors, every one-, two-, three- and four-sensor placement, and each actuator failure. It then plots the
success-score map from paper Fig. 7.

Results are saved as compressed `.npz` files in `results/`.

### Using the package directly

```python
from REACH_Fault_Estimation.fault_estimation import experiments
from REACH_Fault_Estimation.fault_estimation.config import Params, Sensor
from REACH_Fault_Estimation.fault_estimation.model import setup_system
import numpy as np

p = Params(tf=3.0, msmt_choice=Sensor.BEND, sensor_placement=245,
           ek_true2=np.array([1, 1, 0.5, 1, 1]))   # index 0 = tail
result, (Xfem, Xrigid) = experiments.run_case(p, setup_system(p), seed=0)
print(result.ek[:, -1])                              # final health estimates
```

## Conventions

- **Actuator numbering.** In the state vector, `ek` index 0 is the **tail** and index 4 is the **head**. Plots
  follow the paper and label actuators A1 (head) to A5 (tail), so state index `i` appears as `A{5 - i}`.
- **Node numbering.** FEM nodes run from 1 (tail) to 21 (head). Each node has `[x, y, θ]` positions, followed
  by the velocities in the same order.
- **Settings.** Noise values, initial covariances, the sigma-point scaling, the validation window and the
  success thresholds are in `fault_estimation/config.py`. Sensor noise follows the paper's Section IV-A:
  - GPS: 1 m² per axis
  - IMU: 0.0061 m²/s⁴ for acceleration and 0.002 rad²/s² for angular rate
  - Bend sensor: 0.001 rad²

## Scope

This repository contains the simulation studies from Section V of the paper. The experimental validation on the
three-actuator UCSD robot fish (Section VI) used the following, none of which is included here:

- bend sensor data recorded from the physical robot
- a quadratic calibration from sensor output to bend angle
- a 7-segment model adapted to that robot

## Citation

```bibtex
@inproceedings{jiang2025reach,
  title     = {Real-time Estimator of Actuator Control and Health ({REACH}) on an Eel-Inspired Soft Robot},
  author    = {Jiang, Zhangjingyi and Park, Myungsun and Tolley, Michael T. and Campbell, Mark},
  booktitle = {2025 IEEE 8th International Conference on Soft Robotics (RoboSoft)},
  year      = {2025},
  doi       = {10.1109/RoboSoft63089.2025.11020927}
}
```

The robot model and controller come from ASSRSimP:

> Z. Jiang and M. Campbell, "Modeling and control of an eel-inspired soft robot for design optimization,"
> IEEE CASE, 2024.

## Acknowledgments

This work was supported by ONR MURI grant N00014-22-1-2595.
