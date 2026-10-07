# REACH: Real-time Estimator of Actuator Control and Health

Code for the paper **"Real-time Estimator of Actuator Control and Health (REACH) on an Eel-Inspired Soft Robot"**
by Zhangjingyi Jiang, Myungsun Park, Michael T. Tolley and Mark Campbell.

Published at IEEE RoboSoft 2025: [doi:10.1109/RoboSoft63089.2025.11020927](https://doi.org/10.1109/RoboSoft63089.2025.11020927)
· arXiv: <https://arxiv.org/html/2608.14865v1>

**Reach** estimates actuator health for a soft robot that swims like an eel (anguilliform swimming) using IMU or bend sensor measurement. A normalized-innovation chi-square test checks that the filter is statistically consistent.

**Actuator Health** is the ratio of the torque an actuator actually produces to the torque it was asked to produce:

- `0` means the actuator has failed completely
- `1` means it works fully
- values above `1` mean it is over-actuating


## How it works

### Robot model (ASSRSimP)

The fish is an elastic beam with no fixed ends, split into finite elements, with torque actuators spread evenly along its length. The model includes quadratic hydrodynamic drag, and the rigid-body pose is tracked apart from the body's deformation (`model.py`).

### Control

Each actuator receives a sinusoidal torque:

$$
u_{1:n_\tau} = \left[ C_A \cdot A_{1:n_\tau} \right] \sin\left( C_\omega \cdot \omega t + \phi_{1:n_\tau} \right) + C_\tau^O \cdot \tau^O_{1:n_\tau}
$$

where $n_\tau$ is the number of actuators, $A$ the amplitudes, $\phi$ the phase shifts, $\tau^O$ the torque offsets and $\omega$ the oscillation frequency. The coefficients $C_A$ and $C_\omega$ control linear swimming and $C_\tau^O$ controls turning. Three presets set the gait: linear swimming, wide turning and tight turning (`config.CONTROL_PRESETS`).

### Fault injection

The true actuator health is `ek_true1` until `FAULT_TIME = 1 s` and `ek_true2` after. Each actuator's torque is multiplied by its health $H_i$, and the health vector is appended to the state:

$$
u_i^{REACH} = H_i \cdot u_i
$$

$$
X^{REACH} = \begin{bmatrix} X \cr H \end{bmatrix}
$$

### Sensors

The sensor models are GPS (2D position), IMU (acceleration and angular rate) and bend sensors (the bending angle of each actuator). A sensor placement such as `135` means sensors at positions 1, 3 and 5, where position 1 is nearest the head (`measurements.py`).

### Estimator

A sigma-point (unscented) Kalman filter estimates the state: FEM states, the 3-element rigid-body pose and each actuator health (`spf.py`). At each step, sigma points are drawn around the estimate:

$$
S_{k|k} = \mathrm{chol}(P_{k|k}), \qquad \chi^0_{k|k} = \hat{x}_{k|k}, \qquad \chi^{1:2n}_{k|k} = \hat{x}_{k|k} \cdot 1_n \pm n_\sigma S_{k|k}
$$

They are then passed through the nonlinear dynamics $f$ and the measurement model $h$:

$$
\chi^i_{k+1|k} = f\left( \chi^i_{k|k} \right), \qquad \mathcal{Z}^{i}_{k+1|k} = h(\chi^{i}_{k+1|k})
$$

### Filter validation

The filter averages the normalized innovation squared over a window of $N$ steps:

$$
\hat{z}_k = \sum_{i=0}^{2n} w_m^i \mathcal{Z}^i_{k|k-1}, \qquad v_k = z_k - \hat{z}_k
$$

$$
\chi^2_{Nnz} = \sum_{k-N}^{k} v_{k}' \ast inv(S_{k|k}) \ast v_{k}, \qquad \lambda^{KF}_k(N) = \frac{\chi^2_{Nnz}}{N}
$$

The filter is consistent when $b_L < \lambda^{KF}_k(N) < b_U$, with the chi-square bounds

$$
b_L = \frac{Inv\lbrace\chi^2_{Nnz}\rbrace (\frac{\alpha}{2})}{N}, \ b_U = \frac{Inv\lbrace\chi^2_{Nnz}\rbrace (1-\frac{\alpha}{2})}{N}
$$

where $\alpha$ is the false positive rate (5% here, `VALIDATION_GATE = 0.95`).

### Scoring

Each run is scored on two metrics:

- **Rise time:** how long after the fault the failed actuator's estimate takes to drop below 0.1.
- **RMS error:** the error in the health estimates over the last 100 steps:

$$
RMS = \sqrt{ \frac{1}{n_k n_\tau} \sum_{k=1}^{n_k} \sum_{i=1}^{n_\tau} \left( \hat{H}_i(t_k) - H_i(t_k) \right)^2 }
$$

The sensor placement study combines the two into a success score from 0 to 3 (`experiments.py`).

### Experimental validation

REACH is also run on bend sensor data recorded from the three-actuator UCSD robot fish (`experimental.py`). The model is changed to match the robot: 28 elements in 7 sections (head, A1, coupling, A2, coupling, A3, tail), with the section lengths from paper Table I. The Young's modulus of each actuator is tuned to the full-health trial (A1, A2, A3 = 1.4096, 0.9121, 0.9934 MPa), the head and couplings are rigid plastic (2 GPa), and the tail is soft silicone (1 MPa).

In the video calibration, each actuator's bend angle is measured from the markers on either side of it:

$$
v_i = m^L_i - m^R_i, \qquad v_{i+1} = m^L_{i+1} - m^R_{i+1}
$$

$$
\theta_i = \mathrm{atan2}\left( v_i \times v_{i+1}, \ v_i \cdot v_{i+1} \right)
$$

A quadratic maps the normalized sensor output $x_i$ to that angle:

$$
\theta_i = C_1 x_i^2 + C_2 x_i + C_3, \qquad x_i = \frac{\text{raw}_i - 20000}{20000}
$$

The coefficients are in `config.BEND_CALIBRATION`. The filter uses the same angle definition between model nodes as its measurement model. Each trial's angles have their mean removed. The data is recorded at 12 Hz and the filter runs at 100 Hz, so the filter only updates on steps that have a measurement.

## Repository layout

```
REACH_Fault_Estimation/
├── main.py                  # run one experiment (choose it with EXPERIMENT) and plot it
├── main_sensor_choice.py    # sensor quantity / placement study (paper Fig. 7)
├── main_experimental.py     # REACH on the recorded robot data (paper Fig. 9)
├── data/experimental/       # bend sensor recordings, one CSV per actuator health trial
├── fault_estimation/
│   ├── config.py            # Params, sensor locations, noise, filter and scoring settings
│   ├── model.py             # FEM fish model, controls, hydrodynamics, simulation
│   ├── measurements.py      # GPS / IMU / bend sensor models
│   ├── spf.py               # sigma-point filter for state + actuator health
│   ├── experiments.py       # batches of runs (parallel), metrics, save/load
│   ├── experimental.py      # load recorded data, calibrate, run REACH on each trial
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

The scripts import the package as `REACH_Fault_Estimation.fault_estimation`. Run them as modules from the folder that **contains** this repository:

```bash
cd ..                                            # parent of REACH_Fault_Estimation/
python -m REACH_Fault_Estimation.main
python -m REACH_Fault_Estimation.main_sensor_choice
python -m REACH_Fault_Estimation.main_experimental
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
| `filter_validation` | Windowed normalized innovation squared against the chi-square bounds $b_L$ and $b_U$, for each sensor type | Fig. 4 |
| `control_comparison` | Each gait × each single-actuator failure | Sec. V-C |
| `control_repeats` | `control_comparison` repeated 5 times | Fig. 8 |
| `sensor_failure` | Each sensor type with degraded placements, with actuator A3 failed | — |
| `accuracy_vs_health` | Estimation accuracy when one actuator drops to partial health | — |
| `single_run` | One run with the settings above, with 2σ bounds | — |
| `animation` | Animates the estimated fish and the true fish | — |

`main_sensor_choice.py` runs the sensor placement study. It scores 300 filter runs at `TF = 5 s`: IMU and bend
sensors, every one-, two-, three- and four-sensor placement, and each actuator failure. It then plots the
success-score map from paper Fig. 7.

`main_experimental.py` runs REACH on the nine recorded trials from paper Fig. 9 and plots the estimated health of A1–A3 against the true values. Change `TRIALS` to choose trials; `(0.9, 1, 1)` is also recorded. The nine trials take about a minute on all cores.

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

## Experimental data

`data/experimental/` holds one CSV per trial, named by the true health of A1 (head), A2 and A3 in percent. Each file has no header and 500 rows at 12 Hz, with these columns:

| Column | Contents |
|---|---|
| 1 | time (s) |
| 2–4 | pump input for A1, A2, A3 |
| 5–7 | raw bend sensor output for A1, A2, A3 |

Full-capacity pump inputs were 575, 345 and 115 ml/min for A1, A2 and A3; a degraded actuator's input is scaled by its health. The original file names give the input setting per actuator and the recording time (2024-09-08).

| File | Health [A1 A2 A3] | Original file |
|---|---|---|
| `health_100_100_100.csv` | [1 1 1] | `50_30_10_240908_1248_18.csv` |
| `health_050_100_100.csv` | [0.5 1 1] | `25_30_10_240908_1327_14.csv` |
| `health_000_100_100.csv` | [0 1 1] | `00_30_10_240908_1328_30.csv` |
| `health_100_000_000.csv` | [1 0 0] | `50_00_00_240908_1339_25.csv` |
| `health_000_100_000.csv` | [0 1 0] | `00_30_00_240908_1336_09.csv` |
| `health_000_000_100.csv` | [0 0 1] | `00_00_10_240908_1308_12.csv` |
| `health_080_100_100.csv` | [0.8 1 1] | `40_30_10_240908_1325_44.csv` |
| `health_080_050_100.csv` | [0.8 0.5 1] | `40_15_10_240908_1333_20.csv` |
| `health_050_080_100.csv` | [0.5 0.8 1] | `25_24_10_240908_1331_50.csv` |
| `health_090_100_100.csv` | [0.9 1 1] | `45_30_10_240908_1323_45.csv` (not in the paper) |

## Conventions

- **Actuator numbering.** In the state vector, `ek` index 0 is the **tail** and index 4 is the **head**. Plots
  follow the paper and label actuators A1 (head) to A5 (tail), so state index `i` appears as `A{5 - i}`.
- **Experiment numbering.** For the robot fish, health vectors in the filter are ordered A3, A2, A1 (tail end first), like the simulation. Trial names, `TRIALS` and the plots use [A1, A2, A3].
- **Node numbering.** FEM nodes run from 1 (tail) to 21 (head), or 29 (head) for the robot fish model. Each node has `[x, y, θ]` positions, followed
  by the velocities in the same order.
- **Settings.** Noise values, initial covariances, the sigma-point scaling, the validation window and the
  success thresholds are in `fault_estimation/config.py`. Sensor noise follows the paper's Section IV-A:
  - GPS: 1 m² per axis
  - IMU: 0.0061 m²/s⁴ for acceleration and 0.002 rad²/s² for angular rate
  - Bend sensor: 0.001 rad²

## Scope

This repository contains the simulation studies from Section V of the paper and the experimental validation from Section VI. It does not include the video processing used to fit the bend sensor calibration; only the resulting coefficients are included.

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
