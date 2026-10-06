"""Run one fault-estimation experiment and plot it.

Choose the experiment with EXPERIMENT. Results are saved to results/<experiment>.npz;
set LOAD_SAVED = True to re-plot a saved run without re-simulating.

Experiments
-----------
actuator_failure     each sensor type x each single-actuator failure (default)
sensor_failure       each sensor type x degraded sensor placements, actuator A3 failed
control_comparison   each control type x each single-actuator failure
control_repeats      control_comparison repeated 5 times
accuracy_vs_health   accuracy when one actuator drops to partial health
single_run           one run with the settings below, with 2-sigma bounds
filter_validation    normalized-innovation check for each sensor type
animation            animate the estimated and true fish for one run
"""

import os

from REACH_Fault_Estimation.fault_estimation import experiments

# One BLAS thread per process: much faster for these small matrices (set before numpy loads).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from REACH_Fault_Estimation.fault_estimation import plotting  # noqa: E402
from REACH_Fault_Estimation.fault_estimation.config import Params, Sensor  # noqa: E402
from REACH_Fault_Estimation.fault_estimation.model import setup_system  # noqa: E402

# --- Settings ----------------------------------------------------------------
EXPERIMENT = "actuator_failure"
TF = 3.0  # final time [s]; the fault is injected at 1 s, so use TF > 1 to see it
SEED = 0
N_WORKERS = None  # parallel processes (None = all cores, 1 = serial)
LOAD_SAVED = False
RESULTS_DIR = Path(__file__).parent / "results"

# Used by single_run / filter_validation / animation / accuracy_vs_health
MSMT_CHOICE = Sensor.IMU  # Sensor.GPS, Sensor.IMU or Sensor.BEND
SENSOR_PLACEMENT = 12345
CONTROL_TYPE = "straight"  # see config.CONTROL_PRESETS
EK_TRUE2 = np.array([1.0, 1, 1, 1, 1])  # actuator health after the fault (index 0 = tail)


def main():
    p = Params(
        tf=TF,
        msmt_choice=MSMT_CHOICE,
        sensor_placement=SENSOR_PLACEMENT,
        control_type=CONTROL_TYPE,
        ek_true2=EK_TRUE2,
    )
    path = RESULTS_DIR / f"{EXPERIMENT}.npz"

    if LOAD_SAVED:
        results = experiments.load_results(path)
    else:
        system = setup_system(p)
        run = getattr(experiments, EXPERIMENT)
        results = run(p, system, seed=SEED, n_workers=N_WORKERS)
        experiments.save_results(path, results)
        print(f"Saved results to {path}")

    getattr(plotting, f"plot_{EXPERIMENT}")(p, results)
    plt.show()


if __name__ == "__main__":
    main()
