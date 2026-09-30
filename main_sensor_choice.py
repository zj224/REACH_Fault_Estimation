"""Sensor placement study: score every IMU / bend-sensor placement against each
actuator failure and plot the success map.

300 filter runs at TF = 5 s; they run in parallel on all cores. Results are saved to
results/sensor_placement.npz; set LOAD_SAVED = True to re-plot without re-running.
"""

import os

from REACH_Fault_Estimation.fault_estimation import experiments

# One BLAS thread per process: much faster for these small matrices (set before numpy loads).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from REACH_Fault_Estimation.fault_estimation import plotting  # noqa: E402
from REACH_Fault_Estimation.fault_estimation.config import Params  # noqa: E402
from REACH_Fault_Estimation.fault_estimation.model import setup_system  # noqa: E402

# --- Settings ----------------------------------------------------------------
TF = 5.0  # final time [s]
SEED = 0
N_WORKERS = None  # parallel processes (None = all cores, 1 = serial)
LOAD_SAVED = False
RESULTS_PATH = Path(__file__).parent / "results" / "sensor_placement.npz"


def main():
    p = Params(tf=TF)

    if LOAD_SAVED:
        results = experiments.load_results(RESULTS_PATH)
    else:
        results = experiments.sensor_placement(p, setup_system(p), seed=SEED, n_workers=N_WORKERS)
        experiments.save_results(RESULTS_PATH, results)
        print(f"Saved results to {RESULTS_PATH}")

    plotting.plot_sensor_placement(p, results)
    plt.show()


if __name__ == "__main__":
    main()
