"""Experimental validation (paper Sec. VI, Fig. 9): run REACH on bend sensor data recorded
from the three-actuator UCSD robot fish, for each actuator health trial.

Data: data/experimental/health_<A1>_<A2>_<A3>.csv. Results are saved to
results/experimental_validation.npz; set LOAD_SAVED = True to re-plot without re-running.
"""

import os

from REACH_Fault_Estimation.fault_estimation import experimental

# One BLAS thread per process: much faster for these small matrices (set before numpy loads).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from REACH_Fault_Estimation.fault_estimation import experiments, plotting  # noqa: E402
from REACH_Fault_Estimation.fault_estimation.config import ExperimentParams  # noqa: E402

# --- Settings ----------------------------------------------------------------
TRIALS = experimental.PAPER_TRIALS  # health of [A1, A2, A3]; (0.9, 1, 1) is also recorded
N_WORKERS = None  # parallel processes (None = all cores, 1 = serial)
LOAD_SAVED = False
RESULTS_PATH = Path(__file__).parent / "results" / "experimental_validation.npz"


def main():
    p = ExperimentParams()

    if LOAD_SAVED:
        results = experiments.load_results(RESULTS_PATH)
    else:
        results = experimental.experimental_validation(TRIALS, p, n_workers=N_WORKERS)
        experiments.save_results(RESULTS_PATH, results)
        print(f"Saved results to {RESULTS_PATH}")

    plotting.plot_experimental_validation(p, results)
    plt.show()


if __name__ == "__main__":
    main()
