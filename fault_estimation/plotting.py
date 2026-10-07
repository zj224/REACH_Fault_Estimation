"""Figures. Each ``plot_<experiment>(p, results)`` draws the figures for one experiment."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .config import FAULT_TIME, PLACEMENT_SETS, SENSOR_LABELS, Sensor
from .experiments import SENSOR_FAILURE_LABELS, failure_health
from .model import fem_to_inertial

# Colour per actuator state index (index 4 = head = "A1").
ACTUATOR_COLORS = ["#77AC30", "#7E2F8E", "#EDB120", "#D95319", "#0072BD"]
CONTROL_LABELS = ["Linear Swimming", r"Low $C_\tau^O$ Turning", r"High $C_\tau^O$ Turning"]
CONTROL_COLORS = ["red", "blue", "#00FF00"]
SCORE_CMAP = ListedColormap([[0.8, 0.8, 0.8], [0.5, 0.7, 0.5], [0.2, 0.8, 0.2], [0, 1, 0]])


def _label(p, i):
    """Display label for actuator state index i (A1 = head)."""
    return f"A{p.nin - i}"


def _true_health(p, ek_true2):
    """True actuator health over time, (nin, nt)."""
    after = p.t > FAULT_TIME
    return np.where(after, np.asarray(ek_true2, float)[:, None], np.asarray(p.ek_true1, float)[:, None])


# --- shared building blocks ----------------------------------------------------------


def _ek_panel(ax, p, ek_hat, failed, xlim):
    """Estimated (solid) and true (thin / dotted for the failed actuator) health."""
    truth = _true_health(p, failure_health(p, failed))
    for i in reversed(range(p.nin)):
        ax.plot(p.t, ek_hat[i], color=ACTUATOR_COLORS[i], linewidth=1, label=_label(p, i))
        style = dict(linestyle=":", linewidth=1) if i == failed else dict(linewidth=0.5)
        ax.plot(p.t, truth[i], color=ACTUATOR_COLORS[i], **style)
    ax.set_xlim(xlim)
    ax.set_ylim([-0.2, 1.5])


def _ek_grid(p, ek_cases, failed, col_titles, row_labels, xlim, fig_num=1):
    """Grid of _ek_panel: ek_cases is (n_cols, n_rows, nin, nt); failed[row] is the failed actuator."""
    n_cols, n_rows = ek_cases.shape[:2]
    fig, axes = plt.subplots(n_rows, n_cols, num=fig_num, clear=True, squeeze=False, sharex=True, sharey=True)
    for c in range(n_cols):
        for r in range(n_rows):
            _ek_panel(axes[r, c], p, ek_cases[c, r], failed[r], xlim)
        axes[0, c].set_title(col_titles[c])
    for r in range(n_rows):
        axes[r, 0].set_ylabel(row_labels[r])
    axes[-1, n_cols // 2].set_xlabel("Time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=p.nin, frameon=False)
    return fig


def _grouped_bars(ax, data, group_labels, bar_labels, ylabel, ylim):
    """Grouped bar chart of data (groups, bars) with value labels ("NaN" for missing)."""
    n_groups, n_bars = data.shape
    width = min(0.8, n_bars / (n_bars + 1.5)) / n_bars
    x = np.arange(n_groups)
    for b in range(n_bars):
        xb = x + (b - (n_bars - 1) / 2) * width
        ax.bar(xb, np.nan_to_num(data[:, b]), width, label=bar_labels[b])
        for xi, v in zip(xb, data[:, b], strict=True):
            ax.text(
                xi,
                0.02 if np.isnan(v) else v,
                "NaN" if np.isnan(v) else f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                clip_on=True,
            )
    ax.set_xticks(x, group_labels)
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel.replace(" (s)", ""))


def _metric_bars(rms, rise, group_labels, bar_labels, fig_num=2, rms_ylim=(0, 0.6), rise_ylim=(0, 0.5)):
    fig, (ax1, ax2) = plt.subplots(2, 1, num=fig_num, clear=True)
    _grouped_bars(ax1, rms, group_labels, bar_labels, "RMS Error", rms_ylim)
    ax1.legend(loc="upper right", ncol=len(bar_labels))
    _grouped_bars(ax2, rise, group_labels, bar_labels, "Rise Time (s)", rise_ylim)
    fig.tight_layout()


# --- experiments ----------------------------------------------------------------------


def plot_single_run(p, r):
    """Estimated actuator health over time, and the same with 2-sigma bounds per actuator."""
    ek_hat = r["xhat"][p.sl_ek]
    ek_var = r["P_diag"][p.sl_ek]
    truth = _true_health(p, p.ek_true2)

    fig, ax = plt.subplots(num=1, clear=True)
    fig.subplots_adjust(bottom=0.35)
    for i in reversed(range(p.nin)):
        name = f"Actuator {p.nin - i}" + (" (Head)" if i == p.nin - 1 else " (Tail)" if i == 0 else "")
        ax.plot(p.t, ek_hat[i], color=ACTUATOR_COLORS[i], linewidth=1, label=f"{name} Predicted")
        ax.plot(p.t, truth[i], color=ACTUATOR_COLORS[i], linewidth=0.5, label=f"{name} True")
    ax.set_xlim([-0.5, p.tf + 0.5])
    ax.set_ylim([-0.1, 1.5])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Actuator Health")
    ax.set_title("Actuator Health Prediction Over Time")
    ax.legend(
        loc="lower left", bbox_to_anchor=(0.02, 0.0), bbox_transform=fig.transFigure, ncol=2, fontsize=8
    )

    def fmt(v):
        return "  ".join(f"{x:.4g}" for x in np.flip(v))

    text = (
        f"True Actuator Health:\n[{fmt(p.ek_true2)}]\nPredicted Actuator Health:\n[{fmt(r['ek_est_mean'])}]"
    )
    fig.text(0.75, 0.13, text, ha="center", va="center", bbox=dict(facecolor="white"))

    fig, axes = plt.subplots(p.nin, 1, num=2, clear=True, sharex=True)
    for row, i in enumerate(reversed(range(p.nin))):
        ax = axes[row]
        bound = 2 * np.sqrt(ek_var[i])
        ax.plot(p.t, ek_hat[i], "-.", color="r", label="Predicted")
        ax.fill_between(
            p.t, ek_hat[i] - bound, ek_hat[i] + bound, color="b", alpha=0.1, label=r"2$\sigma$ Bounds"
        )
        ax.plot(p.t, truth[i], "k:", label="True")
        ax.set_xlim([0, p.tf])
        ax.set_ylim([-0.1, 1.5])
        ax.grid(True)
        ax.set_title(_label(p, i), fontsize=9)
    axes[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axes[-1].set_xlabel("Time (s)")
    axes[p.nin // 2].set_ylabel("Actuator Health")
    fig.suptitle("Actuator Health Prediction Over Time")


def plot_filter_validation(p, r):
    """Windowed normalized innovations against their chi-square bounds, per sensor."""
    fig, axes = plt.subplots(len(Sensor), 1, num=3, clear=True, sharex=True)
    for ax, sensor, lam, lam_w, bl, bu in zip(
        axes, Sensor, r["lam"], r["lam_window"], r["b_lower"], r["b_upper"], strict=True
    ):
        ax.axhline(bu, color="m", linewidth=2, label="$b_U$")
        ax.axhline(bl, color="b", linewidth=2, label="$b_L$")
        ax.plot(p.t, lam, "g", label=r"$\chi^2_{Nnz}$")
        ax.plot(p.t, lam_w, "r", label=r"$\lambda^{KF}_k(N)$")
        ax.set_ylim([0, 15])
        ax.set_ylabel(SENSOR_LABELS[sensor])
    axes[0].legend(ncol=4)
    axes[0].set_title("Filter Validation")
    axes[-1].set_xlabel("Time (s)")


def plot_actuator_failure(p, r):
    """Health estimates per sensor (columns) and failed actuator (rows, A1..A5), plus metrics."""
    sensor_labels = [SENSOR_LABELS[s] for s in Sensor]
    failed = [p.nin - 1 - row for row in range(p.nin)]  # row 0 = A1 = head
    _ek_grid(p, r["ek_hat"][:, ::-1], failed, sensor_labels, [f"A{row + 1}" for row in range(p.nin)], (0, 3))
    labels = [f"A{a + 1}" for a in range(p.nin)]
    _metric_bars(r["rms"][:, ::-1], r["rise_time"][:, ::-1], sensor_labels, labels)


def plot_sensor_failure(p, r):
    """Health estimates per sensor (columns) and degraded placement (rows), plus metrics."""
    sensor_labels = [SENSOR_LABELS[s] for s in Sensor]
    failed = [int(r["failed_actuator"])] * len(SENSOR_FAILURE_LABELS)
    _ek_grid(p, r["ek_hat"], failed, sensor_labels, SENSOR_FAILURE_LABELS, (0, p.tf))
    _metric_bars(r["rms"], r["rise_time"], sensor_labels, SENSOR_FAILURE_LABELS, rms_ylim=(0, 0.5))


def plot_control_comparison(p, r):
    """Health estimates per control type and failed actuator, plus metrics vs actuator."""
    failed = [p.nin - 1 - row for row in range(p.nin)]
    _ek_grid(
        p, r["ek_hat"][:, ::-1], failed, CONTROL_LABELS, [f"A{row + 1}" for row in range(p.nin)], (0, p.tf)
    )
    x = np.arange(1, p.nin + 1)
    for fig_num, key, ylabel in [(2, "rms", "RMS Error"), (3, "rise_time", "Rise Time (s)")]:
        fig, ax = plt.subplots(num=fig_num, clear=True)
        for c, label in enumerate(CONTROL_LABELS):
            ax.plot(x, r[key][c, ::-1, 0], color=CONTROL_COLORS[c], linewidth=1, label=label)
        ax.set_xticks(x, [f"A{a}" for a in x])
        ax.set_ylabel(ylabel)
        ax.legend()


def plot_control_repeats(p, r):
    """Every repeat (dots) and the mean (line) per control type and actuator."""
    x = np.arange(1, p.nin + 1)
    n_rep = r["rms"].shape[-1]
    offsets = np.linspace(-0.1, 0.1, n_rep)
    for fig_num, key, ylabel, ylim in [
        (2, "rise_time", "Rise Time (s)", (0, 0.5)),
        (3, "rms", "RMS Error", (0, 0.2)),
    ]:
        fig, ax = plt.subplots(num=fig_num, clear=True)
        for c, label in enumerate(CONTROL_LABELS):
            vals = r[key][c, ::-1]  # (actuator A1..A5, repeat)
            ax.scatter((x[:, None] + offsets).ravel(), vals.ravel(), color=CONTROL_COLORS[c], label=label)
            ax.plot(x, np.nanmean(vals, axis=1), color=CONTROL_COLORS[c], label=f"{label} Mean")
        ax.set_xticks(x, [f"A{a}" for a in x])
        ax.set_xlabel("Actuator")
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        ax.set_title(ylabel.replace(" (s)", ""))
        ax.legend()


def plot_accuracy_vs_health(p, r):
    fig, ax = plt.subplots(num=3, clear=True)
    for acc in r["accuracy"]:
        ax.scatter(100 * r["health"], 100 * acc, color="k")
    ax.set_xlabel("Actuator Functionality %")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy of Fault Estimation Algorithm at Different Actuator Functionalities")
    ax.set_xlim([-5, 105])
    ax.set_ylim([90, 100])


def plot_sensor_placement(p, r):
    """Success score (0-3) for each placement (columns) and failed actuator (rows)."""
    titles = ["One Sensor", "Two Sensors", "Three Sensors", "Four Sensors"]
    widths = [len(s) for s in PLACEMENT_SETS]
    sensors = [Sensor.IMU, Sensor.BEND]
    fig, axes = plt.subplots(
        len(sensors),
        len(PLACEMENT_SETS),
        num=3,
        clear=True,
        figsize=(16, 7),
        gridspec_kw={"width_ratios": widths},
    )
    for i, sensor in enumerate(sensors):
        for j, places in enumerate(PLACEMENT_SETS):
            ax = axes[i, j]
            M = r["score"][i, j, : len(places), ::-1].T  # rows A1..A5, columns placements
            im = ax.imshow(M, cmap=SCORE_CMAP, vmin=-0.5, vmax=3.5, aspect="auto")
            ax.set_xticks(np.arange(M.shape[1]), [f"S{pl}" for pl in places], rotation=45)
            ax.set_yticks(np.arange(M.shape[0]), [f"A{a + 1}" for a in range(M.shape[0])])
            ax.set_xticks(np.arange(M.shape[1] + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(M.shape[0] + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="k", linewidth=0.5)
            ax.tick_params(which="minor", length=0)
            if i == 0:
                ax.set_title(titles[j], fontweight="bold", fontsize=20)
        axes[i, 0].set_ylabel(SENSOR_LABELS[sensor], fontweight="bold", fontsize=20)
    fig.colorbar(im, ax=axes[0, -1], ticks=[0, 1, 2, 3])
    fig.tight_layout()


def fish_animation(p, Xinertial, fig_num=1):
    """Animate the fish body and the head's path."""
    x = p.Hx @ Xinertial
    y = p.Hy @ Xinertial
    fig, ax = plt.subplots(num=fig_num)
    ax.set_aspect("equal")
    x_lo, x_hi = x.min() - 0.1 * abs(x.min()), x.max() + 0.1 * abs(x.max())
    y_lo, y_hi = y.min() - 0.1 * abs(y.min()), y.max() + 0.1 * abs(y.max())
    ax.axis([x_lo - 1, x_hi + 1, y_lo - 1, y_hi + 1])
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    (body,) = ax.plot(x[:, 0], y[:, 0], "b-", linewidth=5)
    (trail,) = ax.plot(x[0, :1], y[0, :1], "b-")
    for k in range(1, p.nt):
        body.set_data(x[:, k], y[:, k])
        trail.set_data(x[0, :k], y[0, :k])
        plt.pause(p.t_pause)


def plot_animation(p, r):
    """Animate the estimated fish, then the true one."""
    fish_animation(p, fem_to_inertial(r["xhat"][: p.ns], p, r["xhat"][p.sl_rigid]), fig_num=1)
    fish_animation(p, fem_to_inertial(r["Xfem_true"], p, r["Xrigid_true"]), fig_num=2)


EXP_COLORS = ["r", "g", "b"]  # A1, A2, A3 as in paper Fig. 9


def plot_experimental_validation(p, r):
    """Estimated (solid) and true (dotted) health of A1-A3 for each recorded trial."""
    n = len(r["health"])
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, num=1, clear=True, squeeze=False, sharex=True, sharey=True)
    for ax, health, ek_hat in zip(axes.flat, r["health"], r["ek_hat"], strict=False):
        for a, color in enumerate(EXP_COLORS):  # a = 0 is A1, stored last in ek_hat
            ax.plot(p.t, ek_hat[-1 - a], color=color, linewidth=1.5, label=f"A{a + 1}")
            ax.axhline(health[a], color=color, linestyle=":", linewidth=1)
        ax.set_title("[" + " ".join(f"{h:g}" for h in health) + "]")
        ax.set_xlim([-0.5, p.tf + 0.5])
        ax.set_ylim([-0.1, 1.5])
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    axes[n_rows // 2, 0].set_ylabel("Actuator Health")
    axes[-1, n_cols // 2].set_xlabel("Time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
