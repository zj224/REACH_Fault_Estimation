"""Finite-element fish model: system matrices, controls, hydrodynamics and simulation."""

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from scipy import signal

from .config import CONTROL_PRESETS


@dataclass
class DiscreteSystem:
    """Discrete-time local FEM dynamics: x+ = Fd x + Gd u + Ghd f_hydro."""

    Fd: np.ndarray
    Gd: np.ndarray
    Ghd: np.ndarray


def element_matrices(rho, l, E, r):
    """Mass and stiffness matrices of one 2D beam element (see Crawley, Campbell)."""
    A = np.pi * r**2  # cross-sectional area, circular rod
    I = 1 / 4 * np.pi * r**4  # area moment of inertia
    mi = (rho * A * l / 420) * np.array(
        [
            [140, 0, 0, 70, 0, 0],
            [0, 156, 22 * l, 0, 54, -13 * l],
            [0, 22 * l, 4 * l**2, 0, 13 * l, -3 * l**2],
            [70, 0, 0, 140, 0, 0],
            [0, 54, 13 * l, 0, 156, -22 * l],
            [0, -13 * l, -3 * l**2, 0, -22 * l, 4 * l**2],
        ]
    )
    ki = (E * I / l**3) * np.array(
        [
            [l**2 * A / I, 0, 0, -(l**2) * A / I, 0, 0],
            [0, 12, 6 * l, 0, -12, 6 * l],
            [0, 6 * l, 4 * l**2, 0, -6 * l, 2 * l**2],
            [-(l**2) * A / I, 0, 0, l**2 * A / I, 0, 0],
            [0, -12, -6 * l, 0, 12, -6 * l],
            [0, 6 * l, 2 * l**2, 0, -6 * l, 4 * l**2],
        ]
    )
    return mi, ki


def setup_system(p):
    """Build the free-free beam model and discretize it with step ``p.dt``."""
    rho = 1200  # kg/m^3, approximate oar fish
    ni = 3  # DOF per node

    # Assemble global mass and stiffness (no boundary conditions: free-free beam)
    M = np.zeros((p.ndof, p.ndof))
    K = np.zeros((p.ndof, p.ndof))
    for i in range(p.n):
        mi, ki = element_matrices(rho, p.l_elem[i], p.E_elem[i], p.rvec[i])
        s = slice(ni * i, ni * (i + 2))
        K[s, s] += ki
        M[s, s] += mi

    # Modal decomposition; replace the three rigid-body modes with x, y, th motion
    lam, Phi = scipy.linalg.eigh(K, M)
    lam[0:3] = 0  # remove numerical noise on the zero eigenvalues
    ww = np.sqrt(lam)  # natural frequencies
    Phi2 = Phi.copy()
    Phi2[:, 0:3] = 0
    nodes = np.arange(p.nn) * ni
    Phi2[nodes, 0] = 1  # pure x translation
    Phi2[nodes + 1, 1] = 1  # pure y translation
    th3 = 0.01  # small rotation about the center node
    Phi2[nodes, 2] = p.xnode * (np.cos(th3) - 1)
    Phi2[nodes + 1, 2] = p.xnode * np.sin(th3)
    Phi2[nodes + 2, 2] = th3
    Mrigid = Phi2.T @ M @ Phi2
    for j in range(3):  # mass-normalize the rigid modes
        Phi2[:, j] /= np.sqrt(Mrigid[j, j])

    # Back to x, y, th coordinates with modal damping zeta = 0.7
    Phi_inv = np.linalg.inv(Phi2)
    Phi_invT = Phi_inv.T
    M = Phi_invT @ np.eye(p.ndof) @ Phi_inv
    K = Phi_invT @ np.diag(lam) @ Phi_inv
    C = Phi_invT @ (2 * 0.7 * np.diag(ww)) @ Phi_inv

    # Actuators: a torque couple over each of the nin body sections
    nsec = p.n // p.nin  # nodes per section
    Bin = np.zeros((p.ndof, p.nin))
    for i in range(p.nin):
        th_idx = (i * nsec * ni) + np.arange(1, nsec + 1) * ni - 1
        half = nsec // 2
        Bin[th_idx[:half], i] = 1
        Bin[th_idx[(nsec + 1) // 2 :], i] = -1

    # Hydrodynamic forces: x, y force at each node
    Bh = np.zeros((p.ndof, 2 * p.nn))
    for i in range(p.nn):
        Bh[i * ni, i] = 1
        Bh[i * ni + 1, p.nn + i] = 1

    Minv = np.linalg.inv(M)
    Z = np.zeros((p.ndof, p.ndof))
    F = np.block([[Z, np.eye(p.ndof)], [-Minv @ K, -Minv @ C]])
    G = np.vstack([np.zeros((p.ndof, p.nin)), Minv @ Bin])
    Gh = np.vstack([np.zeros((p.ndof, 2 * p.nn)), Minv @ Bh])
    B = np.hstack([G, Gh])
    sys_d = signal.StateSpace(F, B, np.eye(p.ns), np.zeros((p.ns, B.shape[1]))).to_discrete(p.dt)
    return DiscreteSystem(Fd=sys_d.A, Gd=sys_d.B[:, : p.nin], Ghd=sys_d.B[:, p.nin :])


def get_controls(p, k):
    """Actuator torques at time step k, shape (nin, 1)."""
    c_amp, c_w, c_off = CONTROL_PRESETS[p.control_type]
    phase = 1 / 1.33 * np.linspace(1 / p.nin, 1, p.nin)
    uk = c_amp * p.A * p.amp_co * np.sin(2 * np.pi * (phase + c_w * p.w * p.t[k])) + c_off * p.offset_co
    return uk[:, None]


def hydro_forces(th, vx, vy, p):
    """Quadratic drag perpendicular to each link plus head drag.

    Inputs are (nn, N) arrays of node angles and local velocities for N states;
    returns (2 nn, N): x forces for all nodes, then y forces.
    """
    uw = p.uwvec[:, None]
    ex, ey = -np.sin(th), np.cos(th)  # link normal
    v_perp = vx * ex + vy * ey
    mag = -np.sign(v_perp) * v_perp**2
    fx = uw * mag * ex
    fy = uw * mag * ey

    # head tip drag along the body axis
    er_x, er_y = np.cos(th[-1]), np.sin(th[-1])
    v_head = vx[-1] * er_x + vy[-1] * er_y
    a_head = 0.25 * 3.14 * p.rvec[-1] ** 2
    f_head = 0.5 * 0.167 * 1000 * a_head * (-np.sign(v_head) * v_head**2)
    fx[-1] += f_head * er_x
    fy[-1] += f_head * er_y
    return np.vstack([fx, fy])


def remove_rigid_motion(Xfem, Xrigid_prev, p):
    """Split the mean body motion out of propagated FEM states.

    ``Xfem`` is (ns, N) after one dynamics step, ``Xrigid_prev`` is (3, N).
    Returns the FEM states in the new local frame and the updated rigid-body pose.
    """
    dx = Xfem[p.ix].mean(axis=0)
    dy = Xfem[p.iy].mean(axis=0)
    dth = Xfem[p.ith].mean(axis=0)

    # rigid-body update in the inertial frame
    thk = Xrigid_prev[2] + dth
    Xrigid = np.vstack(
        [
            Xrigid_prev[0] + dx * np.cos(thk) - dy * np.sin(thk),
            Xrigid_prev[1] + dx * np.sin(thk) + dy * np.cos(thk),
            Xrigid_prev[2] + dth,
        ]
    )

    # subtract the rigid motion, then rotate node positions back by -dth
    x = Xfem[p.ix] - dx + p.xnode[:, None]
    y = Xfem[p.iy] - dy
    c, s = np.cos(dth), np.sin(dth)
    out = Xfem.copy()
    out[p.ix] = x * c + y * s - p.xnode[:, None]
    out[p.iy] = -x * s + y * c
    out[p.ith] = Xfem[p.ith] - dth
    return out, Xrigid


def step_dynamics(Xfem, Xrigid, uk, p, system):
    """Propagate FEM (ns, N) and rigid (3, N) states one step with inputs ``uk``."""
    vel = Xfem[p.ndof :]
    f = hydro_forces(Xfem[p.ith], vel[p.ix], vel[p.iy], p)
    Xfem_next = system.Fd @ Xfem + system.Ghd @ f + system.Gd @ uk
    return remove_rigid_motion(Xfem_next, Xrigid, p)


def simulate(p, system):
    """Simulate the true fish; actuators follow ek_true1 until FAULT_TIME, then ek_true2.

    Returns FEM states (ns, nt) and rigid-body pose (3, nt), both starting at rest.
    """
    Xfem = np.zeros((p.ns, p.nt))
    Xrigid = np.zeros((3, p.nt))
    for k in range(1, p.nt):
        ek = p.ek_true1 if k <= p.k_fault else p.ek_true2
        uk = p.input_gains(ek) * get_controls(p, k)
        Xfem[:, [k]], Xrigid[:, [k]] = step_dynamics(Xfem[:, [k - 1]], Xrigid[:, [k - 1]], uk, p, system)
    return Xfem, Xrigid


def fem_to_inertial(Xfem, p, Xrigid):
    """Node positions in the inertial frame (velocity rows are left at zero)."""
    x = Xfem[p.ix] + p.xnode[:, None]
    y = Xfem[p.iy]
    xk, yk, thk = Xrigid[0], Xrigid[1], Xrigid[2]
    Xinertial = np.zeros_like(Xfem)
    Xinertial[p.ix] = x * np.cos(thk) - y * np.sin(thk) + xk
    Xinertial[p.iy] = x * np.sin(thk) + y * np.cos(thk) + yk
    Xinertial[p.ith] = Xfem[p.ith] + thk
    return Xinertial
