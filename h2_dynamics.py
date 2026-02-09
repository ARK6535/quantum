"""Molecular dynamics simulation helpers for H2.

Provides Velocity-Verlet integrators, unit-conversion wrappers, and
parallel-execution utilities used by the main demo script.
"""
from __future__ import annotations

import csv
import logging
import math
import os
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from h2_energy_statevector import compute_h2_energy_quantum_statevector
from h2_helpers import (
    A0,
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    MU,
    T_AU_FS,
    compute_h2_energy_classical,
)

__all__ = [
    "chunk_list",
    "dynamics_seq",
    "energy_classical_bohr",
    "energy_quantum_bohr",
    "energy_worker",
    "force_classical_angstrom",
    "force_classical_bohr",
    "force_quantum_angstrom",
    "force_quantum_bohr",
    "simulate_h2_1d",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unit-conversion helpers
# ---------------------------------------------------------------------------

def _fs_to_au(t_fs: float) -> float:
    return t_fs / T_AU_FS


def _au_to_fs(t_au: float) -> float:
    return t_au * T_AU_FS


def _v_angfs_to_au(v_ang_per_fs: float) -> float:
    return v_ang_per_fs * (T_AU_FS / A0)


def _v_au_to_angfs(v_au: float | np.ndarray) -> float | np.ndarray:
    return v_au * (A0 / T_AU_FS)


# ---------------------------------------------------------------------------
# Energy / force wrappers (Bohr / Ångström interfaces)
# ---------------------------------------------------------------------------

def energy_classical_bohr(r_bohr: float) -> float:
    """Return the classical Full-CI energy at *r_bohr* (Bohr) in Hartree.

    Args:
        r_bohr: Internuclear distance in Bohr.

    Returns:
        Total electronic energy in Hartree.
    """
    r_angstrom = r_bohr * BOHR_TO_ANGSTROM
    return compute_h2_energy_classical(r_angstrom, basis="sto-3g")


def energy_quantum_bohr(
    r_bohr: float,
    timestamp: str,
    backend: Any | None = None,
) -> float:
    """Return the statevector-VQE energy at *r_bohr* (Bohr) in Hartree.

    Args:
        r_bohr: Internuclear distance in Bohr.
        timestamp: Log-directory timestamp string (YYMMDDHHmm).
        backend: Optional pre-initialised backend (unused for statevector).

    Returns:
        Minimum VQE energy in Hartree.
    """
    r_angstrom = r_bohr * BOHR_TO_ANGSTROM
    val = compute_h2_energy_quantum_statevector(
        r_angstrom,
        basis="sto-3g",
        timestamp=timestamp,
        ansatz_reps=1,
        optimizer_maxiter=2000,
        cholesky_tol=1e-10,
    )
    if isinstance(val, tuple):
        return val[0]
    return val


def force_quantum_bohr(
    r_bohr: float,
    timestamp: str,
    h_bohr: float = 0.01,
    backend: Any | None = None,
) -> float:
    """Return the VQE force at *r_bohr* via central finite difference.

    Args:
        r_bohr: Internuclear distance in Bohr.
        timestamp: Log-directory timestamp string.
        h_bohr: Finite-difference step in Bohr.
        backend: Optional pre-initialised backend.

    Returns:
        Force in Hartree/Bohr (negative gradient).
    """
    e_plus = energy_quantum_bohr(r_bohr + h_bohr, backend=backend, timestamp=timestamp)
    e_minus = energy_quantum_bohr(r_bohr - h_bohr, backend=backend, timestamp=timestamp)
    return -(e_plus - e_minus) / (2.0 * h_bohr)


def force_quantum_angstrom(
    r_angstrom: float,
    timestamp: str,
    backend: Any | None = None,
) -> float:
    """Return the VQE force at *r_angstrom* (Å) in Ha/Å.

    Args:
        r_angstrom: Internuclear distance in Ångströms.
        timestamp: Log-directory timestamp string.
        backend: Optional pre-initialised backend.

    Returns:
        Force in Hartree/Ångström.
    """
    r_bohr = r_angstrom * ANGSTROM_TO_BOHR
    f_bohr = force_quantum_bohr(r_bohr, backend=backend, timestamp=timestamp)
    return f_bohr * ANGSTROM_TO_BOHR


def force_classical_bohr(r_bohr: float, h_bohr: float = 0.01) -> float:
    """Return the classical Full-CI force at *r_bohr* via central finite difference.

    Args:
        r_bohr: Internuclear distance in Bohr.
        h_bohr: Finite-difference step in Bohr.

    Returns:
        Force in Hartree/Bohr (negative gradient).
    """
    e_plus = energy_classical_bohr(r_bohr + h_bohr)
    e_minus = energy_classical_bohr(r_bohr - h_bohr)
    return -(e_plus - e_minus) / (2.0 * h_bohr)


def force_classical_angstrom(r_angstrom: float) -> float:
    """Return the classical Full-CI force at *r_angstrom* (Å) in Ha/Å.

    Args:
        r_angstrom: Internuclear distance in Ångströms.

    Returns:
        Force in Hartree/Ångström.
    """
    r_bohr = r_angstrom * ANGSTROM_TO_BOHR
    f_bohr = force_classical_bohr(r_bohr)
    return f_bohr * ANGSTROM_TO_BOHR


# ---------------------------------------------------------------------------
# Velocity-Verlet integrator
# ---------------------------------------------------------------------------

def simulate_h2_1d(
    r0_angstrom: float,
    v_rel0_ang_per_fs: float,
    t_final_fs: float,
    dt_fs: float,
    force_func: Callable[[float], float] = force_classical_bohr,
) -> tuple[np.ndarray, np.ndarray, float | np.ndarray]:
    """Simulate the 1-D relative motion of H2 with Velocity Verlet.

    Args:
        r0_angstrom: Initial internuclear distance in Ångströms.
        v_rel0_ang_per_fs: Initial relative velocity in Å/fs.
        t_final_fs: Total simulation time in femtoseconds.
        dt_fs: Time step in femtoseconds.
        force_func: Callable that takes R (Bohr) and returns force (Ha/Bohr).

    Returns:
        A tuple ``(t_fs, r_angstrom, v_rel_ang_per_fs)`` with arrays of
        time, distance, and velocity in the respective external units.
    """
    r0_bohr = r0_angstrom * ANGSTROM_TO_BOHR
    v0_au = _v_angfs_to_au(v_rel0_ang_per_fs)
    dt_au = _fs_to_au(dt_fs)
    n_steps = int(t_final_fs / dt_fs) + 1

    t_fs_arr = np.linspace(0.0, t_final_fs, n_steps)
    r_bohr = np.zeros(n_steps)
    v_au = np.zeros(n_steps)

    r_bohr[0] = r0_bohr
    v_au[0] = v0_au

    a_old = force_func(r_bohr[0]) / MU

    for i in range(1, n_steps):
        r_bohr[i] = r_bohr[i - 1] + v_au[i - 1] * dt_au + 0.5 * a_old * dt_au**2
        a_new = force_func(r_bohr[i]) / MU
        v_au[i] = v_au[i - 1] + 0.5 * (a_old + a_new) * dt_au
        a_old = a_new

    r_angstrom = r_bohr * BOHR_TO_ANGSTROM
    v_rel_ang_per_fs = _v_au_to_angfs(v_au)

    return t_fs_arr, r_angstrom, v_rel_ang_per_fs


# ---------------------------------------------------------------------------
# Sequential VQE-force MD
# ---------------------------------------------------------------------------

def dynamics_seq(timestamp: str, backend: Any | None = None) -> None:
    """Run a sequential MD simulation using VQE forces at each step.

    Writes per-step results to ``logs/<timestamp>/dynamics_seq.csv``.

    Args:
        timestamp: Log-directory timestamp string (YYMMDDHHmm).
        backend: Optional pre-initialised backend (unused for statevector).
    """
    # シミュレーションパラメータ
    initial_r_angstrom = 0.8
    initial_v_ang_per_fs = 0.0
    time_step_fs = 0.01
    total_step = 1000

    dt_au = time_step_fs / T_AU_FS
    r_bohr = initial_r_angstrom * ANGSTROM_TO_BOHR
    v_bohr_per_au = initial_v_ang_per_fs * ANGSTROM_TO_BOHR / (1.0 / T_AU_FS)

    out_dir = f"logs/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "dynamics_seq.csv")

    results: list[dict[str, float]] = []

    def _eval_energy_force(rb: float) -> tuple[float, float]:
        """Return (energy_hartree, force_hartree_per_bohr) at R=rb (Bohr)."""
        r_ang = rb * BOHR_TO_ANGSTROM
        energy_hartree, force_hartree_per_angstrom = compute_h2_energy_quantum_statevector(
            r_ang,
            basis="sto-3g",
            timestamp=timestamp,
            ansatz_reps=1,
            optimizer_maxiter=2000,
            cholesky_tol=1e-10,
        )
        if not isinstance(force_hartree_per_angstrom, float):
            raise ValueError("compute_h2_energy_quantum_statevector did not return force value.")
        # Ha/Å → Ha/Bohr
        force_hartree_per_bohr = force_hartree_per_angstrom * BOHR_TO_ANGSTROM
        return float(energy_hartree), float(force_hartree_per_bohr)

    energy_hartree, force_hartree_per_bohr = _eval_energy_force(r_bohr)

    for i in range(total_step):
        logger.info("step %d / %d", i, total_step)
        t_fs = i * time_step_fs

        results.append(
            {
                "step": float(i),
                "t_fs": float(t_fs),
                "R_ang": float(r_bohr * BOHR_TO_ANGSTROM),
                "v_ang_per_fs": float(v_bohr_per_au * BOHR_TO_ANGSTROM * T_AU_FS),
                "E_ha": float(energy_hartree),
                "F_ha_per_ang": float(force_hartree_per_bohr / BOHR_TO_ANGSTROM),
            }
        )

        # ガードレール: R が非物理的になったら停止
        if r_bohr * BOHR_TO_ANGSTROM < 0.2 or r_bohr * BOHR_TO_ANGSTROM > 5.0:
            break

        a_bohr_per_au2 = force_hartree_per_bohr / MU
        v_half = v_bohr_per_au + 0.5 * a_bohr_per_au2 * dt_au
        r_next = r_bohr + v_half * dt_au

        energy_next, force_next = _eval_energy_force(r_next)
        a_next = force_next / MU
        v_next = v_half + 0.5 * a_next * dt_au

        r_bohr = r_next
        v_bohr_per_au = v_next
        energy_hartree = energy_next
        force_hartree_per_bohr = force_next

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["step", "t_fs", "R_ang", "v_ang_per_fs", "E_ha", "F_ha_per_ang"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    logger.info("Wrote: %s", out_csv)
    if results:
        last = results[-1]
        print(
            "Last state: "
            f"t={last['t_fs']:.6f} fs, R={last['R_ang']:.6f} Å, "
            f"v={last['v_ang_per_fs']:.6e} Å/fs, "
            f"E={last['E_ha']:.12f} Ha, F={last['F_ha_per_ang']:.6e} Ha/Å"
        )


# ---------------------------------------------------------------------------
# Parallel-execution utilities
# ---------------------------------------------------------------------------

def chunk_list(lst: list, n: int) -> list[list]:
    """Split *lst* into roughly *n* equal-sized chunks.

    Args:
        lst: The list to split.
        n: Desired number of chunks.

    Returns:
        A list of sub-lists.
    """
    k = math.ceil(len(lst) / n)
    return [lst[i : i + k] for i in range(0, len(lst), k)]


def energy_worker(
    target_distance_list: list[float],
    timestamp: str,
    backend_name: str,
) -> list[tuple[float, float]]:
    """Compute VQE forces for a batch of distances (parallel worker).

    Args:
        target_distance_list: Bond distances in Ångströms to evaluate.
        timestamp: Log-directory timestamp string.
        backend_name: IBM backend name for QiskitRuntimeService.

    Returns:
        A list of ``(force, distance)`` tuples.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService as _QRS

    service = _QRS()
    backend = service.backend(backend_name)
    results = []
    for distance in target_distance_list:
        logger.info("Starting VQE for distance=%.4f Å", distance)
        fq = force_quantum_angstrom(float(distance), timestamp=timestamp, backend=backend)
        results.append((fq, distance))
        logger.info("Finished VQE for distance=%.4f Å, force=%.6f Ha/Å", distance, fq)
    return results
