"""Molecular dynamics simulation helpers for H2.

Provides Velocity-Verlet integrators, unit-conversion wrappers, and
parallel-execution utilities used by the main demo script.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import shutil
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from h2_energy_noisy import compute_h2_energy_quantum_noisy
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

def energy_classical_bohr(r_bohr: float, *, basis: str = "sto-3g") -> float:
    """Return the classical Full-CI energy at *r_bohr* (Bohr) in Hartree.

    Args:
        r_bohr: Internuclear distance in Bohr.
        basis: Gaussian basis set name.

    Returns:
        Total electronic energy in Hartree.
    """
    r_angstrom = r_bohr * BOHR_TO_ANGSTROM
    return compute_h2_energy_classical(r_angstrom, basis=basis)


def energy_quantum_bohr(
    r_bohr: float,
    timestamp: str,
    backend: Any | None = None,
    *,
    basis: str = "sto-3g",
    ansatz_reps: int = 1,
    optimizer_maxiter: int = 2000,
    cholesky_tol: float = 1e-10,
    backend_type: str = "statevector",
    shots: int = 4096,
    force_shots: int = 16384,
) -> float:
    """Return the VQE energy at *r_bohr* (Bohr) in Hartree.

    Args:
        r_bohr: Internuclear distance in Bohr.
        timestamp: Log-directory timestamp string (YYMMDDHHmm).
        backend: Optional pre-initialised backend.
        basis: Gaussian basis set name.
        ansatz_reps: Number of ansatz repetition layers.
        optimizer_maxiter: Maximum iterations per optimiser stage.
        cholesky_tol: Tolerance for Cholesky decomposition.
        backend_type: ``"statevector"`` or ``"noisy"``.
        shots: Number of shots for noisy energy evaluation.
        force_shots: Number of shots for noisy force evaluation.

    Returns:
        Minimum VQE energy in Hartree.
    """
    r_angstrom = r_bohr * BOHR_TO_ANGSTROM
    if backend_type == "noisy":
        val = compute_h2_energy_quantum_noisy(
            r_angstrom,
            timestamp=timestamp,
            basis=basis,
            ansatz_reps=ansatz_reps,
            optimizer_maxiter=optimizer_maxiter,
            cholesky_tol=cholesky_tol,
            backend=backend,
            shots=shots,
            force_shots=force_shots,
        )
    else:
        val = compute_h2_energy_quantum_statevector(
            r_angstrom,
            timestamp=timestamp,
            basis=basis,
            ansatz_reps=ansatz_reps,
            optimizer_maxiter=optimizer_maxiter,
            cholesky_tol=cholesky_tol,
        )
    if isinstance(val, tuple):
        return val[0]
    return val


def force_quantum_bohr(
    r_bohr: float,
    timestamp: str,
    h_bohr: float = 0.01,
    backend: Any | None = None,
    *,
    basis: str = "sto-3g",
    ansatz_reps: int = 1,
    optimizer_maxiter: int = 2000,
    cholesky_tol: float = 1e-10,
    backend_type: str = "statevector",
    shots: int = 4096,
    force_shots: int = 16384,
) -> float:
    """Return the VQE force at *r_bohr* via central finite difference.

    Args:
        r_bohr: Internuclear distance in Bohr.
        timestamp: Log-directory timestamp string.
        h_bohr: Finite-difference step in Bohr.
        backend: Optional pre-initialised backend.
        basis: Gaussian basis set name.
        ansatz_reps: Number of ansatz repetition layers.
        optimizer_maxiter: Maximum iterations per optimiser stage.
        cholesky_tol: Tolerance for Cholesky decomposition.
        backend_type: ``"statevector"`` or ``"noisy"``.
        shots: Number of shots for noisy energy evaluation.
        force_shots: Number of shots for noisy force evaluation.

    Returns:
        Force in Hartree/Bohr (negative gradient).
    """
    e_plus = energy_quantum_bohr(
        r_bohr + h_bohr, backend=backend, timestamp=timestamp,
        basis=basis, ansatz_reps=ansatz_reps,
        optimizer_maxiter=optimizer_maxiter, cholesky_tol=cholesky_tol,
        backend_type=backend_type, shots=shots, force_shots=force_shots,
    )
    e_minus = energy_quantum_bohr(
        r_bohr - h_bohr, backend=backend, timestamp=timestamp,
        basis=basis, ansatz_reps=ansatz_reps,
        optimizer_maxiter=optimizer_maxiter, cholesky_tol=cholesky_tol,
        backend_type=backend_type, shots=shots, force_shots=force_shots,
    )
    return -(e_plus - e_minus) / (2.0 * h_bohr)


def force_quantum_angstrom(
    r_angstrom: float,
    timestamp: str,
    backend: Any | None = None,
    *,
    basis: str = "sto-3g",
    ansatz_reps: int = 1,
    optimizer_maxiter: int = 2000,
    cholesky_tol: float = 1e-10,
    backend_type: str = "statevector",
    shots: int = 4096,
    force_shots: int = 16384,
) -> float:
    """Return the VQE force at *r_angstrom* (Å) in Ha/Å.

    Args:
        r_angstrom: Internuclear distance in Ångströms.
        timestamp: Log-directory timestamp string.
        backend: Optional pre-initialised backend.
        basis: Gaussian basis set name.
        ansatz_reps: Number of ansatz repetition layers.
        optimizer_maxiter: Maximum iterations per optimiser stage.
        cholesky_tol: Tolerance for Cholesky decomposition.
        backend_type: ``"statevector"`` or ``"noisy"``.
        shots: Number of shots for noisy energy evaluation.
        force_shots: Number of shots for noisy force evaluation.

    Returns:
        Force in Hartree/Ångström.
    """
    r_bohr = r_angstrom * ANGSTROM_TO_BOHR
    f_bohr = force_quantum_bohr(
        r_bohr, backend=backend, timestamp=timestamp,
        basis=basis, ansatz_reps=ansatz_reps,
        optimizer_maxiter=optimizer_maxiter, cholesky_tol=cholesky_tol,
        backend_type=backend_type, shots=shots, force_shots=force_shots,
    )
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

def dynamics_seq(
    timestamp: str,
    backend: Any | None = None,
    *,
    initial_r_angstrom: float = 0.8,
    initial_v_ang_per_fs: float = 0.0,
    time_step_fs: float = 0.01,
    total_step: int = 1000,
    basis: str = "sto-3g",
    ansatz_reps: int = 1,
    optimizer_maxiter: int = 2000,
    resume_from: str | None = None,
    backend_type: str = "statevector",
    shots: int = 4096,
    force_shots: int = 16384,
) -> None:
    """Run a sequential MD simulation using VQE forces at each step.

    CSV rows are written incrementally after each step so that partial
    results survive process termination.  A ``checkpoint.json`` file is
    updated every step; pass its directory as *resume_from* to continue
    a previous run.

    Args:
        timestamp: Log-directory timestamp string (YYMMDDHHmm).
        backend: Optional pre-initialised backend.
        initial_r_angstrom: Initial H-H distance in Ångströms.
        initial_v_ang_per_fs: Initial relative velocity in Å/fs.
        time_step_fs: Integration time step in femtoseconds.
        total_step: Number of Velocity-Verlet steps.
        basis: Gaussian basis set name.
        ansatz_reps: Number of ansatz repetition layers.
        optimizer_maxiter: Maximum iterations per optimiser stage.
        resume_from: Path to a log directory containing ``checkpoint.json``.
            When given, simulation state is restored and appended to the
            existing CSV.
        backend_type: ``"statevector"`` or ``"noisy"``.
        shots: Number of shots for noisy energy evaluation.
        force_shots: Number of shots for noisy force evaluation.
    """
    out_dir = f"logs/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "dynamics_seq.csv")
    ckpt_path = os.path.join(out_dir, "checkpoint.json")

    fieldnames = ["step", "t_fs", "R_ang", "v_ang_per_fs", "E_ha", "F_ha_per_ang"]

    def _eval_energy_force(rb: float) -> tuple[float, float]:
        """Return (energy_hartree, force_hartree_per_bohr) at R=rb (Bohr)."""
        r_ang = rb * BOHR_TO_ANGSTROM
        if backend_type == "noisy":
            energy_hartree, force_hartree_per_angstrom = compute_h2_energy_quantum_noisy(
                r_ang,
                timestamp=timestamp,
                basis=basis,
                ansatz_reps=ansatz_reps,
                optimizer_maxiter=optimizer_maxiter,
                cholesky_tol=1e-10,
                backend=backend,
                shots=shots,
                force_shots=force_shots,
            )
        else:
            energy_hartree, force_hartree_per_angstrom = compute_h2_energy_quantum_statevector(
                r_ang,
                timestamp=timestamp,
                basis=basis,
                ansatz_reps=ansatz_reps,
                optimizer_maxiter=optimizer_maxiter,
                cholesky_tol=1e-10,
            )
        if not isinstance(force_hartree_per_angstrom, float):
            raise ValueError("VQE did not return a force value.")
        force_hartree_per_bohr = force_hartree_per_angstrom * BOHR_TO_ANGSTROM
        return float(energy_hartree), float(force_hartree_per_bohr)

    # --- 状態の初期化 or チェックポイントからの復元 ---
    start_step = 0
    if resume_from:
        ckpt_src = os.path.join(resume_from, "checkpoint.json")
        if not os.path.isfile(ckpt_src):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_src}")
        with open(ckpt_src) as f:
            ckpt = json.load(f)
        start_step = ckpt["next_step"]
        r_bohr = ckpt["r_bohr"]
        v_bohr_per_au = ckpt["v_bohr_per_au"]
        energy_hartree = ckpt["energy_hartree"]
        force_hartree_per_bohr = ckpt["force_hartree_per_bohr"]
        time_step_fs = ckpt["time_step_fs"]
        logger.info(
            "Resumed from checkpoint: step=%d, R=%.6f Bohr, E=%.12f Ha",
            start_step, r_bohr, energy_hartree,
        )
        # 前回の CSV を新ディレクトリにコピーして続きを追記
        prev_csv = os.path.join(resume_from, "dynamics_seq.csv")
        if resume_from != out_dir and os.path.isfile(prev_csv):
            shutil.copy2(prev_csv, out_csv)
    else:
        r_bohr = initial_r_angstrom * ANGSTROM_TO_BOHR
        v_bohr_per_au = initial_v_ang_per_fs * ANGSTROM_TO_BOHR / (1.0 / T_AU_FS)
        energy_hartree, force_hartree_per_bohr = _eval_energy_force(r_bohr)

    dt_au = time_step_fs / T_AU_FS

    # --- CSV を追記モードで開く ---
    csv_exists = os.path.isfile(out_csv) and os.path.getsize(out_csv) > 0
    csv_file = open(out_csv, "a", newline="")  # noqa: SIM115
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not csv_exists:
        writer.writeheader()
        csv_file.flush()

    last_row: dict[str, float] | None = None

    try:
        for i in range(start_step, total_step):
            logger.info("step %d / %d", i, total_step)
            t_fs = i * time_step_fs

            row = {
                "step": float(i),
                "t_fs": float(t_fs),
                "R_ang": float(r_bohr * BOHR_TO_ANGSTROM),
                "v_ang_per_fs": float(v_bohr_per_au * BOHR_TO_ANGSTROM * T_AU_FS),
                "E_ha": float(energy_hartree),
                "F_ha_per_ang": float(force_hartree_per_bohr / BOHR_TO_ANGSTROM),
            }
            writer.writerow(row)
            csv_file.flush()
            last_row = row

            # ガードレール: R が非物理的になったら停止
            if r_bohr * BOHR_TO_ANGSTROM < 0.2 or r_bohr * BOHR_TO_ANGSTROM > 5.0:
                logger.warning(
                    "R=%.4f Å is outside [0.2, 5.0] Å — stopping.",
                    r_bohr * BOHR_TO_ANGSTROM,
                )
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

            # チェックポイント: アトミック書き込み
            ckpt_data = {
                "next_step": i + 1,
                "total_step": total_step,
                "r_bohr": r_bohr,
                "v_bohr_per_au": v_bohr_per_au,
                "energy_hartree": energy_hartree,
                "force_hartree_per_bohr": force_hartree_per_bohr,
                "time_step_fs": time_step_fs,
                "timestamp": timestamp,
            }
            ckpt_tmp = ckpt_path + ".tmp"
            with open(ckpt_tmp, "w") as cf:
                json.dump(ckpt_data, cf, indent=2)
            os.replace(ckpt_tmp, ckpt_path)

    finally:
        csv_file.close()

    logger.info("Wrote: %s", out_csv)
    if last_row:
        print(
            "Last state: "
            f"t={last_row['t_fs']:.6f} fs, R={last_row['R_ang']:.6f} Å, "
            f"v={last_row['v_ang_per_fs']:.6e} Å/fs, "
            f"E={last_row['E_ha']:.12f} Ha, F={last_row['F_ha_per_ang']:.6e} Ha/Å"
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
