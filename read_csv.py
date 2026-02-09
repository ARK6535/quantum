"""Visualize MD trajectory data from CSV logs."""
from __future__ import annotations

import argparse
import csv
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit

from h2_dynamics import force_classical_angstrom, force_classical_bohr, simulate_h2_1d
from h2_helpers import (
    AMU_TO_KG,
    ANGSTROM_TO_METER,
    FS_TO_SECOND,
    HARTREE_TO_JOULE,
    compute_h2_energy_classical,
)

__all__ = [
    "kinetic_from_positions",
]

logger = logging.getLogger(__name__)


def kinetic_from_positions(
    times_fs: ArrayLike,
    positions_angstrom: ArrayLike,
    mass_amu: float = 1.00784,
) -> np.ndarray:
    """Compute kinetic energy from a position-vs-time trace using reduced mass.

    Args:
        times_fs: Time points in femtoseconds.
        positions_angstrom: Internuclear distances in Ångströms.
        mass_amu: Atomic mass of hydrogen in AMU.

    Returns:
        Kinetic energy at each time point in Hartree.
    """
    times_fs = np.asarray(times_fs, dtype=float)
    positions_angstrom = np.asarray(positions_angstrom, dtype=float)
    if len(times_fs) < 2:
        return np.zeros_like(positions_angstrom)

    mu_kg = (mass_amu / 2.0) * AMU_TO_KG
    times_s = times_fs * FS_TO_SECOND
    positions_m = positions_angstrom * ANGSTROM_TO_METER
    velocities = np.gradient(positions_m, times_s, edge_order=2)
    ke_joule = 0.5 * mu_kg * velocities**2
    ke_hartree = ke_joule / HARTREE_TO_JOULE
    return ke_hartree


def _sine(t: np.ndarray, A: float, omega: float, phi: float, C: float) -> np.ndarray:
    """Sine model for force-vs-time fitting."""
    return A * np.sin(omega * t + phi) + C


def _load_csv(
    csv_path: str,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Load MD trajectory CSV and return per-step arrays.

    Args:
        csv_path: Path to the dynamics_seq.csv file.

    Returns:
        Tuple of (times_fs, distances_angstrom, forces_ha_per_ang,
        energies_hartree).
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        times_fs: list[float] = []
        distances_angstrom: list[float] = []
        forces_ha_per_ang: list[float] = []
        energies_hartree: list[float] = []
        for row in reader:
            # 横軸: step * 0.01 → fs
            times_fs.append(float(row["step"]) * 0.01)
            distances_angstrom.append(float(row["R_ang"]))
            forces_ha_per_ang.append(float(row["F_ha_per_ang"]))
            if "E_ha" in row:
                energies_hartree.append(float(row["E_ha"]))
            else:
                energies_hartree.append(np.nan)
    return times_fs, distances_angstrom, forces_ha_per_ang, energies_hartree


def _compute_classical_trajectory(
    log_dir: str,
    r0_angstrom: float,
    dt_fs: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute or load cached classical trajectory.

    Args:
        log_dir: Log directory path for caching.
        r0_angstrom: Initial internuclear distance in Ångströms.
        dt_fs: Time step in femtoseconds.
        n_points: Number of trajectory points.

    Returns:
        Tuple of (times_fs, distances_angstrom, forces_ha_per_ang,
        energies_hartree) as NumPy arrays.
    """
    cache_path = os.path.join(log_dir, "classical_dynamics.npz")

    if os.path.exists(cache_path):
        logger.info("Loading cached classical dynamics data...")
        cache = np.load(cache_path)
        times_classical = cache["times_classical"]
        Rs_classical = cache["Rs_classical"]
        f_classical = cache["f_classical"]
        energies_classical = (
            cache["energies_classical"]
            if "energies_classical" in cache.files
            else None
        )
    else:
        t_final_fs = dt_fs * (n_points - 1)
        times_classical, Rs_classical, _ = simulate_h2_1d(
            r0_angstrom=r0_angstrom,
            v_rel0_ang_per_fs=0.0,
            t_final_fs=t_final_fs,
            dt_fs=dt_fs,
            force_func=force_classical_bohr,
        )
        f_classical = np.array(
            [force_classical_angstrom(r) for r in Rs_classical],
        )
        energies_classical = np.array(
            [compute_h2_energy_classical(r) for r in Rs_classical],
        )
        np.savez(
            cache_path,
            times_classical=np.asarray(times_classical, dtype=float),
            Rs_classical=np.asarray(Rs_classical, dtype=float),
            f_classical=np.asarray(f_classical, dtype=float),
            energies_classical=np.asarray(energies_classical, dtype=float),
        )

    # 古いキャッシュにエネルギーが無い場合の補完
    if energies_classical is None:
        energies_classical = np.array(
            [compute_h2_energy_classical(r) for r in Rs_classical],
        )
        np.savez(
            cache_path,
            times_classical=np.asarray(times_classical, dtype=float),
            Rs_classical=np.asarray(Rs_classical, dtype=float),
            f_classical=np.asarray(f_classical, dtype=float),
            energies_classical=np.asarray(energies_classical, dtype=float),
        )

    return times_classical, Rs_classical, f_classical, energies_classical


def _fit_sine_to_force(
    times: np.ndarray,
    forces: np.ndarray,
) -> tuple[np.ndarray | None, bool]:
    """Fit a sine wave to force-vs-time data.

    Args:
        times: Time array in fs.
        forces: Force array in Ha/Å.

    Returns:
        Tuple of (fitted_parameters, success_flag).
    """
    mask = np.isfinite(times) & np.isfinite(forces)
    if np.count_nonzero(mask) < 4:
        return None, False

    t_fit = times[mask]
    y_fit = forces[mask]
    span = max(t_fit) - min(t_fit) if len(t_fit) > 1 else 1.0
    A0 = 0.5 * (np.nanmax(y_fit) - np.nanmin(y_fit)) if len(y_fit) else 0.1
    C0 = float(np.nanmean(y_fit)) if len(y_fit) else 0.0
    omega0 = 2 * np.pi / span if span > 0 else 1.0
    phi0 = 0.0

    try:
        popt, _ = curve_fit(
            _sine, t_fit, y_fit, p0=[A0, omega0, phi0, C0], maxfev=10000,
        )
        return popt, True
    except Exception as exc:
        logger.warning("Sine fit failed: %s", exc)
        return None, False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace with ``log_dir``.
    """
    parser = argparse.ArgumentParser(
        description="Visualize MD trajectory data from CSV logs.",
    )
    parser.add_argument(
        "--log-dir",
        required=True,
        help="Path to the log directory containing dynamics_seq.csv.",
    )
    return parser.parse_args()


def main() -> None:  # noqa: C901 — plotting script with many sequential plots
    """Entry point for the MD trajectory visualization script."""
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger(__name__).setLevel(logging.INFO)

    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    # --- CSV読み込み ---
    csv_path = os.path.join(log_dir, "dynamics_seq.csv")
    times_vqe, Rs_vqe, f_vqe, energies_vqe = _load_csv(csv_path)
    dt_fs = times_vqe[1] - times_vqe[0] if len(times_vqe) > 1 else 0.01

    # --- 古典軌道の計算（キャッシュ付き） ---
    times_classical, Rs_classical, f_classical, energies_classical = (
        _compute_classical_trajectory(
            log_dir,
            r0_angstrom=Rs_vqe[0],
            dt_fs=dt_fs,
            n_points=len(times_vqe),
        )
    )

    # --- Distance vs Time ---
    plt.figure(figsize=(10, 6))
    plt.plot(times_vqe, Rs_vqe, "o-", label="VQE")
    plt.plot(times_classical, Rs_classical, "--", label="Classical")
    plt.xlabel("Time (fs)")
    plt.ylabel("H-H Distance (Angstrom)")
    plt.title("H2 Molecular Dynamics Simulation")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(log_dir, "h2_dynamics_distance_vs_time.png"), dpi=200,
    )
    plt.close()
    logger.info("Saved plot to h2_dynamics_distance_vs_time.png")

    # 比較プロット
    plt.figure(figsize=(10, 6))
    plt.plot(times_vqe, Rs_vqe, "o-", label="VQE")
    plt.plot(times_classical, Rs_classical, "--", label="Classical")
    plt.xlabel("Time (fs)")
    plt.ylabel("Distance (Angstrom)")
    plt.title("H2 Vibration Trajectory")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(log_dir, "_h2_dynamics_comparison.png"), dpi=200)
    plt.close()
    logger.info("Saved plot to _h2_dynamics_comparison.png")

    # --- Force vs Time ---
    plt.figure(figsize=(10, 6))
    plt.plot(times_vqe, f_vqe, "o-", label="VQE Force")
    plt.plot(times_classical, f_classical, "--", label="Classical Force")
    plt.xlabel("Time (fs)")
    plt.ylabel("Force (Ha/Angstrom)")
    plt.title("H2 Force vs Time")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(log_dir, "h2_dynamics_force_vs_time.png"), dpi=200,
    )
    plt.close()
    logger.info("Saved plot to h2_dynamics_force_vs_time.png")

    # --- サインフィット (力 vs 時間) ---
    steps_arr = np.asarray(times_vqe, dtype=float)
    f_vqe_arr = np.asarray(f_vqe, dtype=float)
    popt_vqe, fit_succeeded = _fit_sine_to_force(steps_arr, f_vqe_arr)

    if fit_succeeded:
        assert popt_vqe is not None
        t_dense = np.linspace(
            float(np.nanmin(steps_arr)), float(np.nanmax(steps_arr)), 400,
        )
        fit_curve_vqe = _sine(t_dense, *popt_vqe)

        # 古典力にもサインフィット
        f_class_arr = np.asarray(f_classical, dtype=float)
        popt_classical, fit_classical_ok = _fit_sine_to_force(
            np.asarray(times_classical, dtype=float), f_class_arr,
        )
        fit_curve_classical = (
            _sine(t_dense, *popt_classical)
            if fit_classical_ok and popt_classical is not None
            else None
        )

        plt.figure(figsize=(10, 6))
        plt.plot(steps_arr, f_vqe_arr, "o", label="VQE Force (data)")
        plt.plot(t_dense, fit_curve_vqe, "-", label="VQE Sine fit")
        plt.plot(times_classical, f_classical, "x", label="Classical Force (data)")
        plt.xlabel("Time (fs)")
        plt.ylabel("Force (Ha/Angstrom)")
        plt.title("Force vs Time (Sine Fits)")
        plt.grid(True)
        plt.legend()
        plt.savefig(
            os.path.join(log_dir, "h2_dynamics_force_vs_time_fit.png"), dpi=200,
        )
        plt.close()
        logger.info("Saved plot to h2_dynamics_force_vs_time_fit.png")
    else:
        logger.info("Sine fit skipped (insufficient or invalid data)")

    # --- Classical force vs distance ---
    plt.figure(figsize=(10, 6))
    plt.plot(Rs_classical, f_classical, "o-", label="Classical Force")
    plt.xlabel("H-H Distance (Angstrom)")
    plt.ylabel("Force (Ha/Angstrom)")
    plt.title("Classical Force vs Distance")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(log_dir, "h2_classical_force_vs_distance.png"), dpi=200,
    )
    plt.close()
    logger.info("Saved plot to h2_classical_force_vs_distance.png")

    # --- Classical energy vs distance ---
    plt.figure(figsize=(10, 6))
    plt.plot(Rs_classical, energies_classical, "o-", label="Classical Energy")
    plt.plot(Rs_vqe, energies_vqe, "x--", label="VQE Energy")
    plt.xlabel("H-H Distance (Angstrom)")
    plt.ylabel("Energy (Hartree)")
    plt.title("Classical Energy vs Distance")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(log_dir, "h2_classical_energy_vs_distance.png"), dpi=200,
    )
    plt.close()
    logger.info("Saved plot to h2_classical_energy_vs_distance.png")

    # --- Energy vs distance (VQE vs Classical) ---
    plt.figure(figsize=(10, 6))
    plt.plot(Rs_vqe, energies_vqe, "o-", label="VQE Energy")
    plt.plot(Rs_classical, energies_classical, "--", label="Classical Energy")
    plt.xlabel("H-H Distance (Angstrom)")
    plt.ylabel("Energy (Hartree)")
    plt.title("Energy vs Distance (VQE vs Classical)")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(log_dir, "h2_energy_vs_distance.png"), dpi=200,
    )
    plt.close()
    logger.info("Saved plot to h2_energy_vs_distance.png")

    # --- Energy vs time ---
    plt.figure(figsize=(10, 6))
    plt.plot(times_vqe, energies_vqe, "o-", label="VQE Energy")
    plt.plot(times_classical, energies_classical, "--", label="Classical Energy")
    plt.xlabel("Time (fs)")
    plt.ylabel("Energy (Hartree)")
    plt.title("Energy vs Time (VQE vs Classical)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(log_dir, "h2_energy_vs_time.png"), dpi=200)
    plt.close()
    logger.info("Saved plot to h2_energy_vs_time.png")

    # --- Total energy (potential + kinetic) vs time ---
    ke_classical = kinetic_from_positions(times_classical, Rs_classical)
    ke_vqe = kinetic_from_positions(times_vqe, Rs_vqe)
    total_classical = np.asarray(energies_classical, dtype=float) + ke_classical
    total_vqe = np.asarray(energies_vqe, dtype=float) + ke_vqe

    plt.figure(figsize=(10, 6))
    plt.plot(times_vqe, total_vqe, "o-", label="VQE Total Energy")
    plt.plot(times_classical, total_classical, "--", label="Classical Total Energy")
    plt.xlabel("Time (fs)")
    plt.ylabel("Energy (Hartree)")
    plt.title("Total Energy vs Time (VQE vs Classical)")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(log_dir, "h2_total_energy_vs_time.png"), dpi=200,
    )
    plt.close()
    logger.info("Saved plot to h2_total_energy_vs_time.png")

    # --- サマリープロット ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    axes[0].plot(times_vqe, Rs_vqe, "o-", label="VQE")
    axes[0].plot(times_classical, Rs_classical, "--", label="Full CI")
    axes[0].set_ylabel("H-H Distance (Angstrom)", fontsize=20)
    axes[0].set_title("H2 Molecular Dynamics Simulation", fontsize=20)
    axes[0].grid(True)
    axes[0].legend(fontsize=20)
    axes[0].tick_params(axis="y", which="major", labelsize=20)

    axes[1].plot(times_vqe, f_vqe, "o-", label="VQE Force")
    axes[1].plot(times_classical, f_classical, "--", label="Full CI Force")
    axes[1].set_xlabel("Time (fs)")
    axes[1].set_ylabel("Force (Ha/Angstrom)", fontsize=20)
    axes[1].set_title("H2 Force vs Time", fontsize=20)
    axes[1].grid(True)
    axes[1].legend(fontsize=20)
    axes[1].tick_params(axis="y", which="major", labelsize=20)

    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, "h2_dynamics_summary.png"), dpi=200)
    plt.close(fig)
    logger.info("Saved plot to h2_dynamics_summary.png")


if __name__ == "__main__":
    main()
