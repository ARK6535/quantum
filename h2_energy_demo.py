"""Quick demo comparing classical and VQE energies for H2."""
from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from matplotlib import pyplot as plt

from h2_dynamics import (
    chunk_list,
    dynamics_seq,
    energy_classical_bohr,
    energy_worker,
    force_classical_angstrom,
)
from h2_helpers import ANGSTROM_TO_BOHR, compute_h2_energy_classical

__all__ = [
    "main",
    "parse_args",
]

logger = logging.getLogger(__name__)


def plot_dynamics_results(
    results: list[tuple[np.ndarray, np.ndarray, float | np.ndarray, str]],
    timestamp: str,
    title_suffix: str = "",
) -> None:
    """Plot MD simulation results (distance and velocity vs time).

    Args:
        results: List of (t_fs, r_angstrom, v_rel, label) tuples.
        timestamp: Log-directory timestamp for saving the figure.
        title_suffix: Optional suffix appended to plot titles.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot R
    plt.subplot(2, 1, 1)
    for t_fs, R_ang, _, label in results:
        plt.plot(t_fs, R_ang, label=label)
    plt.xlabel("Time (fs)")
    plt.ylabel("R (Å)")
    plt.title(f"H2 Nuclear Distance vs Time {title_suffix}")
    plt.grid(True)
    plt.legend()

    # Plot v
    plt.subplot(2, 1, 2)
    for t_fs, _, v_rel, label in results:
        plt.plot(t_fs, v_rel, label=label)
    plt.xlabel("Time (fs)")
    plt.ylabel("dR/dt (Å/fs)")
    plt.title(f"H2 Relative Velocity vs Time {title_suffix}")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    os.makedirs(f"logs/{timestamp}", exist_ok=True)
    plt.savefig(f"logs/{timestamp}/h2_dynamics{title_suffix.replace(' ', '_')}.png")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H2 energy helper demo")
    parser.add_argument(
        "--ansatz-reps",
        type=int,
        default=0,
        help="Number of EfficientSU2 repetitions.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=80,
        help="Maximum COBYLA iterations.",
    )
    parser.add_argument(
        "--backend-name",
        default=None,
        help="IBM Quantum backend to mimic with AerSimulator (defaults to least busy real backend).",
    )
    return parser.parse_args()

def plot_force_curve(
    timestamp: str,
    start_ang: float = 0.5,
    end_ang: float = 3.0,
    step_ang: float = 0.1,
) -> None:
    """Plot the classical force curve F(R) over a range of distances.

    Args:
        timestamp: Log-directory timestamp for saving the figure.
        start_ang: Start distance in Ångströms.
        end_ang: End distance in Ångströms.
        step_ang: Distance step in Ångströms.
    """
    distances_ang = np.arange(start_ang, end_ang + 1e-9, step_ang)
    forces = []

    logger.info("Calculating classical forces for R=[%.1f, %.1f] Å", start_ang, end_ang)
    for r_ang in distances_ang:
        f = force_classical_angstrom(float(r_ang))
        forces.append(f)

    plt.figure(figsize=(8, 5))
    plt.plot(distances_ang, forces, "o-", label="Force (Full CI)")
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Distance R (Å)")
    plt.ylabel("Force (Hartree/Å)")
    plt.title("H2 Force vs Distance")
    plt.grid(True)
    plt.legend()
    
    os.makedirs(f"logs/{timestamp}", exist_ok=True)
    output_path = f"logs/{timestamp}/h2_force_curve.png"
    plt.savefig(output_path)
    logger.info("Force curve saved to %s", output_path)


def plot_force_curve_comparison(
    timestamp: str,
    backend_name: str = "None",
    start_ang: float = 0.5,
    end_ang: float = 3.0,
    step_ang: float = 0.1,
) -> None:
    """Plot classical vs quantum force curves over a range of distances.

    Classical forces are computed serially; quantum forces are computed in
    parallel using :func:`h2_dynamics.energy_worker`.

    Args:
        timestamp: Log-directory timestamp for saving the figure.
        backend_name: IBM backend name for the quantum force computation.
        start_ang: Start distance in Ångströms.
        end_ang: End distance in Ångströms.
        step_ang: Distance step in Ångströms.
    """
    distances_ang = np.arange(start_ang, end_ang + 1e-9, step_ang)
    forces_classical = []

    logger.info("Calculating forces (Classical & Quantum) for R=[%.1f, %.1f] Å", start_ang, end_ang)
    for r_ang in distances_ang:
        logger.debug("Processing R=%.2f Å", r_ang)
        fc = force_classical_angstrom(float(r_ang))
        forces_classical.append(fc)
    
    # Parallelize quantum force evaluations across CPU cores using ProcessPoolExecutor.
    n_cores = os.cpu_count() or 1
    n_workers = max(1, n_cores - 2)
    n_workers = min(n_workers, len(distances_ang))
    chunks = chunk_list(list(distances_ang), n_workers)

    forces_quantum = []
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for chunk in chunks:
                new_chunk = [float(r_ang) for r_ang in chunk]
                futures.append(executor.submit(energy_worker, new_chunk, timestamp, backend_name))
            for fut in futures:
                forces_quantum.extend(fut.result())
    except Exception as e:
        logger.error("Parallel execution failed: %s", e)
        raise
    # Sort forces_quantum based on distances to ensure correct order
    forces_quantum_sorted = [0.0] * len(distances_ang)
    for fq, r_ang in forces_quantum:
        index = np.where(distances_ang == r_ang)[0][0]
        forces_quantum_sorted[index] = fq

    plt.figure(figsize=(8, 5))
    plt.plot(distances_ang, forces_classical, "o-", label="Force (Full CI)")
    plt.plot(distances_ang, forces_quantum_sorted, "x--", label="Force (VQE)")
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Distance R (Å)")
    plt.ylabel("Force (Hartree/Å)")
    plt.title("H2 Force vs Distance: Classical vs Quantum")
    plt.grid(True)
    plt.legend()
    
    os.makedirs(f"logs/{timestamp}", exist_ok=True)
    output_path = f"logs/{timestamp}/_h2_force_curve_comparison.png"
    plt.savefig(output_path)
    logger.info("Comparison force curve saved to %s", output_path)


def main() -> None:
    """Entry point for the H2 energy demo."""
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # プロジェクト内ロガーだけ INFO にする
    for name in ("h2_energy_demo", "h2_dynamics", "h2_energy", "h2_energy_statevector", "h2_helpers"):
        logging.getLogger(name).setLevel(logging.INFO)
    timestamp = time.strftime("%y%m%d%H%M")

    r_angstrom = 0.735
    r_bohr = r_angstrom * ANGSTROM_TO_BOHR
    e = energy_classical_bohr(r_bohr)
    logger.info("Classical Full CI energy at R=%.3f Å: %.12f Ha", r_angstrom, e)

    dynamics_seq(timestamp)


if __name__ == "__main__":
    main()