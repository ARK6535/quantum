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
from h2_helpers import ANGSTROM_TO_BOHR, compute_h2_energy_classical, save_run_config

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
        "--initial-distance",
        type=float,
        default=0.8,
        help="Initial H-H distance in Ångströms (default: 0.8).",
    )
    parser.add_argument(
        "--initial-velocity",
        type=float,
        default=0.0,
        help="Initial relative velocity in Å/fs (default: 0.0).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Integration time step in femtoseconds (default: 0.01).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=1000,
        help="Number of Velocity-Verlet steps (default: 1000).",
    )
    parser.add_argument(
        "--basis",
        default="sto-3g",
        help="Gaussian basis set name (default: sto-3g).",
    )
    parser.add_argument(
        "--ansatz-reps",
        type=int,
        default=1,
        help="Number of ansatz repetition layers (default: 1).",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=2000,
        help="Maximum optimiser iterations per stage (default: 2000).",
    )
    parser.add_argument(
        "--backend-name",
        default=None,
        help="IBM Quantum backend to mimic with AerSimulator (defaults to least busy real backend).",
    )
    parser.add_argument(
        "--backend-type",
        choices=["statevector", "noisy"],
        default="statevector",
        help="VQE backend type: 'statevector' (noiseless) or 'noisy' (default: statevector).",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=4096,
        help="Number of shots for noisy energy evaluation (default: 4096).",
    )
    parser.add_argument(
        "--force-shots",
        type=int,
        default=16384,
        help="Number of shots for noisy force evaluation (default: 16384).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="LOG_DIR",
        help="Resume from a previous run's log directory (e.g. logs/2602091619).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
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
    log_level = logging.DEBUG if args.verbose else logging.INFO
    for name in ("h2_energy_demo", "h2_dynamics", "h2_energy", "h2_energy_statevector", "h2_energy_noisy", "h2_helpers"):
        logging.getLogger(name).setLevel(log_level)

    # --resume 時は前回のタイムスタンプを引き継ぐ
    if args.resume:
        import json as _json

        ckpt_file = os.path.join(args.resume, "checkpoint.json")
        if not os.path.isfile(ckpt_file):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
        with open(ckpt_file) as f:
            ckpt = _json.load(f)
        timestamp = ckpt["timestamp"]
        log_dir = f"logs/{timestamp}"
        logger.info("Resuming run from %s (timestamp=%s)", args.resume, timestamp)
    else:
        timestamp = time.strftime("%y%m%d%H%M")
        log_dir = f"logs/{timestamp}"

    save_run_config(
        log_dir,
        args,
        backend_type=args.backend_type,
        extra={
            "cholesky_tol": 1e-10,
            "h_bohr_finite_diff": 0.01,
            "resumed_from": args.resume,
            "shots": args.shots,
            "force_shots": args.force_shots,
        },
    )

    if not args.resume:
        r_angstrom = args.initial_distance
        r_bohr = r_angstrom * ANGSTROM_TO_BOHR
        e = energy_classical_bohr(r_bohr, basis=args.basis)
        logger.info("Classical Full CI energy at R=%.3f Å: %.12f Ha", r_angstrom, e)

    dynamics_seq(
        timestamp,
        initial_r_angstrom=args.initial_distance,
        initial_v_ang_per_fs=args.initial_velocity,
        time_step_fs=args.dt,
        total_step=args.n_steps,
        basis=args.basis,
        ansatz_reps=args.ansatz_reps,
        optimizer_maxiter=args.maxiter,
        resume_from=args.resume,
        backend_type=args.backend_type,
        shots=args.shots,
        force_shots=args.force_shots,
    )


if __name__ == "__main__":
    main()