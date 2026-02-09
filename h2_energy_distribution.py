"""Statistical sampling of VQE energies for H2 at a fixed bond length."""
from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

from h2_energy_statevector import compute_h2_energy_quantum_statevector

__all__ = [
    "main",
    "parse_args",
    "run_batch",
]

logger = logging.getLogger(__name__)


def run_batch(
    n_samples: int,
    distance_angstrom: float,
    backend_name: str,
    timestamp: str,
    batch_id: int,
) -> list[float]:
    """Run a batch of VQE energy calculations in a single worker.

    Args:
        n_samples: Number of energy samples to compute.
        distance_angstrom: H-H distance in Ångströms.
        backend_name: IBM backend name for QiskitRuntimeService.
        timestamp: Log-directory timestamp string (YYMMDDHHmm).
        batch_id: Integer identifier for this batch.

    Returns:
        List of computed energies in Hartree.
    """
    logger.info("Batch %d starting: %d samples.", batch_id, n_samples)

    service = QiskitRuntimeService()
    if backend_name:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(operational=True, simulator=False)

    energies: list[float] = []
    for i in range(n_samples):
        result = compute_h2_energy_quantum_statevector(
            distance_angstrom=distance_angstrom,
            timestamp=timestamp,
            basis="sto-3g",
            ansatz_reps=1,
        )
        # 戻り値は (energy, force) タプル
        energy = result[0] if isinstance(result, tuple) else result
        energies.append(energy)

    logger.info("Batch %d finished: %d energies collected.", batch_id, len(energies))
    return energies


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the distribution sampler.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="H2 Energy Distribution")
    parser.add_argument(
        "--distance",
        type=float,
        default=0.735,
        help="Interatomic distance in Ångströms.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Total number of VQE samples.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="IBM Quantum backend name.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the H2 energy distribution sampler."""
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    for name in ("h2_energy_distribution", "h2_energy_statevector", "h2_helpers"):
        logging.getLogger(name).setLevel(logging.INFO)

    timestamp = time.strftime("%y%m%d%H%M")

    # バックエンド解決（メインプロセスで一度だけ）
    service = QiskitRuntimeService()
    if args.backend:
        backend_name = args.backend
    else:
        logger.info("Selecting least busy backend...")
        backend = service.least_busy(operational=True, simulator=False)
        backend_name = backend.name

    logger.info(
        "Config: backend=%s, distance=%.3f Å, samples=%d, workers=%d",
        backend_name, args.distance, args.samples, args.workers,
    )

    # バッチ分割
    base_batch_size = args.samples // args.workers
    remainder = args.samples % args.workers
    batches = [
        base_batch_size + (1 if i < remainder else 0)
        for i in range(args.workers)
    ]
    batches = [b for b in batches if b > 0]
    logger.info("Batch sizes: %s", batches)

    # 並列実行
    all_energies: list[float] = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(run_batch, size, args.distance, backend_name, timestamp, i)
            for i, size in enumerate(batches)
        ]
        for fut in futures:
            all_energies.extend(fut.result())

    elapsed = time.time() - start_time
    logger.info("Collected %d samples in %.2f seconds.", len(all_energies), elapsed)

    # 結果保存
    os.makedirs(f"logs/{timestamp}", exist_ok=True)
    output_file = f"logs/{timestamp}/energy_distribution_{args.distance:.3f}.txt"
    np.savetxt(output_file, all_energies)
    logger.info("Saved energies to %s", output_file)

    # ヒストグラム描画
    plt.figure(figsize=(10, 6))
    plt.hist(all_energies, bins=50, alpha=0.7, color="blue", edgecolor="black")
    plt.title(
        f"Energy Distribution for H2 at R={args.distance} Å\n"
        f"(Backend: {backend_name}, Samples: {len(all_energies)})"
    )
    plt.xlabel("Energy (Hartree)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    mean_e = float(np.mean(all_energies))
    plt.axvline(mean_e, color="red", linestyle="dashed", linewidth=1, label=f"Mean: {mean_e:.6f}")
    plt.legend()

    plot_file = f"logs/{timestamp}/energy_distribution_{args.distance:.3f}.png"
    plt.savefig(plot_file)
    logger.info("Saved plot to %s", plot_file)
    plt.show()


if __name__ == "__main__":
    main()
