"""Plot VQE and Full CI energy vs internuclear distance from log files."""
from __future__ import annotations

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from h2_helpers import compute_h2_energy_classical, parse_log_files

__all__ = [
    "plot_data",
    "plot_energy_vs_distance",
]

logger = logging.getLogger(__name__)


def plot_data(
    data: list[tuple[float, float, float, float | None]],
    output_file: str,
    log_dir: str,
) -> None:
    """Plot VQE and Full CI energy curves and save the figure.

    Args:
        data: Parsed log data as returned by ``parse_log_files``.
            Each element is (distance, vqe_energy, hf_energy, force).
        output_file: Filename for the output PNG.
        log_dir: Directory path where the plot will be saved.
    """
    if not data:
        logger.warning("No data to plot.")
        return

    distances = [x[0] for x in data]
    vqe_energies = [x[1] for x in data]
    classical_energies = [compute_h2_energy_classical(d) for d in distances]

    plt.figure(figsize=(10, 6))
    plt.plot(distances, vqe_energies, "o-", label="VQE Energy")
    plt.plot(distances, classical_energies, "x--", label="Full CI Energy")
    plt.xlabel("Distance (Angstrom)", fontsize=20)
    plt.ylabel("Energy (Hartree)", fontsize=20)
    plt.title("H2 Molecule Energy vs Distance", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.legend(fontsize=16)
    plt.tight_layout()

    output_path = os.path.join(log_dir, output_file)
    plt.savefig(output_path)
    logger.info("Plot saved to %s", output_path)
    plt.close()


def plot_energy_vs_distance(log_dir: str) -> None:
    """Load log files, plot energy vs distance, and write a summary table.

    Args:
        log_dir: Absolute or relative path to the log directory.
    """
    abs_log_dir = os.path.abspath(log_dir)
    datetime_stamp = os.path.basename(abs_log_dir)
    output_file = f"_energy_vs_distance_{datetime_stamp}.png"

    if not os.path.exists(abs_log_dir):
        logger.error("Directory %s does not exist.", abs_log_dir)
        return

    data = parse_log_files(abs_log_dir)
    plot_data(data, output_file, abs_log_dir)

    # プロット対象のデータをテキストでも保存
    summary_path = os.path.join(
        abs_log_dir, f"_energy_vs_distance_{datetime_stamp}.txt",
    )
    with open(summary_path, "w") as f:
        f.write(
            "Distance(Angstrom),VQE_Energy(Hartree),"
            "HF_Energy_from_Log(Hartree),Classical_Energy(Hartree),Error(%)\n"
        )
        error_percents: list[float] = []
        for row in data:
            distance = row[0]
            vqe_energy = row[1]
            hf_energy = row[2]
            classical_energy = compute_h2_energy_classical(distance)
            error_percent = (
                abs((vqe_energy - classical_energy) / classical_energy) * 100
                if classical_energy != 0
                else 0.0
            )
            error_percents.append(error_percent)
            f.write(
                f"{distance},{vqe_energy},{hf_energy},"
                f"{classical_energy},{error_percent}\n"
            )
            if vqe_energy < classical_energy:
                logger.warning(
                    "At distance %.4f Å, VQE energy %.6f Ha "
                    "is lower than classical energy %.6f Ha.",
                    distance,
                    vqe_energy,
                    classical_energy,
                )

        avg_error = (
            sum(error_percents) / len(error_percents) if error_percents else 0.0
        )
        max_error = max(error_percents) if error_percents else 0.0
        min_error = min(error_percents) if error_percents else 0.0
        f.write(f"Average Error(%):,{avg_error:.6f}\n")
        f.write(f"max error(%):,{max_error:.6f}\n")
        f.write(f"min error(%):,{min_error:.6f}\n")

    logger.info("Summary table saved to %s", summary_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace with ``log_dir``.
    """
    parser = argparse.ArgumentParser(
        description="Plot VQE and Full CI energy vs distance from log files.",
    )
    parser.add_argument(
        "--log-dir",
        required=True,
        help="Path to the log directory (e.g. logs/2511301524).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the energy-vs-distance plotting script."""
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger(__name__).setLevel(logging.INFO)

    plot_energy_vs_distance(args.log_dir)


if __name__ == "__main__":
    main()
