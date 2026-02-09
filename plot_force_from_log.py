"""Plot force vs internuclear distance from VQE energy log files."""
from __future__ import annotations

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from h2_dynamics import force_classical_angstrom
from h2_helpers import compute_h2_energy_classical, parse_log_files

__all__ = [
    "compute_force",
    "plot_force",
    "write_force_table",
]

logger = logging.getLogger(__name__)


def compute_force(distances: np.ndarray, energies: np.ndarray) -> np.ndarray:
    """Compute force as negative numerical derivative of energy.

    Uses central differences inside the interval and one-sided differences
    at the edges.

    Args:
        distances: Internuclear distances in Ångströms.
        energies: Corresponding energies in Hartree.

    Returns:
        Forces in Hartree/Ångström (negative energy gradient).
    """
    if distances.size < 2 or energies.size < 2:
        raise ValueError("Need at least two points to compute force.")

    forces = np.zeros_like(energies)
    for i in range(len(distances)):
        if i == 0:
            slope = (energies[1] - energies[0]) / (distances[1] - distances[0])
        elif i == len(distances) - 1:
            slope = (energies[-1] - energies[-2]) / (distances[-1] - distances[-2])
        else:
            slope = (
                (energies[i + 1] - energies[i - 1])
                / (distances[i + 1] - distances[i - 1])
            )
        forces[i] = -slope
    return forces


def plot_force(
    distances: np.ndarray,
    forces: np.ndarray,
    classical_forces: np.ndarray,
    label: str,
    output_png: str,
    source_label: str,
) -> None:
    """Plot force curves and save the figure.

    Args:
        distances: Internuclear distances in Ångströms.
        forces: Forces derived from the selected energy column.
        classical_forces: Full CI reference forces.
        label: Label for the selected force curve.
        output_png: Output PNG file path.
        source_label: Label identifying the data source.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(distances, forces, "o-", label=f"Force from {label} energy")
    plt.plot(distances, classical_forces, "s--", label="Full CI energy")
    plt.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Distance (Angstrom)", fontsize=20)
    plt.ylabel("Force (Hartree/Angstrom)", fontsize=20)
    plt.title("H2 Force vs Distance", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    logger.info("Saved plot to %s", output_png)
    plt.close()


def write_force_table(
    output_txt: str,
    distances: np.ndarray,
    forces: np.ndarray,
    classical_forces: np.ndarray,
) -> None:
    """Write a CSV table of forces to disk.

    Args:
        output_txt: Output text file path.
        distances: Internuclear distances in Ångströms.
        forces: Forces from the selected energy column.
        classical_forces: Full CI reference forces.
    """
    with open(output_txt, "w") as f:
        f.write("Distance(Angstrom),Force_Selected(Ha/A),Force_Classical(Ha/A)\n")
        for r, f_sel, f_cls in zip(distances, forces, classical_forces):
            f.write(f"{r},{f_sel},{f_cls}\n")
    logger.info("Saved force table to %s", output_txt)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Plot distance-force curve from energy_vs_distance log.",
    )
    parser.add_argument(
        "--log-dir",
        required=True,
        help="Path to the log directory (e.g. logs/2601060722).",
    )
    parser.add_argument(
        "--column",
        choices=["vqe", "hf", "classical"],
        default="vqe",
        help="Energy column to differentiate for force calculation.",
    )
    parser.add_argument(
        "--output",
        help="Optional output PNG path. Defaults to the log directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the force-vs-distance plotting script."""
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger(__name__).setLevel(logging.INFO)

    log_dir = args.log_dir
    data = parse_log_files(log_dir)

    distances = np.array([row[0] for row in data], dtype=float)
    vqe_energy = np.array([row[1] for row in data], dtype=float)
    hf_energy = np.array([row[2] for row in data], dtype=float)
    # ログに Hellmann-Feynman force が含まれている場合はそちらを使う
    hellmann_forces = np.array([row[3] for row in data], dtype=float)

    classical_energy = np.array(
        [compute_h2_energy_classical(d) for d in distances], dtype=float,
    )

    if args.column == "vqe":
        energies = vqe_energy
    elif args.column == "hf":
        energies = hf_energy
    else:
        energies = classical_energy

    # ログの Hellmann-Feynman force が有効ならそれを使い、
    # 無ければエネルギーの数値微分から力を計算する
    if np.all(np.isfinite(hellmann_forces)):
        forces = hellmann_forces
    else:
        forces = compute_force(distances, energies)

    classical_forces = np.array(
        [force_classical_angstrom(d) for d in distances], dtype=float,
    )

    out_dir = os.path.abspath(log_dir)
    suffix = args.column
    default_png = os.path.join(out_dir, f"force_vs_distance_{suffix}.png")

    output_png = args.output or default_png
    output_txt = (
        os.path.join(out_dir, f"force_vs_distance_{suffix}.txt")
        if not args.output
        else os.path.splitext(args.output)[0] + ".txt"
    )

    source_label = os.path.basename(out_dir)
    plot_force(
        distances, forces, classical_forces,
        args.column.upper(), output_png, source_label,
    )
    write_force_table(output_txt, distances, forces, classical_forces)


if __name__ == "__main__":
    main()
