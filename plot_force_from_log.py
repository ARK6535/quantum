import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from h2_helpers import compute_h2_energy_classical, parse_log_files

from read_csv import compute_classical_force


# Hardcode the log datetime; edit this value when switching targets
LOG_DATETIME = "2601060722"


def compute_force(distances: np.ndarray, energies: np.ndarray) -> np.ndarray:
    """Compute force as negative numerical derivative of energy.

    Uses central differences inside the interval and one-sided differences at the edges.
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
            slope = (energies[i + 1] - energies[i - 1]) / (distances[i + 1] - distances[i - 1])
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
    plt.figure(figsize=(8, 5))
    plt.plot(distances, forces, "o-", label=f"Force from {label} energy")
    plt.plot(distances, classical_forces, "s--", label="Full CI energy")
    plt.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Distance (Angstrom)",fontsize=20)
    plt.ylabel("Force (Hartree/Angstrom)",fontsize=20)
    plt.title(f"H2 Force vs Distance",fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    print(f"Saved plot to {output_png}")
    plt.close()


def write_force_table(
    output_txt: str,
    distances: np.ndarray,
    forces: np.ndarray,
    classical_forces: np.ndarray,
) -> None:
    with open(output_txt, "w") as f:
        f.write("Distance(Angstrom),Force_Selected(Ha/A),Force_Classical(Ha/A)\n")
        for r, f_sel, f_cls in zip(distances, forces, classical_forces):
            f.write(f"{r},{f_sel},{f_cls}\n")
    print(f"Saved force table to {output_txt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot distance-force curve from energy_vs_distance log")
    parser.add_argument(
        "--column",
        choices=["vqe", "hf", "classical"],
        default="vqe",
        help="Energy column to differentiate for force calculation.",
    )
    parser.add_argument(
        "--output",
        help="Optional output PNG path. Defaults to the same directory as the log.",
    )
    args = parser.parse_args()

    log_dir = os.path.join("logs", LOG_DATETIME)
    data = parse_log_files(log_dir)

    distances = np.array([row[0] for row in data], dtype=float)
    vqe_energy = np.array([row[1] for row in data], dtype=float)
    hf_energy = np.array([row[2] for row in data], dtype=float)
    forces = np.array([row[3] for row in data], dtype=float)


    classical_energy = np.array([compute_h2_energy_classical(d) for d in distances], dtype=float)

    if args.column == "vqe":
        energies = vqe_energy
    elif args.column == "hf":
        energies = hf_energy
    else:
        energies = classical_energy

    classical_forces = []

    for i in range(len(distances)):
        classical_forces.append(compute_classical_force(distances[i]))

    out_dir = os.path.abspath(log_dir)
    suffix = args.column
    default_png = os.path.join(out_dir, f"force_vs_distance_{suffix}.png")
    default_txt = os.path.join(out_dir, f"force_vs_distance_{suffix}.txt")

    output_png = args.output or default_png
    output_txt = os.path.join(out_dir, f"force_vs_distance_{suffix}.txt") if not args.output else os.path.splitext(args.output)[0] + ".txt"

    source_label = os.path.basename(out_dir)
    plot_force(distances, forces, classical_forces, args.column.upper(), output_png, source_label)
    write_force_table(output_txt, distances, forces, classical_forces)


if __name__ == "__main__":
    main()
