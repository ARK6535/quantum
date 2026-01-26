from matplotlib.animation import FuncAnimation
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from h2_helpers import (
    AMU_TO_KG,
    ANGSTROM_TO_METER,
    FS_TO_SECOND,
    HARTREE_TO_JOULE,
    BOHR_TO_ANGSTROM,
    compute_h2_energy_classical,
)

# timestamp = "2601061836"
# timestamp = "2601201757"
timestamp = "2601261451"


def kinetic_from_positions(times_fs, positions_angstrom, mass_amu=1.00784):
    """Compute kinetic energy (Hartree) from position-vs-time trace using reduced mass.

    times_fs: array-like in femtoseconds
    positions_angstrom: array-like in Angstrom (internuclear distance)
    mass_amu: mass of hydrogen atom; reduced mass is mass_amu/2
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


def compute_classical_force(r, delta=1e-4):
    e_plus = compute_h2_energy_classical(r + delta)
    e_minus = compute_h2_energy_classical(r - delta)
    return -(e_plus - e_minus) / (2 * delta)


def simulate_h2_vibration_general(force_func, r0, v0=0.0, dt=0.02, steps=1000, mass_amu=1.00784):
    # Velocity Verlet in SI for stability, then convert back to Angstrom.
    mu_kg = (mass_amu / 2.0) * AMU_TO_KG

    r_si = r0 * ANGSTROM_TO_METER
    v_si = v0 * 1.0e5
    dt_si = dt * FS_TO_SECOND

    times, positions = [], []

    f_newton = force_func(r0) * (HARTREE_TO_JOULE / ANGSTROM_TO_METER)
    a_si = f_newton / mu_kg

    for i in range(steps):
        times.append(i * dt)
        positions.append(r_si / ANGSTROM_TO_METER)

        r_si_new = r_si + v_si * dt_si + 0.5 * a_si * (dt_si**2)
        r_new = r_si_new / ANGSTROM_TO_METER

        f_newton_new = force_func(r_new) * (HARTREE_TO_JOULE / ANGSTROM_TO_METER)
        a_si_new = f_newton_new / mu_kg

        v_si_new = v_si + 0.5 * (a_si + a_si_new) * dt_si

        r_si = r_si_new
        v_si = v_si_new
        a_si = a_si_new

    return times, positions

with open(f"logs/{timestamp}/dynamics_seq.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    steps = []
    Rs = []
    f_per_ang = []
    energies_vqe = []
    for row in reader:
        steps.append(float(row["step"]) * 0.01)  # fs
        Rs.append(float(row["R_ang"]))
        f_per_ang.append(float(row["F_ha_per_ang"]))  # Ha/Angstrom
        if "E_ha" in row:
            energies_vqe.append(float(row["E_ha"]))
        else:
            energies_vqe.append(np.nan)
    # 横軸: step * 0.01 / fs
    # 縦軸: R_ang / Angstrom
dt_fs = steps[1] - steps[0] if len(steps) > 1 else 0.01
os.makedirs(f"logs/{timestamp}", exist_ok=True)

# Cache classical trajectory/force to avoid recomputation.
cache_path = f"logs/{timestamp}/classical_dynamics.npz"
force_classical = lambda r: compute_classical_force(r)
if os.path.exists(cache_path):
    print("Loading cached classical dynamics data...")
    cache = np.load(cache_path)
    times_classical = cache["times_classical"]
    Rs_classical = cache["Rs_classical"]
    f_classical = cache["f_classical"]
    energies_classical = cache["energies_classical"] if "energies_classical" in cache.files else None
else:
    times_classical, Rs_classical = simulate_h2_vibration_general(
        force_classical, r0=Rs[0], v0=0.0, dt=dt_fs, steps=len(steps)
    )
    f_classical = np.array([compute_classical_force(r) for r in Rs_classical])
    energies_classical = np.array([compute_h2_energy_classical(r) for r in Rs_classical])
    np.savez(
        cache_path,
        times_classical=np.asarray(times_classical, dtype=float),
        Rs_classical=np.asarray(Rs_classical, dtype=float),
        f_classical=np.asarray(f_classical, dtype=float),
        energies_classical=np.asarray(energies_classical, dtype=float),
    )

# Backfill energies into cache if older cache file lacks them.
if "energies_classical" not in locals() or energies_classical is None:
    energies_classical = np.array([compute_h2_energy_classical(r) for r in Rs_classical])
    np.savez(
        cache_path,
        times_classical=np.asarray(times_classical, dtype=float),
        Rs_classical=np.asarray(Rs_classical, dtype=float),
        f_classical=np.asarray(f_classical, dtype=float),
        energies_classical=np.asarray(energies_classical, dtype=float),
    )

plt.figure(figsize=(10, 6))
plt.plot(steps, Rs, "o-", label="VQE")
plt.plot(times_classical, Rs_classical, "--", label="Classical")
plt.xlabel("Time (fs)")
plt.ylabel("H-H Distance (Angstrom)")
plt.title("H2 Molecular Dynamics Simulation")
plt.grid(True)
plt.legend()

if not os.path.exists(f"logs/{timestamp}"):
    os.makedirs(f"logs/{timestamp}")

plt.savefig(f"logs/{timestamp}/h2_dynamics_distance_vs_time.png", dpi=200)

print("Saved plot to h2_dynamics_distance_vs_time.png")

# Also emit a comparison-style plot like plot_energy_vs_distance.py but using the logged trajectory.
plt.figure(figsize=(10, 6))
plt.plot(steps, Rs, "o-", label="VQE")
plt.plot(times_classical, Rs_classical, "--", label="Classical")
plt.xlabel("Time (fs)")
plt.ylabel("Distance (Angstrom)")
plt.title("H2 Vibration Trajectory")
plt.grid(True)
plt.legend()
plt.savefig(f"logs/{timestamp}/_h2_dynamics_comparison.png", dpi=200)
plt.close()

print("Saved plot to _h2_dynamics_comparison.png")

plt.figure(figsize=(10, 6))
plt.plot(steps, f_per_ang, "o-", label="VQE Force")
plt.plot(times_classical, f_classical, "--", label="Classical Force")
plt.xlabel("Time (fs)")
plt.ylabel("Force (Ha/Angstrom)")
plt.title("H2 Force vs Time")
plt.grid(True)
plt.legend()
plt.savefig(f"logs/{timestamp}/h2_dynamics_force_vs_time.png", dpi=200)
plt.close()

print("Saved plot to h2_dynamics_force_vs_time.png")

# Fit VQE force vs time to a sine wave.
def _sine(t, A, omega, phi, C):
    return A * np.sin(omega * t + phi) + C

steps_arr = np.asarray(steps, dtype=float)
f_vqe_arr = np.asarray(f_per_ang, dtype=float)
mask = np.isfinite(steps_arr) & np.isfinite(f_vqe_arr)

fit_succeeded = False
if np.count_nonzero(mask) >= 4:
    t_fit = steps_arr[mask]
    y_fit = f_vqe_arr[mask]
    span = max(t_fit) - min(t_fit) if len(t_fit) > 1 else 1.0
    A0 = 0.5 * (np.nanmax(y_fit) - np.nanmin(y_fit)) if len(y_fit) else 0.1
    C0 = float(np.nanmean(y_fit)) if len(y_fit) else 0.0
    omega0 = 2 * np.pi / span if span > 0 else 1.0
    phi0 = 0.0
    try:
        popt, _ = curve_fit(_sine, t_fit, y_fit, p0=[A0, omega0, phi0, C0], maxfev=10000)
        fit_succeeded = True
    except Exception as e:
        print(f"Sine fit failed: {e}")

if fit_succeeded:
    t_dense = np.linspace(float(np.nanmin(steps_arr)), float(np.nanmax(steps_arr)), 400)
    fit_curve_vqe = _sine(t_dense, *popt)
    # Fit classical force to a sine wave as well.
    f_class_arr = np.asarray(f_classical, dtype=float)
    mask_c = np.isfinite(times_classical) & np.isfinite(f_class_arr)
    if np.count_nonzero(mask_c) >= 4:
        t_fit_c = np.asarray(times_classical, dtype=float)[mask_c]
        y_fit_c = f_class_arr[mask_c]
        span_c = max(t_fit_c) - min(t_fit_c) if len(t_fit_c) > 1 else 1.0
        A0_c = 0.5 * (np.nanmax(y_fit_c) - np.nanmin(y_fit_c)) if len(y_fit_c) else 0.1
        C0_c = float(np.nanmean(y_fit_c)) if len(y_fit_c) else 0.0
        omega0_c = 2 * np.pi / span_c if span_c > 0 else 1.0
        phi0_c = 0.0
        try:
            popt_c, _ = curve_fit(_sine, t_fit_c, y_fit_c, p0=[A0_c, omega0_c, phi0_c, C0_c], maxfev=10000)
            fit_curve_classical = _sine(t_dense, *popt_c)
        except Exception as e:
            print(f"Classical sine fit failed: {e}")
            fit_curve_classical = None
    else:
        fit_curve_classical = None

    plt.figure(figsize=(10, 6))
    plt.plot(steps_arr, f_vqe_arr, "o", label="VQE Force (data)")
    plt.plot(t_dense, fit_curve_vqe, "-", label="VQE Sine fit")
    plt.plot(times_classical, f_classical, "x", label="Classical Force (data)")
    plt.xlabel("Time (fs)")
    plt.ylabel("Force (Ha/Angstrom)")
    plt.title("Force vs Time (Sine Fits)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"logs/{timestamp}/h2_dynamics_force_vs_time_fit.png", dpi=200)
    plt.close()
    print("Saved plot to h2_dynamics_force_vs_time_fit.png")
else:
    print("Sine fit skipped (insufficient or invalid data)")

# Classical force vs distance (R, F(R)) for the classical trajectory.
plt.figure(figsize=(10, 6))
plt.plot(Rs_classical, f_classical, "o-", label="Classical Force")
plt.xlabel("H-H Distance (Angstrom)")
plt.ylabel("Force (Ha/Angstrom)")
plt.title("Classical Force vs Distance")
plt.grid(True)
plt.legend()
plt.savefig(f"logs/{timestamp}/h2_classical_force_vs_distance.png", dpi=200)
plt.close()

print("Saved plot to h2_classical_force_vs_distance.png")

# Classical energy vs distance (R, E(R)).
plt.figure(figsize=(10, 6))
plt.plot(Rs_classical, energies_classical, "o-", label="Classical Energy")
plt.plot(Rs, energies_vqe, "x--", label="VQE Energy")
plt.xlabel("H-H Distance (Angstrom)")
plt.ylabel("Energy (Hartree)")
plt.title("Classical Energy vs Distance")
plt.grid(True)
plt.legend()
plt.savefig(f"logs/{timestamp}/h2_classical_energy_vs_distance.png", dpi=200)
plt.close()

print("Saved plot to h2_classical_energy_vs_distance.png")

# Energy vs distance (VQE vs Classical) on one plot.
plt.figure(figsize=(10, 6))
plt.plot(Rs, energies_vqe, "o-", label="VQE Energy")
plt.plot(Rs_classical, energies_classical, "--", label="Classical Energy")
plt.xlabel("H-H Distance (Angstrom)")
plt.ylabel("Energy (Hartree)")
plt.title("Energy vs Distance (VQE vs Classical)")
plt.grid(True)
plt.legend()
plt.savefig(f"logs/{timestamp}/h2_energy_vs_distance.png", dpi=200)
plt.close()

print("Saved plot to h2_energy_vs_distance.png")

# Energy vs time (VQE vs Classical)
plt.figure(figsize=(10, 6))
plt.plot(steps, energies_vqe, "o-", label="VQE Energy")
plt.plot(times_classical, energies_classical, "--", label="Classical Energy")
plt.xlabel("Time (fs)")
plt.ylabel("Energy (Hartree)")
plt.title("Energy vs Time (VQE vs Classical)")
plt.grid(True)
plt.legend()
plt.savefig(f"logs/{timestamp}/h2_energy_vs_time.png", dpi=200)
plt.close()

print("Saved plot to h2_energy_vs_time.png")

# Total energy (potential + kinetic) vs time.
ke_classical = kinetic_from_positions(times_classical, Rs_classical)
ke_vqe = kinetic_from_positions(steps, Rs)
total_classical = np.asarray(energies_classical, dtype=float) + ke_classical
total_vqe = np.asarray(energies_vqe, dtype=float) + ke_vqe

plt.figure(figsize=(10, 6))
plt.plot(steps, total_vqe, "o-", label="VQE Total Energy")
plt.plot(times_classical, total_classical, "--", label="Classical Total Energy")
plt.xlabel("Time (fs)")
plt.ylabel("Energy (Hartree)")
plt.title("Total Energy vs Time (VQE vs Classical)")
plt.grid(True)
plt.legend()
plt.savefig(f"logs/{timestamp}/h2_total_energy_vs_time.png", dpi=200)
plt.close()

print("Saved plot to h2_total_energy_vs_time.png")

# Combined distance and force summary plot for quick inspection (tighter vertical footprint).
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

axes[0].plot(steps, Rs, "o-", label="VQE")
axes[0].plot(times_classical, Rs_classical, "--", label="Full CI")
axes[0].set_ylabel("H-H Distance (Angstrom)", fontsize=20)
axes[0].set_title("H2 Molecular Dynamics Simulation", fontsize=20)
axes[0].grid(True)
axes[0].legend(fontsize=20)
axes[0].tick_params(axis="y", which="major", labelsize=20)

axes[1].plot(steps, f_per_ang, "o-", label="VQE Force")
axes[1].plot(times_classical, f_classical, "--", label="Full CI Force")
axes[1].set_xlabel("Time (fs)")
axes[1].set_ylabel("Force (Ha/Angstrom)", fontsize=20)
axes[1].set_title("H2 Force vs Time", fontsize=20)
axes[1].grid(True)
axes[1].legend(fontsize=20)
axes[1].tick_params(axis="y", which="major", labelsize=20)

fig.tight_layout()
fig.savefig(f"logs/{timestamp}/h2_dynamics_summary.png", dpi=200)
plt.close(fig)

print("Saved plot to h2_dynamics_summary.png")