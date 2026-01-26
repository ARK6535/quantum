import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from h2_helpers import (
    AMU_TO_KG,
    ANGSTROM_TO_METER,
    FS_TO_SECOND,
    HARTREE_TO_JOULE,
    compute_h2_energy_classical,
    parse_log_files,
)


def morse_potential(r, De, a, re, E0):
    """
    Morse potential function.
    V(r) = De * (1 - exp(-a * (r - re)))^2 + E0
    """
    return De * (1 - np.exp(-a * (r - re)))**2 + E0


def morse_force(r, De, a, re):
    """
    Calculate force from Morse potential.
    F(r) = -dV/dr
    V(r) = De * (1 - exp(-a * (r - re)))^2 + E0
    dV/dr = 2 * De * (1 - exp(-a * (r - re))) * (a * exp(-a * (r - re)))
    F(r) = -2 * De * a * (1 - exp(-a * (r - re))) * exp(-a * (r - re))
    """
    term = np.exp(-a * (r - re))
    return -2 * De * a * (1 - term) * term


def compute_classical_force(r, delta=1e-4):
    """
    Calculate force from classical energy using finite difference.
    F(r) = -dE/dr
    """
    e_plus = compute_h2_energy_classical(r + delta)
    e_minus = compute_h2_energy_classical(r - delta)
    return -(e_plus - e_minus) / (2 * delta)

def simulate_h2_vibration_general(
    force_func,
    r0,
    v0=0.0,
    dt=0.02,
    steps=1000,
    mass_amu=1.00784,
):
    """
    force_func: function r -> force [Hartree / Angstrom]
    """

    mu_amu = mass_amu / 2.0
    mu_kg = mu_amu * AMU_TO_KG

    r = r0
    v = v0

    v_si = v * 1.0e5
    r_si = r * ANGSTROM_TO_METER
    dt_si = dt * FS_TO_SECOND

    times, positions, energies = [], [], []

    # Initial force
    f_ha_a = force_func(r)
    f_newton = f_ha_a * (HARTREE_TO_JOULE / ANGSTROM_TO_METER)
    a_si = f_newton / mu_kg

    for i in range(steps):
        times.append(i * dt)
        positions.append(r)

        ke_joule = 0.5 * mu_kg * v_si**2
        ke_ha = ke_joule / HARTREE_TO_JOULE
        energies.append(ke_ha)

        # Verlet
        r_si_new = r_si + v_si * dt_si + 0.5 * a_si * dt_si**2
        r_new = r_si_new / ANGSTROM_TO_METER

        f_ha_a_new = force_func(r_new)
        f_newton_new = f_ha_a_new * (HARTREE_TO_JOULE / ANGSTROM_TO_METER)
        a_si_new = f_newton_new / mu_kg

        v_si_new = v_si + 0.5 * (a_si + a_si_new) * dt_si

        r, r_si = r_new, r_si_new
        v_si = v_si_new
        a_si = a_si_new

    return times, positions, energies

def morse_force_wrapper(De, a, re):
    return lambda r: morse_force(r, De, a, re)
def classical_force_wrapper(delta=1e-4):
    return lambda r: compute_classical_force(r, delta=delta)


def fit_morse_potential(distances, energies):
    """
    Fit energy data to Morse potential.
    """
    distances = np.array(distances)
    energies = np.array(energies)
    
    # Initial guesses
    # Find the index of the minimum energy
    min_idx = np.argmin(energies)
    E0_guess = energies[min_idx]
    re_guess = distances[min_idx]
    
    # Estimate De (Depth)
    # Assuming the last point is close to dissociation limit if it's far enough
    # Or just use a reasonable default for H2 (~0.17 Hartree) if data is insufficient
    if len(energies) > 0:
        De_guess = abs(energies[-1] - E0_guess)
        if De_guess < 0.01: # If flat or not enough range
            De_guess = 0.2
    else:
        De_guess = 0.2
        
    a_guess = 1.0 # Width parameter

    p0 = [De_guess, a_guess, re_guess, E0_guess]
    
    try:
        popt, pcov = curve_fit(morse_potential, distances, energies, p0=p0, maxfev=10000)
        return popt
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None


def plot_data(data, output_file, datetime):
    if not data:
        print("No data to plot.")
        return

    distances = [x[0] for x in data]
    vqe_energies = [x[1] for x in data]
    hf_energies_from_log = [x[2] for x in data]

    classical_energies = []
    for d in distances:
        e = compute_h2_energy_classical(d)
        classical_energies.append(e)

    # Fit Morse potential to VQE energies
    popt = fit_morse_potential(distances, vqe_energies)

    # --- Main Energy Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(distances, vqe_energies, "o-", label="VQE Energy")
    # plt.plot(distances, hf_energies_from_log, "s-.", label="HF Energy")
    plt.plot(distances, classical_energies, "x--", label="Full CI Energy")

    if popt is not None:
        De, a, re, E0 = popt
        print(f"Morse Potential Fit Parameters: De={De:.4f}, a={a:.4f}, re={re:.4f}, E0={E0:.4f}")
        
        # Generate smooth curve for plotting
        r_smooth = np.linspace(min(distances), max(distances), 200)
        v_smooth = morse_potential(r_smooth, *popt)
        # plt.plot(r_smooth, v_smooth, "r-", linewidth=2, label=f"Morse Fit (re={re:.2f})")

    plt.xlabel("Distance (Angstrom)", fontsize=20)
    plt.ylabel("Energy (Hartree)", fontsize=20)
    plt.title(f"H2 Molecule Energy vs Distance", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.legend(fontsize=16)
    plt.tight_layout()

    plt.savefig(f"logs/{datetime}/{output_file}")
    print(f"Plot saved to logs/{datetime}/{output_file}")
    plt.close()

    # --- Extra Plots if Fit Successful ---
    if popt is not None:
        De, a, re, E0 = popt
        r_smooth = np.linspace(min(distances), max(distances), 200)
        
        # 1. Force Plot
        plt.figure(figsize=(10, 6))
        f_smooth = morse_force(r_smooth, De, a, re)
        plt.plot(r_smooth, f_smooth, "g-", linewidth=2, label="Force (Morse Fit)")
        
        # Calculate classical force for comparison
        print("Calculating classical force curve (this may take a moment)...")
        f_classical = [compute_classical_force(r) for r in r_smooth]
        plt.plot(r_smooth, f_classical, "b--", linewidth=1.5, label="Force (Classical Finite Diff)")

        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.xlabel("Distance (Angstrom)")
        plt.ylabel("Force (Hartree/Angstrom)")
        plt.title(f"H2 Molecule Force vs Distance (Comparison)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"logs/{datetime}/_h2_force_curve_derived.png")
        print(f"Force plot saved to logs/{datetime}/_h2_force_curve_derived.png")
        plt.close()
        
        # 2. Molecular Dynamics Simulation
        print("Running molecular dynamics simulations (VQE vs Classical)...")

        r0 = 0.8
        v0 = 0.0

        # VQE (Morse)
        force_morse = morse_force_wrapper(De, a, re)
        t_m, r_m, e_m = simulate_h2_vibration_general(
            force_morse, r0, v0
        )

        # Classical
        force_classical = classical_force_wrapper()
        t_c, r_c, e_c = simulate_h2_vibration_general(
            force_classical, r0, v0
        )

        plt.figure(figsize=(10, 6))
        plt.plot(t_m, r_m, label="VQE-derived (Morse)")
        plt.plot(t_c, r_c, "--", label="Classical (HF finite diff)")
        plt.xlabel("Time (fs)")
        plt.ylabel("Distance (Angstrom)")
        plt.title("H2 Vibration: VQE vs Classical")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"logs/{datetime}/_h2_dynamics_comparison.png")
        plt.close()

        # 結果をテキストで保存
        with open(f"logs/{datetime}/_h2_dynamics_data.txt", "w") as f:
            f.write("Time_fs,VQE_Distance_Angstrom,Classical_Distance_Angstrom\n")
            for i in range(len(t_m)):
                f.write(f"{t_m[i]},{r_m[i]},{r_c[i]}\n")
        print(f"Dynamics data saved to logs/{datetime}/_h2_dynamics_data.txt")



def plot_energy_vs_distance(datetime):
    log_dir = f"logs/{datetime}"
    output_file = f"_energy_vs_distance_{datetime}.png"

    base_dir = os.getcwd()
    abs_log_dir = os.path.join(base_dir, log_dir)

    if not os.path.exists(abs_log_dir):
        print(f"Error: Directory {abs_log_dir} does not exist.")
        return

    data = parse_log_files(abs_log_dir)
    plot_data(data, output_file, datetime)
    # プロット対象のデータをテキストでも保存
    with open(f"logs/{datetime}/_energy_vs_distance_{datetime}.txt", "w") as f:
        f.write("Distance(Angstrom),VQE_Energy(Hartree),HF_Energy_from_Log(Hartree),Classical_Energy(Hartree),Error(%)\n")
        error_percents = []
        for i in range(len(data)):
            distance = data[i][0]
            vqe_energy = data[i][1]
            hf_energy = data[i][2]
            classical_energy = compute_h2_energy_classical(distance)
            error_percent = abs((vqe_energy - classical_energy) / classical_energy) * 100 if classical_energy != 0 else 0.0
            error_percents.append(error_percent)
            f.write(f"{distance},{vqe_energy},{hf_energy},{classical_energy},{error_percent}\n")
            if vqe_energy < classical_energy:
                print(f"Warning: At distance {distance} Å, VQE energy {vqe_energy} Ha is lower than classical energy {classical_energy} Ha.")
        
        f.write("Average Error(%):,{:.6f}\n".format(sum(error_percents)/len(error_percents) if error_percents else 0.0))
        f.write("max error(%):,{:.6f}\n".format(max(error_percents) if error_percents else 0.0))
        f.write("min error(%):,{:.6f}\n".format(min(error_percents) if error_percents else 0.0))


if __name__ == "__main__":
    plot_energy_vs_distance("2511301524")
