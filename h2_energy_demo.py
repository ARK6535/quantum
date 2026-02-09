"""Quick demo comparing classical and VQE energies for H2."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import math

from h2_energy import compute_h2_energy_classical, compute_h2_energy_quantum, debug_compare_hamiltonian_vs_fci
from h2_energy_statevector import compute_h2_energy_quantum_statevector
from h2_helpers import A0, MU, T_AU_FS
import numpy as np

from typing import Dict, List, Tuple, Union, Callable, Any

from matplotlib import pyplot as plt


import time
import os
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

import plot_energy_vs_distance

def energy_classical_R_bohr(R_bohr: float) -> float:
    """R [Bohr] を受け取って classical エネルギー [Hartree] を返すヘルパー。"""
    R_ang = R_bohr * A0
    return compute_h2_energy_classical(R_ang, basis="sto-3g")

def energy_quantum_R_bohr(R_bohr: float, timestamp: str, backend: Any | None = None) -> float:
    """R [Bohr] を受け取って quantum エネルギー [Hartree] を返すヘルパー。計算手法を変えたい場合はここのエネルギー計算関数を差し替える。"""
    R_ang = R_bohr * A0
    val = compute_h2_energy_quantum_statevector(
        R_ang,
        basis="sto-3g",
        timestamp=timestamp,
        ansatz_reps=1,
        optimizer_maxiter=2000,
        cholesky_tol=1e-10,
    )
    if isinstance(val, tuple):
        return val[0]
    return val

def force_quantum_R_bohr(R_bohr: float, timestamp: str, h_bohr: float = 0.01, backend: Any | None = None,) -> float:
    """量子計算による力（数値微分）。"""
    e_plus = energy_quantum_R_bohr(R_bohr + h_bohr, backend=backend, timestamp=timestamp)
    e_minus = energy_quantum_R_bohr(R_bohr - h_bohr, backend=backend, timestamp=timestamp)
    dEdR = (e_plus - e_minus) / (2.0 * h_bohr)
    return -dEdR

def force_quantum_R_ang(R_ang: float, timestamp: str, backend: Any | None = None) -> float:
    R_bohr = R_ang / A0
    F_bohr = force_quantum_R_bohr(R_bohr, backend=backend, timestamp=timestamp)
    return F_bohr / A0

def force_classical_R_bohr(R_bohr: float, h_bohr: float = 0.01) -> float:
    """
    R [Bohr] における力 F(R) を返す (Classical Full CI)。
    ここで F は R 方向の力で、単位は Hartree/Bohr（原子単位の力）。
    """
    e_plus = energy_classical_R_bohr(R_bohr + h_bohr)
    e_minus = energy_classical_R_bohr(R_bohr - h_bohr)
    dEdR = (e_plus - e_minus) / (2.0 * h_bohr)  # Hartree/Bohr
    F = -dEdR  # F = -dE/dR
    return F

def force_classical_R_ang(R_ang: float) -> float:
    """
    R [Å] における力 F(R) を返す (Classical Full CI)。
    単位は Hartree/Å。
    """
    R_bohr = R_ang / A0
    F_bohr = force_classical_R_bohr(R_bohr)
    return F_bohr / A0

def fs_to_au(t_fs: float) -> float:
    return t_fs / T_AU_FS

def au_to_fs(t_au: float) -> float:
    return t_au * T_AU_FS
def v_angfs_to_au(v_ang_per_fs: float) -> float:
    return v_ang_per_fs * (T_AU_FS / A0)

ArrayLike = Union[float, np.ndarray]

def v_au_to_angfs(v_au: ArrayLike) -> ArrayLike:
    return v_au * (A0 / T_AU_FS)

def simulate_h2_1d(
    R0_angstrom: float,
    v_rel0_ang_per_fs: float,
    t_final_fs: float,
    dt_fs: float,
    force_func: Callable[[float], float] = force_classical_R_bohr,
) -> tuple[np.ndarray, np.ndarray, ArrayLike]:
    """
    H2 の 1次元相対運動をシミュレートする。
    - R0_angstrom : 初期核間距離 [Å]
    - v_rel0_ang_per_fs : 相対座標の初期速度 dR/dt [Å/fs]
    - t_final_fs : 総シミュレーション時間 [fs]
    - dt_fs : タイムステップ [fs]
    - force_func : R [Bohr] を受け取り、力 [Hartree/Bohr] を返す関数

    戻り値:
    t_fs : 時間 [fs]
    R_ang : 核間距離 [Å]
    v_rel_ang_per_fs : 相対速度 [Å/fs]
    """
    # --- 単位変換 ---
    R0_bohr = R0_angstrom / A0
    v0_au = v_angfs_to_au(v_rel0_ang_per_fs)
    dt_au = fs_to_au(dt_fs)
    n_steps = int(t_final_fs / dt_fs) + 1

    # 配列確保
    t_fs = np.linspace(0.0, t_final_fs, n_steps)
    R_bohr = np.zeros(n_steps)
    v_au = np.zeros(n_steps)

    # 初期条件
    R_bohr[0] = R0_bohr
    v_au[0] = v0_au

    # 初期力・加速度
    F0 = force_func(R_bohr[0])      # Hartree/Bohr = a.u.の力
    a0 = F0 / MU                      # 加速度 [Bohr / (a.u.^2)]

    # velocity Verlet
    a_old = a0
    for i in range(1, n_steps):
        # 位置の更新
        R_bohr[i] = R_bohr[i-1] + v_au[i-1]*dt_au + 0.5*a_old*(dt_au**2)

        # 新しい力・加速度
        F_new = force_func(R_bohr[i])
        a_new = F_new / MU

        # 速度の更新fre
        v_au[i] = v_au[i-1] + 0.5*(a_old + a_new)*dt_au

        a_old = a_new

    # 最後に Å, Å/fs に戻す
    R_ang = R_bohr * A0
    v_rel_ang_per_fs = v_au_to_angfs(v_au)

    return t_fs, R_ang, v_rel_ang_per_fs


def plot_dynamics_results(
    results: list[tuple[np.ndarray, np.ndarray, ArrayLike, str]],
    timestamp: str,
    title_suffix: str = ""
) -> None:
    """
    シミュレーション結果をまとめてプロットする。
    results: list of (t_fs, R_ang, v_rel, label)
    timestamp: 保存先フォルダ名に使用するタイムスタンプ
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

def plot_force_curve(timestamp: str, start_ang: float = 0.5, end_ang: float = 3.0, step_ang: float = 0.1) -> None:
    """R=start_angからend_angまでstep_ang刻みで力F(R)を計算してプロットする。"""
    distances_ang = np.arange(start_ang, end_ang + 1e-9, step_ang)
    forces = []

    print(f"Calculating forces for R=[{start_ang}, {end_ang}] Å...")
    for r_ang in distances_ang:
        f = force_classical_R_ang(float(r_ang))
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
    print(f"Force curve saved to {output_path}")


def plot_force_curve_comparison(timestamp: str,backend_name: str = "None",start_ang: float = 0.5, end_ang: float = 3.0, step_ang: float = 0.1) -> None:
    """R=start_angからend_angまでstep_ang刻みで力F(R)を計算してプロットする（古典 vs 量子）。"""
    distances_ang = np.arange(start_ang, end_ang + 1e-9, step_ang)
    forces_classical = []
    forces_quantum = []

    print(f"Calculating forces (Classical & Quantum) for R=[{start_ang}, {end_ang}] Å...")
    for r_ang in distances_ang:
        print(f"Processing R={r_ang:.2f} Å...")
        fc = force_classical_R_ang(float(r_ang))
        forces_classical.append(fc)
    
    # Parallelize quantum force evaluations across CPU cores using ProcessPoolExecutor.
    n_cores = os.cpu_count() or 1
    n_workers = max(1, n_cores - 2)
    n_workers = min(n_workers, len(distances_ang))
    chunks = chunk_list(list(distances_ang), n_workers)

    forces_quantum = []
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # futures = [executor.submit(energy_worker, chunk, timestamp, backend_name) for chunk in chunks]
            futures = []
            for chunk in chunks:
                new_chunk = [float(r_ang) for r_ang in chunk]
                futures.append(executor.submit(energy_worker, new_chunk, timestamp, backend_name))
            # Gather results in the same chunk order to preserve ordering of distances_ang
            for fut in futures:
                forces_quantum.extend(fut.result())
    except Exception as e:
        print(f"Parallel execution failed ({e}), shutting down.")
        raise e
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
    print(f"Comparison force curve saved to {output_path}")
    plot_energy_vs_distance.plot_energy_vs_distance(timestamp)

# ===== Unit conversions =====
BOHR_TO_ANG = A0
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG

AU_TIME_TO_FS = T_AU_FS
FS_TO_AU_TIME = 1.0 / AU_TIME_TO_FS

def dynamics_seq(timestamp: str, backend: Any | None = None) -> None:
    """
    Sequential MD for H2 vibration on the 1D coordinate R (internuclear distance).

    Assumptions:
      - compute_h2_energy_quantum_statevector(R_ang, ... ) returns (E_Ha, F_Ha_per_Ang)
        where F = - dE/dR (force along increasing R).
      - Classical equation for relative coordinate:
            MU * d^2 R / dt^2 = F(R)
        in atomic units (R in Bohr, t in a.u. time, F in Ha/Bohr).

    Output:
      - writes CSV under ./results/<timestamp>/dynamics_seq.csv
    """
    # ===== Simulation parameters (user-specified) =====
    initial_R_ang = 0.8
    initial_v_ang_per_fs = 0.0
    time_step_fs = 0.01
    total_step = 1000

    # ===== Convert initial conditions to atomic units =====
    dt_au = time_step_fs * FS_TO_AU_TIME
    R_bohr = initial_R_ang * ANG_TO_BOHR
    v_bohr_per_au = initial_v_ang_per_fs * ANG_TO_BOHR / FS_TO_AU_TIME

    # ===== Prepare output =====
    out_dir = f"logs/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "dynamics_seq.csv")

    results: List[Dict[str, float]] = []

    def eval_energy_force(Rb: float) -> Tuple[float, float]:
        """Return (E_Ha, F_Ha_per_Bohr) at R=Rb (Bohr)."""
        R_ang = Rb * BOHR_TO_ANG
        energy_ha, force_ha_per_ang = compute_h2_energy_quantum_statevector(
            R_ang,
            basis="sto-3g",
            timestamp=timestamp,
            ansatz_reps=1,
            optimizer_maxiter=2000,
            cholesky_tol=1e-10,
        )
        if not isinstance(force_ha_per_ang, float):
            raise ValueError("compute_h2_energy_quantum_statevector did not return force value.")
        # Convert Ha/Å -> Ha/Bohr
        force_ha_per_bohr = force_ha_per_ang * BOHR_TO_ANG
        return float(energy_ha), float(force_ha_per_bohr)

    # ===== Initial force =====
    energy_ha, force_ha_per_bohr = eval_energy_force(R_bohr)

    # ===== Main integration loop (velocity Verlet) =====
    for i in range(total_step):
        print("step", i, "/", total_step)
        t_fs = i * time_step_fs

        # Record current state (convert back to convenient units)
        results.append(
            {
                "step": float(i),
                "t_fs": float(t_fs),
                "R_ang": float(R_bohr * BOHR_TO_ANG),
                "v_ang_per_fs": float(v_bohr_per_au * BOHR_TO_ANG * AU_TIME_TO_FS),
                "E_ha": float(energy_ha),
                "F_ha_per_ang": float(force_ha_per_bohr / BOHR_TO_ANG),
            }
        )

        # Guardrails (optional): stop if R becomes unphysical
        if R_bohr * BOHR_TO_ANG < 0.2 or R_bohr * BOHR_TO_ANG > 5.0:
            break

        # a_n
        a_bohr_per_au2 = force_ha_per_bohr / MU

        # v_{n+1/2}
        v_half = v_bohr_per_au + 0.5 * a_bohr_per_au2 * dt_au

        # R_{n+1}
        R_next = R_bohr + v_half * dt_au

        # Evaluate at new position
        energy_next, force_next = eval_energy_force(R_next)

        # a_{n+1}
        a_next = force_next / MU

        # v_{n+1}
        v_next = v_half + 0.5 * a_next * dt_au

        # Update
        R_bohr = R_next
        v_bohr_per_au = v_next
        energy_ha = energy_next
        force_ha_per_bohr = force_next

    # ===== Write CSV =====
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["step", "t_fs", "R_ang", "v_ang_per_fs", "E_ha", "F_ha_per_ang"]
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Wrote: {out_csv}")
    if results:
        last = results[-1]
        print(
            "Last state:",
            f"t={last['t_fs']:.6f} fs, R={last['R_ang']:.6f} Å, v={last['v_ang_per_fs']:.6e} Å/fs, "
            f"E={last['E_ha']:.12f} Ha, F={last['F_ha_per_ang']:.6e} Ha/Å"
        )


    

def chunk_list(lst, n):
    """リスト lst を n 個前後のチャンクに分割する簡易関数。"""
    k = math.ceil(len(lst) / n)
    return [lst[i:i+k] for i in range(0, len(lst), k)]

def energy_worker(target_distance_list: list[float], timestamp: str, backend_name: str) -> list[tuple[float, float]]:
    """指定されたエネルギーリストに対して量子計算を行うワーカー関数。"""
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    results = []
    for distance in target_distance_list:
        print(f"Starting VQE for target distance: {distance:.6f} Å")
        fq = force_quantum_R_ang(float(distance), timestamp=timestamp, backend=backend)
        results.append((fq, distance))
        print(f"Finished VQE for target distance: {distance:.6f} Å, Computed energy: {fq:.6f} Ha")
    print(f"All VQE computations for distances {target_distance_list} completed.")
    return results

def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%y%m%d%H%M")
    R_ang = 0.735
    R_bohr = R_ang / A0
    e = energy_classical_R_bohr(R_bohr)
    print(f"Classical Full CI energy at R={R_ang:.3f} Å: {e:.12f} Ha")
    # Fetch backend once
    service = QiskitRuntimeService()
    backend = service.backend("ibm_kawasaki")
    
    print(f"Using backend: {backend.name}")

    # e,f = compute_h2_energy_quantum_statevector(
    #     R_ang,
    #     basis="sto-3g",
    #     timestamp=timestamp,
    #     ansatz_reps=args.ansatz_reps,
    #     optimizer_maxiter=args.maxiter,
    #     cholesky_tol=1e-10,
    #     backend_arg=backend
    # )

    # print(f"Quantum VQE energy at R={R_ang:.3f} Å: {e:.12f} Ha, Force: {f:.6f} Ha/Å")

    # plot_force_curve_comparison(timestamp, start_ang=0.5, end_ang=3.0, step_ang=0.1, backend_name=backend.name)
    dynamics_seq(timestamp, backend=backend)

if __name__ == "__main__":
    main()