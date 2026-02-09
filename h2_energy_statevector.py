"""Utilities for evaluating the H2 ground-state energy.

This module exposes two convenience functions:

* ``compute_h2_energy_quantum`` evaluates the energy with a small VQE loop
  executed on ``AerSimulator``.
* ``compute_h2_energy_classical`` evaluates the same geometry with a
  reference Hartree–Fock calculation (PySCF).

Both helpers expect the interatomic distance in Angstrom and return the energy
in Hartree.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from pyscf import ao2mo, gto, mcscf, scf
from qiskit import QuantumCircuit,transpile
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BackendEstimatorV2
from qiskit_aer import AerSimulator
from scipy.optimize import minimize,show_options
import os
import time
from matplotlib import pyplot as plt
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper
from h2_helpers import _build_h2_force_operator, _build_h2_qubit_hamiltonian, _key
from h2_energy import compute_h2_energy_classical

def build_hf_reference_circuit(num_qubits: int, occ: Sequence[int]) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for q in occ:
        qc.x(q)  # 対応するモードを占有状態に
    return qc

def compute_h2_energy_quantum_statevector(
    distance_angstrom: float,
    timestamp: str,
    *,
    basis: str = "sto-3g",
    ansatz_reps: int = 1,
    optimizer_maxiter: int = 200,
    return_trace: bool = False,
    cholesky_tol: float = 1e-6,
    occ: Sequence[int] = (1, 3),
    random_seed: int = 42,
) -> Tuple[float,float] | Tuple[float, Sequence[float]]:
    """Estimate the H2 ground-state energy with a VQE loop on Aer."""
    hamiltonian, mol, mx, mo = _build_h2_qubit_hamiltonian(
        distance_angstrom=distance_angstrom,
        basis=basis,
        cholesky_tol=cholesky_tol,
    )

    classical_energy = compute_h2_energy_classical(distance_angstrom)

    print(f"Classical HF energy at R={distance_angstrom:.2f} Å: {classical_energy:.6f} Ha")

    force_op = _build_h2_force_operator(mol, mo)

    num_qubits = hamiltonian.num_qubits
    if num_qubits is None:
        raise ValueError("Hamiltonian does not define the number of qubits.")

    hf_circuit = build_hf_reference_circuit(int(num_qubits), occ)

    # ansatz = efficient_su2(
    #     int(num_qubits),
    #     su2_gates=["ry"],
    #     entanglement="linear",
    #     reps=ansatz_reps,
    # )
    mapper = JordanWignerMapper()
    ansatz = UCCSD(
        num_spatial_orbitals=2,
        num_particles=(1,1),
        reps=ansatz_reps,
        qubit_mapper=mapper,
    )
    np.set_printoptions(
        precision=30,    # 小数点以下18桁
        suppress=False,  # 指数表記を抑制しない（=丸め誤認を減らす）
        floatmode="maxprec_equal",  # 可能な限り情報を落とさない表示
    )

    ucc_decomposed = ansatz
    for _ in range(3):  # 必要なら 4,5 に増やす
        ucc_decomposed = ucc_decomposed.decompose()


    full_ansatz = hf_circuit.compose(ucc_decomposed)

    if full_ansatz is None:
        raise ValueError("Failed to compose the full ansatz circuit.")

    num_params = full_ansatz.num_parameters
    
    rng = np.random.default_rng(random_seed)
    # HF から ±0.1 ラジアン程度だけずらした初期値
    initial_point = np.zeros(num_params)
    backend = AerSimulator(method="statevector")
    estimator = BackendEstimatorV2(backend=backend)

    eval_chache: dict[bytes, float] = {}

    def cost_function(params: np.ndarray) -> float:
        k = _key(params)
        if k in eval_chache:
            return eval_chache[k]
        pub = (full_ansatz, [hamiltonian], [params])
        result = estimator.run(pubs=[pub]).result()  # type: ignore[arg-type]
        value = float(result[0].data.evs[0])  # type: ignore[attr-defined]
        eval_chache[k] = value
        return value
    
    def cost_function_force(params: np.ndarray) -> float:
        pub = (full_ansatz, [force_op], [params])
        result = estimator.run(pubs=[pub]).result()  # type: ignore[arg-type]
        value = float(result[0].data.evs[0])  # type: ignore[attr-defined]
        return value

    trace: List[float] = []


    def _callback(xk: np.ndarray) -> None:
        trace.append(cost_function(xk))
        # print("current parameters:", xk)

    start_time = time.time()

    print(f"Starting optimization for distance: {distance_angstrom:.2f} Å")

    res = minimize(
        cost_function,
        initial_point,
        method="COBYLA",
        callback=_callback,
        options={"maxiter": optimizer_maxiter, "disp": False},
        tol=1e-8,
    )

    cobyla_point = res.x
    shifted_step = len(trace)
    print(f"Switching to L-BFGS-B optimizer at step {shifted_step}, energy: {trace[-1]:.6f} Ha")

    res = minimize(
        cost_function,
        cobyla_point,
        method="L-BFGS-B",
        callback=_callback,
        options={"maxiter": optimizer_maxiter, "disp": False, "gtol": 1e-12, "ftol": 1e-12},
    )

    # Ensure the optimizer returned a value and append it to the trace, then take the minimum.
    if res.fun is None:
        raise ValueError("Minimization did not return a result.")
    trace.append(float(res.fun))
    result_energy = float(res.fun)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"VQE optimization Finished in {total_time:.2f} seconds. Minimum energy: {result_energy:.6f} Ha")
    

    force = cost_function_force(res.x) * -1.0

    gradient_hartree_angstrom = force

    # ヘルマン・ファインマン「力」は F = -dE/dR (勾配の逆符号)
    force_val = -1.0 * gradient_hartree_angstrom

    print(f"Computed force at optimized geometry: {force_val:.6f} Ha/Angstrom")

    os.makedirs(f"logs/{timestamp}", exist_ok=True)
    with open(f"logs/{timestamp}/h2_energy_quantum_{distance_angstrom:.2f}.txt", "w") as f:
        # Log the optimization trace
        f.write("target distance: {:.2f} Angstrom\n".format(distance_angstrom))
        for step, energy in enumerate(trace):
            f.write(f"{step},{energy:.12f}\n")
            if step == shifted_step - 1:
                f.write("# Switched from COBYLA to L-BFGS-B here\n")
        f.write(f"Minimum energy: {result_energy:.12f} Ha\n")
        f.write(f"Optimization time: {total_time:.2f} seconds\n")
        f.write(f"executed with statevector simulator\n")
        f.write(f"final x parameters: {res.x}\n")
        f.write(f"computed force at optimized geometry: {force_val:.12f} Ha/Angstrom\n")

    if return_trace:
        if not trace and res.x is not None:
            trace.append(cost_function(res.x))
        return result_energy, tuple(trace)
    return result_energy, force_val
