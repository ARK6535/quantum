"""Noiseless statevector VQE energy and force computation for H2.

Evaluates the H2 ground-state energy and Hellmann-Feynman force using a
VQE loop on the Aer statevector simulator (no noise model).
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import BackendEstimatorV2
from qiskit_aer import AerSimulator
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper
from scipy.optimize import minimize

from h2_helpers import (
    _build_h2_force_operator,
    _build_h2_qubit_hamiltonian,
    _key,
    compute_h2_energy_classical,
)

__all__ = [
    "compute_h2_energy_quantum_statevector",
]

logger = logging.getLogger(__name__)

def build_hf_reference_circuit(num_qubits: int, occ: Sequence[int]) -> QuantumCircuit:
    """Build a Hartree-Fock reference circuit by flipping occupied qubits.

    Args:
        num_qubits: Total number of qubits in the circuit.
        occ: Indices of qubits to set to |1> (occupied modes).

    Returns:
        A QuantumCircuit with X gates applied to the occupied qubits.
    """
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
) -> tuple[float, float] | tuple[float, Sequence[float]]:
    """Estimate the H2 ground-state energy and force via statevector VQE.

    Uses a noiseless ``AerSimulator(method='statevector')`` backend with a
    two-stage optimisation (COBYLA then L-BFGS-B).  Returns the minimum
    energy and the Hellmann-Feynman force at the optimised geometry.

    Args:
        distance_angstrom: H-H distance in Ångströms.
        timestamp: Timestamp string for the log directory (YYMMDDHHmm).
        basis: Gaussian basis set name.
        ansatz_reps: Number of repetition layers for the UCCSD ansatz.
        optimizer_maxiter: Maximum iterations per optimiser stage.
        return_trace: If True, return the energy trace instead of the force.
        cholesky_tol: Tolerance for Cholesky decomposition of 2e integrals.
        occ: Qubit indices for the HF reference occupation.
        random_seed: Random seed (reserved for future use).

    Returns:
        A tuple ``(energy, force)`` in Hartree and Ha/Å respectively.
        If *return_trace* is True, returns ``(energy, trace)`` instead.
    """
    hamiltonian, mol, mx, mo = _build_h2_qubit_hamiltonian(
        distance_angstrom=distance_angstrom,
        basis=basis,
        cholesky_tol=cholesky_tol,
    )

    classical_energy = compute_h2_energy_classical(distance_angstrom)

    logger.info(
        "Classical FCI energy at R=%.2f Å: %.6f Ha",
        distance_angstrom, classical_energy,
    )

    force_op = _build_h2_force_operator(mol, mo)

    num_qubits = hamiltonian.num_qubits
    if num_qubits is None:
        raise ValueError("Hamiltonian does not define the number of qubits.")

    hf_circuit = build_hf_reference_circuit(int(num_qubits), occ)

    mapper = JordanWignerMapper()
    ansatz = UCCSD(
        num_spatial_orbitals=2,
        num_particles=(1, 1),
        reps=ansatz_reps,
        qubit_mapper=mapper,
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

    eval_cache: dict[bytes, float] = {}

    def cost_function(params: np.ndarray) -> float:
        k = _key(params)
        if k in eval_cache:
            return eval_cache[k]
        pub = (full_ansatz, [hamiltonian], [params])
        result = estimator.run(pubs=[pub]).result()  # type: ignore[arg-type]
        value = float(result[0].data.evs[0])  # type: ignore[attr-defined]
        eval_cache[k] = value
        return value

    def cost_function_force(params: np.ndarray) -> float:
        pub = (full_ansatz, [force_op], [params])
        result = estimator.run(pubs=[pub]).result()  # type: ignore[arg-type]
        value = float(result[0].data.evs[0])  # type: ignore[attr-defined]
        return value

    trace: list[float] = []


    def _callback(xk: np.ndarray) -> None:
        trace.append(cost_function(xk))
        # print("current parameters:", xk)

    start_time = time.time()

    logger.info("Starting COBYLA for distance=%.2f Å (statevector)", distance_angstrom)

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
    logger.info(
        "Switching to L-BFGS-B at step %d, energy=%.6f Ha",
        shifted_step, trace[-1],
    )

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

    logger.info(
        "VQE finished for %.2f Å in %.2f s, energy=%.6f Ha",
        distance_angstrom, total_time, result_energy,
    )
    

    force = cost_function_force(res.x) * -1.0

    gradient_hartree_angstrom = force

    # ヘルマン・ファインマン「力」は F = -dE/dR (勾配の逆符号)
    force_val = -1.0 * gradient_hartree_angstrom

    logger.info("Force at optimised geometry: %.6f Ha/Å", force_val)

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
