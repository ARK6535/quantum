"""Noisy-backend VQE energy computation for H2.

Evaluates the H2 ground-state energy using a VQE loop executed on an
``AerSimulator`` noise model derived from a real IBM backend.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import BackendEstimatorV2
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper
from scipy.optimize import minimize

from h2_helpers import (
    _build_h2_qubit_hamiltonian,
    _build_pass_manager,
    _key,
    compute_h2_energy_classical,
)

__all__ = [
    "build_hf_reference_circuit",
    "compute_h2_energy_quantum",
    "compute_h2_energy_qubit_exact",
    "debug_compare_hamiltonian_vs_fci",
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


def compute_h2_energy_qubit_exact(
    distance_angstrom: float,
    *,
    basis: str = "sto-3g",
    cholesky_tol: float = 1e-10,
) -> float:
    """Return the exact ground-state energy by diagonalising the qubit Hamiltonian.

    Args:
        distance_angstrom: H-H distance in Ångströms.
        basis: Gaussian basis set name.
        cholesky_tol: Tolerance for Cholesky decomposition of 2e integrals.

    Returns:
        Lowest eigenvalue of the qubit Hamiltonian matrix in Hartree.
    """
    hamiltonian = _build_h2_qubit_hamiltonian(
        distance_angstrom=distance_angstrom,
        basis=basis,
        cholesky_tol=cholesky_tol,
    )

    h_mat = hamiltonian.to_matrix()
    evals = np.linalg.eigvalsh(h_mat)
    return float(evals[0])


def debug_compare_hamiltonian_vs_fci(
    distance_angstrom: float,
    *,
    basis: str = "sto-3g",
    cholesky_tol: float = 1e-10,
) -> None:
    """Print a comparison of Full-CI and qubit-Hamiltonian exact energies.

    Args:
        distance_angstrom: H-H distance in Ångströms.
        basis: Gaussian basis set name.
        cholesky_tol: Tolerance for Cholesky decomposition of 2e integrals.
    """
    e_fci = compute_h2_energy_classical(distance_angstrom, basis=basis)
    e_qubit = compute_h2_energy_qubit_exact(
        distance_angstrom, basis=basis, cholesky_tol=cholesky_tol,
    )
    diff = e_qubit - e_fci

    print(f"R = {distance_angstrom:.3f} Å, basis = {basis}")
    print(f"  FCI (PySCF CASCI) energy     : {e_fci:.12f} Ha")
    print(f"  Qubit Hamiltonian exact energy: {e_qubit:.12f} Ha")
    print(f"  Difference (qubit - FCI)      : {diff:+.3e} Ha")


def compute_h2_energy_quantum(
    distance_angstrom: float,
    *,
    basis: str = "sto-3g",
    ansatz_reps: int = 1,
    optimizer_maxiter: int = 200,
    random_seed: int | None = 7,
    return_trace: bool = False,
    cholesky_tol: float = 1e-6,
    backend_name: str | None = None,
    runtime_service: QiskitRuntimeService | None = None,
    backend: Any | None = None,
    timestamp: str,
    occ: Sequence[int] = (1, 3),
) -> float | tuple[float, Sequence[float]]:
    """Estimate the H2 ground-state energy via VQE on a noisy simulator.

    The circuit is transpiled to the selected backend's ISA, then executed
    on ``AerSimulator.from_backend`` (noisy simulator, no real hardware).
    Uses a two-stage optimisation: COBYLA followed by L-BFGS-B.

    Args:
        distance_angstrom: H-H distance in Ångströms.
        basis: Gaussian basis set name.
        ansatz_reps: Number of repetition layers for the UCCSD ansatz.
        optimizer_maxiter: Maximum iterations per optimiser stage.
        random_seed: Random seed (unused, kept for interface compatibility).
        return_trace: If True, return the energy trace alongside the result.
        cholesky_tol: Tolerance for Cholesky decomposition of 2e integrals.
        backend_name: IBM backend name to derive the noise model from.
        runtime_service: Pre-initialised QiskitRuntimeService instance.
        backend: Pre-initialised backend object (takes precedence).
        timestamp: Timestamp string for the log directory (YYMMDDHHmm).
        occ: Qubit indices for the HF reference occupation.

    Returns:
        The minimum VQE energy in Hartree.  If *return_trace* is True,
        returns a tuple ``(energy, trace)``.
    """
    hamiltonian = _build_h2_qubit_hamiltonian(
        distance_angstrom=distance_angstrom,
        basis=basis,
        cholesky_tol=cholesky_tol,
    )

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
    for _ in range(3):
        ucc_decomposed = ucc_decomposed.decompose()

    full_ansatz = hf_circuit.compose(ucc_decomposed)

    if full_ansatz is None:
        raise ValueError("Failed to compose the full ansatz circuit.")

    num_params = full_ansatz.num_parameters
    # initial_point = np.zeros(num_params)
    initial_point = np.array([ 0.02261334, -0.06357566,  0.14920625])
    if backend is None:
        service = runtime_service or QiskitRuntimeService()
        backend = (
            service.backend(backend_name)
            if backend_name is not None
            else service.least_busy(operational=True, simulator=False)
        )

    pass_manager = _build_pass_manager(backend)
    ansatz_isa = pass_manager.run(full_ansatz)
    hamiltonian_isa = hamiltonian.apply_layout(ansatz_isa.layout)

    backend_sim = AerSimulator.from_backend(backend)
    estimator = BackendEstimatorV2(backend=backend_sim)

    eval_cache: dict[bytes, float] = {}

    def cost_function(params: np.ndarray) -> float:
        k = _key(params)
        if k in eval_cache:
            return eval_cache[k]
        pub = (ansatz_isa, [hamiltonian_isa], [params])
        result = estimator.run(pubs=[pub]).result()  # type: ignore[arg-type]
        value = float(result[0].data.evs[0])  # type: ignore[attr-defined]
        eval_cache[k] = value
        return value

    trace: list[float] = []

    def _callback(xk: np.ndarray) -> None:
        trace.append(cost_function(xk))
        logger.debug(
            "dist=%.4f Å  step=%d  energy=%.6f Ha",
            distance_angstrom, len(trace), trace[-1],
        )

    start_time = time.time()

    logger.info("Starting COBYLA for distance=%.2f Å", distance_angstrom)

    res = minimize(
        cost_function,
        initial_point,
        method="COBYLA",
        callback=_callback,
        options={"maxiter": optimizer_maxiter, "disp": True, "rhobeg": 0.5,"tol": 1e-8},
        tol=1e-8,
    )

    cobyla_point = res.x
    shifted_step = len(trace)
    logger.info(
        "Switching to L-BFGS-B at step %d, energy=%.6f Ha, distance=%.4f Å",
        shifted_step, trace[-1], distance_angstrom,
    )

    res = minimize(
        cost_function,
        cobyla_point,
        method="L-BFGS-B",
        callback=_callback,
        options={"maxiter": optimizer_maxiter, "disp": False, "gtol": 1e-12, "ftol": 1e-12},
    )

    if res.fun is None:
        raise ValueError("Minimization did not return a result.")
    if not trace or abs(trace[-1] - float(res.fun)) > 1e-12:
        trace.append(float(res.fun))

    result_energy = float(res.fun)

    end_time = time.time()
    total_time = end_time - start_time

    logger.info(
        "VQE finished for %.2f Å in %.2f s, energy=%.6f Ha",
        distance_angstrom, total_time, result_energy,
    )

    os.makedirs(f"logs/{timestamp}", exist_ok=True)
    with open(f"logs/{timestamp}/{distance_angstrom:.2f}_h2_energy_quantum.txt", "w") as f:
        # Log the optimization trace
        f.write("target distance: {:.2f} Angstrom\n".format(distance_angstrom))
        for step, energy in enumerate(trace):
            f.write(f"{step},{energy:.12f}\n")
            if step == shifted_step - 1:
                f.write("# Switched from COBYLA to L-BFGS-B here\n")
        f.write(f"Minimum energy: {result_energy:.12f} Ha\n")
        f.write(f"Optimization time: {total_time:.2f} seconds\n")
        f.write(f"executed with {backend.name} simulator\n")

    if return_trace:
        return result_energy, tuple(trace)
    return result_energy



