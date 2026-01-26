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
from pyscf import mcscf, scf
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.primitives import BackendEstimatorV2
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
import os
import time
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper

from h2_helpers import _build_h2_qubit_hamiltonian, _build_pass_manager, _key, compute_h2_energy_classical

__all__ = [
    "compute_h2_energy_quantum",
]

def build_hf_reference_circuit(num_qubits: int, occ: Sequence[int]) -> QuantumCircuit:
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
    """_build_h2_qubit_hamiltonian が返す qubit Hamiltonian を
    そのまま行列にして厳密対角化し、基底エネルギーを返す。
    """
    hamiltonian = _build_h2_qubit_hamiltonian(
        distance_angstrom=distance_angstrom,
        basis=basis,
        cholesky_tol=cholesky_tol,
    )

    Hmat = hamiltonian.to_matrix()       # 4qubit → 16x16 の行列
    evals = np.linalg.eigvalsh(Hmat)     # Hermitian なので eigvalsh でOK
    return float(evals[0])               # 最小固有値が基底エネルギー


def debug_compare_hamiltonian_vs_fci(
    distance_angstrom: float,
    *,
    basis: str = "sto-3g",
    cholesky_tol: float = 1e-10,
) -> None:
    """与えられた R, basis について、
    PySCF CASCI(F CI相当) エネルギーと
    qubit Hamiltonian の厳密対角化エネルギーを比較して表示する。
    """
    E_fci = compute_h2_energy_classical(
        distance_angstrom,
        basis=basis,
    )
    E_qubit = compute_h2_energy_qubit_exact(
        distance_angstrom,
        basis=basis,
        cholesky_tol=cholesky_tol,
    )

    diff = E_qubit - E_fci

    print(f"R = {distance_angstrom:.3f} Å, basis = {basis}")
    print(f"  FCI (PySCF CASCI) energy     : {E_fci:.12f} Ha")
    print(f"  Qubit Hamiltonian exact energy: {E_qubit:.12f} Ha")
    print(f"  Difference (qubit - FCI)      : {diff:+.3e} Ha")


def compute_h2_energy_quantum(
    distance_angstrom: float,
    *,
    basis: str = "sto-3g",
    ansatz_reps: int = 1,
    optimizer_maxiter: int = 200,
    random_seed: Optional[int] = 7,
    return_trace: bool = False,
    cholesky_tol: float = 1e-6,
    backend_name: Optional[str] = None,
    runtime_service: Optional[QiskitRuntimeService] = None,
    backend: Optional[Any] = None,
    timestamp: str,
    occ: Sequence[int] = (1, 3),
) -> float | Tuple[float, Sequence[float]]:
    """Estimate the H2 ground-state energy with the VQE stack used in ``vqe_LiH``.

    The circuit is transpiled to the selected backend's ISA, but the execution
    always happens on ``AerSimulator.from_backend`` so that only a noisy
    simulator is used (no real hardware shots).
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
        num_particles=(1,1),
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

    eval_chache: dict[bytes, float] = {}

    def cost_function(params: np.ndarray) -> float:
        k = _key(params)
        if k in eval_chache:
            return eval_chache[k]
        pub = (ansatz_isa, [hamiltonian_isa], [params])
        result = estimator.run(pubs=[pub]).result()  # type: ignore[arg-type]
        value = float(result[0].data.evs[0])  # type: ignore[attr-defined]
        eval_chache[k] = value
        return value

    trace: List[float] = []

    def _callback(xk: np.ndarray) -> None:
        trace.append(cost_function(xk))
        print(f"dist: {distance_angstrom:.6f} Current energy: {trace[-1]:.6f} Ha")
    
    start_time = time.time()

    print(f"Starting optimization for distance: {distance_angstrom:.2f} Å")

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
    print(f"Switching to L-BFGS-B optimizer at step {shifted_step}, energy: {trace[-1]:.6f} Ha,distance: {distance_angstrom:.6f} Å")

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

    print(f"VQE optimization for {distance_angstrom:.2f} Å Finished in {total_time:.2f} seconds")

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



