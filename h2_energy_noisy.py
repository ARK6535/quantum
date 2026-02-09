"""Noisy-backend VQE energy and force computation for H2.

Evaluates the H2 ground-state energy and Hellmann-Feynman force using a
VQE loop on an ``AerSimulator`` noise model derived from a real IBM backend.
"""
from __future__ import annotations

import logging
import math
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
    _build_h2_force_operator,
    _build_h2_qubit_hamiltonian,
    _build_pass_manager,
    _key,
    compute_h2_energy_classical,
)

__all__ = [
    "compute_h2_energy_quantum_noisy",
]

logger = logging.getLogger(__name__)


def _build_hf_reference_circuit(
    num_qubits: int,
    occ: Sequence[int],
) -> QuantumCircuit:
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


def _shots_to_precision(shots: int) -> float:
    """Convert a shot count to the Estimator precision parameter.

    The ``BackendEstimatorV2`` relation is ``shots = ceil(1 / precision**2)``,
    so precision = 1 / sqrt(shots).
    """
    return 1.0 / math.sqrt(shots)


def compute_h2_energy_quantum_noisy(
    distance_angstrom: float,
    timestamp: str,
    *,
    basis: str = "sto-3g",
    ansatz_reps: int = 1,
    optimizer_maxiter: int = 200,
    return_trace: bool = False,
    cholesky_tol: float = 1e-6,
    occ: Sequence[int] = (1, 3),
    random_seed: int | None = 7,
    shots: int = 4096,
    force_shots: int = 16384,
    backend_name: str | None = None,
    runtime_service: QiskitRuntimeService | None = None,
    backend: Any | None = None,
) -> tuple[float, float] | tuple[float, Sequence[float]]:
    """Estimate the H2 ground-state energy and force via noisy VQE.

    The circuit is transpiled to the selected backend's ISA, then executed
    on ``AerSimulator.from_backend`` (noisy simulator).  Uses a two-stage
    optimisation (COBYLA then L-BFGS-B).  Returns the minimum energy and
    the Hellmann-Feynman force at the optimised geometry.

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
        shots: Number of shots for energy evaluation during optimisation.
        force_shots: Number of shots for force expectation-value evaluation
            (higher than *shots* to reduce statistical noise on the force).
        backend_name: IBM backend name to derive the noise model from.
        runtime_service: Pre-initialised QiskitRuntimeService instance.
        backend: Pre-initialised backend object (takes precedence over
            *backend_name* and *runtime_service*).

    Returns:
        A tuple ``(energy, force)`` in Hartree and Ha/Å respectively.
        If *return_trace* is True, returns ``(energy, trace)`` instead.
    """
    # ハミルトニアン・力演算子を構築
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

    # HF 参照回路 + UCCSD アンザッツ
    hf_circuit = _build_hf_reference_circuit(int(num_qubits), occ)

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
    initial_point = np.zeros(num_params)

    # --- ノイジーバックエンド取得 ---
    if backend is None:
        service = runtime_service or QiskitRuntimeService()
        backend = (
            service.backend(backend_name)
            if backend_name is not None
            else service.least_busy(operational=True, simulator=False)
        )

    # --- ISA トランスパイル ---
    pass_manager = _build_pass_manager(backend)
    ansatz_isa = pass_manager.run(full_ansatz)

    # ハミルトニアン・力演算子にレイアウトを適用
    hamiltonian_isa = hamiltonian.apply_layout(ansatz_isa.layout)
    force_op_isa = force_op.apply_layout(ansatz_isa.layout)

    # --- ノイズモデル付きシミュレータ ---
    backend_sim = AerSimulator.from_backend(backend)
    energy_precision = _shots_to_precision(shots)
    estimator = BackendEstimatorV2(
        backend=backend_sim,
        options={"default_precision": energy_precision},
    )

    logger.info(
        "Noisy backend: %s, energy shots=%d (precision=%.6f), "
        "force shots=%d (precision=%.6f)",
        backend.name, shots, energy_precision,
        force_shots, _shots_to_precision(force_shots),
    )

    # --- VQE 最適化 ---
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

    logger.info(
        "Starting COBYLA for distance=%.2f Å (noisy, backend=%s)",
        distance_angstrom, backend.name,
    )

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
        shifted_step, trace[-1] if trace else float(res.fun),
    )

    res = minimize(
        cost_function,
        cobyla_point,
        method="L-BFGS-B",
        callback=_callback,
        options={
            "maxiter": optimizer_maxiter,
            "disp": False,
            "gtol": 1e-12,
            "ftol": 1e-12,
        },
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

    # --- 力の計算（ショット数を増やして精度を確保）---
    force_precision = _shots_to_precision(force_shots)
    pub_force = (ansatz_isa, [force_op_isa], [res.x])
    force_result = estimator.run(
        pubs=[pub_force],  # type: ignore[arg-type]
        precision=force_precision,
    ).result()
    raw_force = float(force_result[0].data.evs[0])  # type: ignore[attr-defined]

    # _build_h2_force_operator は -dH/dR を返す → そのまま力
    force_val = raw_force

    logger.info(
        "Force at optimised geometry: %.6f Ha/Å (force_shots=%d)",
        force_val, force_shots,
    )

    # --- ログファイル書き出し ---
    os.makedirs(f"logs/{timestamp}", exist_ok=True)
    log_path = f"logs/{timestamp}/h2_energy_quantum_{distance_angstrom:.2f}.txt"
    with open(log_path, "w") as f:
        f.write(f"target distance: {distance_angstrom:.2f} Angstrom\n")
        for step, energy in enumerate(trace):
            f.write(f"{step},{energy:.12f}\n")
            if step == shifted_step - 1:
                f.write("# Switched from COBYLA to L-BFGS-B here\n")
        f.write(f"Minimum energy: {result_energy:.12f} Ha\n")
        f.write(f"Optimization time: {total_time:.2f} seconds\n")
        f.write(f"executed with {backend.name} noisy simulator\n")
        f.write(f"shots: {shots}, force_shots: {force_shots}\n")
        f.write(f"final x parameters: {res.x}\n")
        f.write(f"computed force at optimized geometry: {force_val:.12f} Ha/Angstrom\n")

    if return_trace:
        if not trace and res.x is not None:
            trace.append(cost_function(res.x))
        return result_energy, tuple(trace)
    return result_energy, force_val
