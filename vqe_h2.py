"""Single-shot noisy VQE for H₂ (educational / experimental).

Runs a COBYLA-based VQE on a noisy Aer simulator that mimics a real IBM
backend.  The qubit Hamiltonian is hard-coded for R ≈ 0.735 Å (STO-3G).
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
from matplotlib import pyplot as plt
from qiskit.circuit.library import XGate, efficient_su2
from qiskit.primitives import BackendEstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    ALAPScheduleAnalysis,
    ConstrainedReschedule,
    PadDynamicalDecoupling,
)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from scipy.optimize import minimize

from h2_helpers import (
    _build_h2_qubit_hamiltonian,
    save_run_config,
)

__all__: list[str] = []

logger = logging.getLogger(__name__)


def _cost_func(
    params: np.ndarray,
    ansatz,
    hamiltonian: SparsePauliOp,
    estimator: BackendEstimatorV2,
) -> float:
    """Evaluate the VQE cost function (energy expectation value)."""
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    return float(result[0].data.evs[0])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Single-shot noisy VQE for H₂")
    parser.add_argument(
        "--distance",
        type=float,
        default=0.735,
        help="H-H distance in angstroms (default: 0.735)",
    )
    parser.add_argument(
        "--basis",
        default="sto-3g",
        help="Gaussian basis set name (default: sto-3g)",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("QISKIT_BACKEND", "ibm_kawasaki"),
        help="IBM backend name (default: $QISKIT_BACKEND or ibm_kawasaki)",
    )
    parser.add_argument(
        "--ansatz-reps",
        type=int,
        default=0,
        help="Number of EfficientSU2 repetition layers (default: 0)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Number of measurement shots (default: 1024)",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=300,
        help="Maximum COBYLA iterations (default: 300)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    """Run single-shot noisy VQE for H₂."""
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    for name in ("vqe_h2",):
        logging.getLogger(name).setLevel(
            logging.DEBUG if args.verbose else logging.INFO,
        )

    timestamp = time.strftime("%y%m%d%H%M")
    log_dir = f"logs/{timestamp}"

    # ハミルトニアンを動的に構築
    hamiltonian, mol, mx, _mo = _build_h2_qubit_hamiltonian(
        distance_angstrom=args.distance,
        basis=args.basis,
        cholesky_tol=1e-6,
    )
    nuclear_repulsion = mol.energy_nuc()
    logger.info(
        "Hamiltonian: %d qubits, %d terms, nuclear_repulsion=%.6f",
        hamiltonian.num_qubits, len(hamiltonian), nuclear_repulsion,
    )

    # 古典 CASCI エネルギーを参照値として計算
    casci_energy = mx.e_tot
    logger.info("CASCI reference energy: %.6f Ha", casci_energy)

    save_run_config(
        log_dir,
        args,
        backend_type=f"noisy:{args.backend}",
        extra={
            "cholesky_tol": 1e-6,
            "nuclear_repulsion": nuclear_repulsion,
            "casci_energy": casci_energy,
            "hamiltonian_num_qubits": hamiltonian.num_qubits,
            "hamiltonian_num_terms": len(hamiltonian),
        },
    )

    logger.info("Backend: %s, maxiter: %d", args.backend, args.maxiter)

    # IBM Runtime接続
    service = QiskitRuntimeService()
    backend = service.backend(args.backend)
    logger.info("Using backend: %s", backend.name)

    # アンザッツ構築
    ansatz = efficient_su2(
        hamiltonian.num_qubits,
        su2_gates=["ry"],
        entanglement="linear",
        reps=args.ansatz_reps,
    )
    x0 = 2 * np.pi * np.random.random(ansatz.num_parameters)
    logger.debug(
        "Ansatz depth: %d, parameters: %d",
        ansatz.decompose().depth(),
        ansatz.num_parameters,
    )

    # トランスパイル
    target = backend.target
    pm = generate_preset_pass_manager(target=target, optimization_level=0)
    pm.scheduling = PassManager(
        [
            ALAPScheduleAnalysis(target=target),
            ConstrainedReschedule(
                acquire_alignment=target.acquire_alignment,
                pulse_alignment=target.pulse_alignment,
                target=target,
            ),
            PadDynamicalDecoupling(
                target=target,
                dd_sequence=[XGate(), XGate()],
                pulse_alignment=target.pulse_alignment,
            ),
        ]
    )
    ansatz_isa = pm.run(ansatz)
    hamiltonian_isa = hamiltonian.apply_layout(ansatz_isa.layout)

    # ノイズシミュレータ構築
    backend_sim = AerSimulator.from_backend(backend)
    estimator = BackendEstimatorV2(backend=backend_sim)

    # VQE最適化
    energies: list[float] = []
    start_time = time.time()

    def callback(xk: np.ndarray) -> None:
        energy = _cost_func(xk, ansatz_isa, hamiltonian_isa, estimator)
        energies.append(energy)
        elapsed = time.time() - start_time
        logger.info(
            "Step %d: energy=%.6f Ha, elapsed=%.2f s",
            len(energies), energy, elapsed,
        )

    result = minimize(
        _cost_func,
        x0,
        args=(ansatz_isa, hamiltonian_isa, estimator),
        method="cobyla",
        callback=callback,
        options={"maxiter": args.maxiter, "disp": False},
        tol=1e-3,
    )

    # 結果がない場合のフォールバック
    if not energies and hasattr(result, "x"):
        energies.append(
            _cost_func(result.x, ansatz_isa, hamiltonian_isa, estimator),
        )

    elapsed_total = time.time() - start_time
    logger.info("Optimization finished in %.2f s", elapsed_total)

    # 結果表示
    print(f"Final energy - nuclear repulsion: {result.fun - nuclear_repulsion:.6f} Ha")
    print(result)
    print(f"Elapsed time: {elapsed_total:.2f} seconds")

    # プロット
    steps = list(range(len(energies)))
    plt.figure(figsize=(8, 5))
    plt.plot(steps, energies, marker="o", label="VQE energy")
    plt.axhline(
        casci_energy,
        color="red",
        linestyle="--",
        label=f"CASCI energy = {casci_energy:.6f}",
    )
    plt.xlabel("Step")
    plt.ylabel("Energy (Hartree)")
    plt.title(f"H₂ VQE Energy vs Step (R={args.distance} Å, {args.basis})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

