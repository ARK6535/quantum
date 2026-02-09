"""Single-shot noisy VQE for LiH (educational / experimental).

Runs a COBYLA-based VQE on a noisy Aer simulator.  The qubit Hamiltonian is
constructed from a PySCF CASCI calculation with Cholesky decomposition and
Jordan-Wigner mapping.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
from matplotlib import pyplot as plt
from pyscf import ao2mo, gto, mcscf, scf
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

__all__ = [
    "cholesky",
    "identity",
    "creators_destructors",
    "build_hamiltonian",
]

logger = logging.getLogger(__name__)


def cholesky(V: np.ndarray, eps: float) -> tuple[np.ndarray, int]:
    """Compute Cholesky decomposition of a two-electron integral tensor.

    References:
        - https://arxiv.org/pdf/1711.02242.pdf section B2
        - https://arxiv.org/abs/1808.02625
        - https://arxiv.org/abs/2104.08957

    Args:
        V: Four-index electron repulsion integral tensor.
        eps: Convergence threshold for the decomposition.

    Returns:
        Tuple of (L, ng) where L is the Cholesky vectors reshaped to
        (no, no, ng) and ng is the number of vectors.
    """
    no = V.shape[0]
    chmax, ng = 20 * no, 0
    W = V.reshape(no**2, no**2)
    L = np.zeros((no**2, chmax))
    Dmax = np.diagonal(W).copy()
    nu_max = np.argmax(Dmax)
    vmax = Dmax[nu_max]
    while vmax > eps:
        L[:, ng] = W[:, nu_max]
        if ng > 0:
            L[:, ng] -= np.dot(L[:, 0:ng], (L.T)[0:ng, nu_max])
        L[:, ng] /= np.sqrt(vmax)
        Dmax[: no**2] -= L[: no**2, ng] ** 2
        ng += 1
        nu_max = np.argmax(Dmax)
        vmax = Dmax[nu_max]
    L = L[:, :ng].reshape((no, no, ng))
    accuracy = np.abs(np.einsum("prg,qsg->prqs", L, L) - V).max()
    logger.debug("Cholesky decomposition accuracy: %e", accuracy)
    return L, ng


def identity(n: int) -> SparsePauliOp:
    """Return the n-qubit identity operator.

    Args:
        n: Number of qubits.

    Returns:
        Identity SparsePauliOp.
    """
    return SparsePauliOp.from_list([("I" * n, 1)])


def creators_destructors(
    n: int,
    mapping: str = "jordan_wigner",
) -> tuple[list[SparsePauliOp], list[SparsePauliOp]]:
    """Build fermionic creation and annihilation operators.

    Args:
        n: Number of spin-orbitals.
        mapping: Qubit mapping scheme (only 'jordan_wigner' supported).

    Returns:
        Tuple of (creators, destructors) as lists of SparsePauliOp.
    """
    c_list: list[SparsePauliOp] = []
    if mapping == "jordan_wigner":
        for p in range(n):
            if p == 0:
                ell, r = "I" * (n - 1), ""
            elif p == n - 1:
                ell, r = "", "Z" * (n - 1)
            else:
                ell, r = "I" * (n - p - 1), "Z" * p
            cp = SparsePauliOp.from_list(
                [(ell + "X" + r, 0.5), (ell + "Y" + r, 0.5j)],
            )
            c_list.append(cp)
    else:
        raise ValueError(f"Unsupported mapping: {mapping}")
    d_list = [cp.adjoint() for cp in c_list]
    return c_list, d_list


def build_hamiltonian(
    ecore: float,
    h1e: np.ndarray,
    h2e: np.ndarray,
) -> SparsePauliOp:
    """Build qubit Hamiltonian from molecular integrals via Cholesky + Jordan-Wigner.

    Args:
        ecore: Core (frozen) energy including nuclear repulsion.
        h1e: One-electron integral matrix.
        h2e: Two-electron integral tensor.

    Returns:
        Qubit Hamiltonian as SparsePauliOp.
    """
    ncas, _ = h1e.shape

    C, D = creators_destructors(2 * ncas, mapping="jordan_wigner")
    Exc: list[list[SparsePauliOp]] = []
    for p in range(ncas):
        Excp = [C[p] @ D[p] + C[ncas + p] @ D[ncas + p]]
        for r in range(p + 1, ncas):
            Excp.append(
                C[p] @ D[r]
                + C[ncas + p] @ D[ncas + r]
                + C[r] @ D[p]
                + C[ncas + r] @ D[ncas + p]
            )
        Exc.append(Excp)

    # Cholesky分解による低ランクハミルトニアン構築
    Lop, ng = cholesky(h2e, 1e-6)
    t1e = h1e - 0.5 * np.einsum("pxxr->pr", h2e)

    H = ecore * identity(2 * ncas)
    # 一体項
    for p in range(ncas):
        for r in range(p, ncas):
            H += t1e[p, r] * Exc[p][r - p]
    # 二体項
    for g in range(ng):
        Lg = 0 * identity(2 * ncas)
        for p in range(ncas):
            for r in range(p, ncas):
                Lg += Lop[p, r, g] * Exc[p][r - p]
        H += 0.5 * Lg @ Lg

    return H.chop().simplify()


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
    parser = argparse.ArgumentParser(description="Single-shot noisy VQE for LiH")
    parser.add_argument(
        "--backend",
        default=os.environ.get("QISKIT_BACKEND"),
        help="IBM backend name (default: least busy via Runtime)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=1.56,
        help="Li-H distance in angstroms (default: 1.56)",
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
    """Run single-shot noisy VQE for LiH."""
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    for name in ("vqe_LiH",):
        logging.getLogger(name).setLevel(
            logging.DEBUG if args.verbose else logging.INFO,
        )

    # IBM Runtime接続
    service = QiskitRuntimeService()
    if args.backend:
        backend = service.backend(args.backend)
    else:
        backend = service.least_busy(operational=True, simulator=False)
    logger.info("Using backend: %s", backend.name)

    # LiH分子構築
    mol = gto.Mole()
    mol.build(
        verbose=0,
        atom=[["Li", (0, 0, 0)], ["H", (0, 0, args.distance)]],
        basis="sto-6g",
        spin=0,
        charge=0,
        symmetry="Coov",
    )
    mf = scf.RHF(mol)
    mf.scf()
    nuclear_repulsion = mf.energy_nuc()
    true_energy = mf.energy_tot()
    logger.info(
        "RHF: nuclear_repulsion=%.6f, E_elec=%.6f, E_tot=%.6f",
        nuclear_repulsion,
        mf.energy_elec()[0],
        true_energy,
    )

    # CASCI計算
    active_space = range(mol.nelectron // 2 - 1, mol.nelectron // 2 + 1)
    mx = mcscf.CASCI(mf, ncas=5, nelecas=(1, 1))
    cas_space_symmetry = {"A1": 3, "E1x": 1, "E1y": 1}
    mo = mcscf.sort_mo_by_irrep(mx, mf.mo_coeff, cas_space_symmetry)
    casci_result = mx.kernel(mo)[:2]
    logger.info("CASCI energy: %s", casci_result)

    h1e, ecore = mx.get_h1eff()
    h2e = ao2mo.restore(1, mx.get_h2eff(), mx.ncas)

    # ハミルトニアン構築
    H = build_hamiltonian(ecore, h1e, h2e)
    logger.info("Hamiltonian: %d qubits, %d terms", H.num_qubits, len(H))

    # アンザッツ構築
    ansatz = efficient_su2(
        H.num_qubits, su2_gates=["ry"], entanglement="linear", reps=2,
    )
    x0: list[float] = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
    x0.extend([0.0] * (ansatz.num_parameters - 10))
    logger.debug(
        "Ansatz depth: %d, parameters: %d",
        ansatz.decompose().depth(),
        ansatz.num_parameters,
    )

    # トランスパイル
    target = backend.target
    pm = generate_preset_pass_manager(target=target, optimization_level=3)
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
    hamiltonian_isa = H.apply_layout(ansatz_isa.layout)

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

    # VQEトレース保存
    np.savetxt(f"vqe_energies_{time.strftime('%y%m%d%H%M')}.txt", energies)

    # 結果表示
    print(f"Final energy - nuclear repulsion: {result.fun - nuclear_repulsion:.6f} Ha")
    print(result)
    print(f"Elapsed time: {elapsed_total:.2f} seconds")

    # プロット
    steps = list(range(len(energies)))
    plt.figure(figsize=(8, 5))
    plt.plot(steps, energies, marker="o", label="VQE energy")
    plt.axhline(
        true_energy,
        color="red",
        linestyle="--",
        label=f"True energy = {true_energy:.6f}",
    )
    plt.xlabel("Step")
    plt.ylabel("Energy (Hartree)")
    plt.title("LiH VQE Energy vs Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
