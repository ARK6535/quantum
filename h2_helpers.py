from __future__ import annotations

import glob
import logging
import os
import re
from itertools import combinations
from typing import Any

import numpy as np
from pyscf import ao2mo, gto, mcscf, scf
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate
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

# Shared scientific constants (kept here to avoid duplication across scripts)
HARTREE_TO_JOULE = 4.3597447222071e-18
ANGSTROM_TO_METER = 1.0e-10
AMU_TO_KG = 1.66053906660e-27
FS_TO_SECOND = 1.0e-15
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
M_H = 1836.15267389  # Hydrogen nuclear mass [a.u.]
MU = M_H / 2.0
T_AU_FS = 0.02418884254  # 1 atomic unit of time in femtoseconds
A0 = BOHR_TO_ANGSTROM

__all__ = [
    "A0",
    "ANGSTROM_TO_BOHR",
    "ANGSTROM_TO_METER",
    "AMU_TO_KG",
    "BOHR_TO_ANGSTROM",
    "FS_TO_SECOND",
    "HARTREE_TO_JOULE",
    "M_H",
    "MU",
    "T_AU_FS",
    "compute_h2_energy_classical",
    "parse_log_files",
]

logger = logging.getLogger(__name__)


def _build_sparse_pauli_hamiltonian(
    ecore: float,
    h1e: np.ndarray,
    h2e: np.ndarray,
    cholesky_tol: float,
) -> SparsePauliOp:
    ncas, _ = h1e.shape

    creators, destructors = _creators_destructors(2 * ncas)
    excitations: list[list[SparsePauliOp]] = []
    for p in range(ncas):
        terms = [creators[p] @ destructors[p] + creators[ncas + p] @ destructors[ncas + p]]
        for r in range(p + 1, ncas):
            terms.append(
                creators[p] @ destructors[r]
                + creators[ncas + p] @ destructors[ncas + r]
                + creators[r] @ destructors[p]
                + creators[ncas + r] @ destructors[ncas + p]
            )
        excitations.append(terms)

    lop, rank = _cholesky(h2e, cholesky_tol)
    t1e = h1e - 0.5 * np.einsum("pxxr->pr", h2e)

    hamiltonian = ecore * _identity(2 * ncas)
    for p in range(ncas):
        for r in range(p, ncas):
            hamiltonian += t1e[p, r] * excitations[p][r - p]

    for g in range(rank):
        lg = 0 * _identity(2 * ncas)
        for p in range(ncas):
            for r in range(p, ncas):
                lg += lop[p, r, g] * excitations[p][r - p]
        hamiltonian += 0.5 * lg @ lg

    return hamiltonian.chop().simplify()


def _identity(num_qubits: int) -> SparsePauliOp:
    return SparsePauliOp.from_list([("I" * num_qubits, 1.0)])


def _creators_destructors(num_modes: int) -> tuple[list[SparsePauliOp], list[SparsePauliOp]]:
    creators: list[SparsePauliOp] = []
    for index in range(num_modes):
        if index == 0:
            left, right = "I" * (num_modes - 1), ""
        elif index == num_modes - 1:
            left, right = "", "Z" * (num_modes - 1)
        else:
            left, right = "I" * (num_modes - index - 1), "Z" * index
        creator = SparsePauliOp.from_list(
            [
                (left + "X" + right, 0.5),
                (left + "Y" + right, 0.5j),
            ]
        )
        creators.append(creator)
    destructors = [op.adjoint() for op in creators]
    return creators, destructors


def _cholesky(tensor: np.ndarray, eps: float) -> tuple[np.ndarray, int]:
    num_orbitals = tensor.shape[0]
    ch_max = 20 * num_orbitals
    reshaped = tensor.reshape(num_orbitals**2, num_orbitals**2)
    factors = np.zeros((num_orbitals**2, ch_max))
    diag = np.diagonal(reshaped).copy()

    nu_max = np.argmax(diag)
    vmax = diag[nu_max]
    rank = 0

    while vmax > eps:
        factors[:, rank] = reshaped[:, nu_max]
        if rank > 0:
            factors[:, rank] -= np.dot(
                factors[:, :rank],
                (factors.T)[:rank, nu_max],
            )
        factors[:, rank] /= np.sqrt(vmax)
        diag -= factors[:, rank] ** 2
        rank += 1
        nu_max = np.argmax(diag)
        vmax = diag[nu_max]

    factors = factors[:, :rank].reshape((num_orbitals, num_orbitals, rank))
    return factors, rank

def _build_pass_manager(backend: Any) -> PassManager:
    target = backend.target
    preset_pm = generate_preset_pass_manager(target=target, optimization_level=3)
    preset_pm.scheduling = PassManager(
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
    return preset_pm


def _build_h2_qubit_hamiltonian(
    *,
    distance_angstrom: float,
    basis: str,
    cholesky_tol: float,
) -> tuple[SparsePauliOp, gto.Mole, Any, np.ndarray]:
    mol = _build_h2_molecule(distance_angstrom, basis)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    active_space = range(max(mol.nelectron // 2 - 1, 0), mol.nelectron // 2 + 1)
    mx: Any = mcscf.CASCI(mf, ncas=2, nelecas=(1, 1))
    mo = mx.sort_mo(active_space, base=0)
    mx.kernel(mo)

    h1e, ecore = mx.get_h1eff()
    h2e = ao2mo.restore(1, mx.get_h2eff(), mx.ncas)

    return _build_sparse_pauli_hamiltonian(ecore, h1e, h2e, cholesky_tol), mol, mx, mo


def _build_h2_molecule(distance_angstrom: float, basis: str) -> gto.Mole:
    mol = gto.Mole()
    mol.build(
        atom=[
            ["H", (0.0, 0.0, -distance_angstrom / 2.0)],
            ["H", (0.0, 0.0, distance_angstrom / 2.0)],
        ],
        unit="Angstrom",
        basis=basis,
        charge=0,
        spin=0,
        symmetry=None,
        verbose=0,
    )
    return mol

def _find_lowest_det_occupation(
    hamiltonian: SparsePauliOp,
    num_electrons: int,
    backend: AerSimulator,
) -> tuple[tuple[int, ...], float]:
    """Find the lowest-energy single-determinant among all computational-basis states.

    Iterates over every determinant with exactly *num_electrons* occupied
    qubits and returns the one with the lowest energy expectation value.

    Args:
        hamiltonian: Qubit Hamiltonian as a SparsePauliOp.
        num_electrons: Number of occupied qubits (popcount constraint).
        backend: AerSimulator backend used for expectation-value evaluation.

    Returns:
        A tuple ``(occupied_indices, energy)`` where *occupied_indices* is a
        tuple of qubit indices that are occupied, and *energy* is the
        corresponding energy in Hartree.
    """
    num_qubits = hamiltonian.num_qubits
    if num_qubits is None:
        raise ValueError("Hamiltonian does not define the number of qubits.")

    estimator = BackendEstimatorV2(backend=backend)

    best_occ: tuple[int, ...] | None = None
    best_energy: float = float("inf")

    # 例: num_qubits=4, num_electrons=2 → combinations(range(4), 2) = 6通り
    for occ in combinations(range(num_qubits), num_electrons):
        qc = QuantumCircuit(num_qubits)
        # occ に含まれる qubit に X をかけて |...1100> のような状態を作る
        for q in occ:
            qc.x(q)

        # いまの VQE と同じく BackendEstimatorV2 + statevector で期待値を評価
        # パラメータなし回路なので parameter_values は空でよい
        pub = (qc, [hamiltonian], [np.array([])])
        result = estimator.run(pubs=[pub]).result()
        energy = float(result[0].data.evs[0])

        if energy < best_energy:
            best_energy = energy
            best_occ = tuple(occ)

    if best_occ is None:
        raise RuntimeError("Failed to find any determinant occupation pattern.")

    return best_occ, best_energy

def parse_log_files(log_dir: str) -> list[tuple[float, float, float, float | None]]:
    """Parse VQE log files and extract per-distance results.

    Reads all ``h2_energy_quantum_*.txt`` files (and the legacy
    ``*_h2_energy_quantum.txt`` format) in *log_dir* and extracts the
    bond distance, minimum VQE energy, initial (HF) energy, and
    Hellmann-Feynman force for each file.

    Args:
        log_dir: Path to the log directory to scan.

    Returns:
        A list of ``(distance, min_energy, hf_energy, force)`` tuples
        sorted by distance.  *force* is ``None`` when not present in
        the log file.
    """
    data = []
    pattern = os.path.join(log_dir, "*_h2_energy_quantum.txt")
    pattern2 = os.path.join(log_dir, "h2_energy_quantum_*.txt")

    files = glob.glob(pattern) + glob.glob(pattern2)

    logger.info("Found %d files in %s", len(files), log_dir)

    for file_path in files:
        filename = os.path.basename(file_path)

        # ファイル名から距離を抽出
        match = re.search(r"h2_energy_quantum_([\d\.]+)\.txt", filename)
        match2 = re.search(r"([\d\.]+)_h2_energy_quantum\.txt", filename)
        match = match or match2
        if not match:
            logger.warning("Skipping file with unexpected name format: %s", filename)
            continue

        distance = float(match.group(1))
        min_energy = None
        hf_energy = None  # ログ中の最初のエネルギー
        force = None

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 0,-1.0xxxxx のようなトレースの最初の行を HF エネルギーとする
                # 「まだ hf_energy が未設定」かつ 「カンマ区切りの行」のときに拾う
                if hf_energy is None and "," in line and "Minimum energy:" not in line:
                    # 例: "0,-1.055159794471"
                    parts = line.split(",")
                    if len(parts) >= 2:
                        try:
                            hf_energy = float(parts[1])
                        except ValueError:
                            logger.warning("Could not parse HF energy from line: %s in %s", line, filename)

                # 最適化後の最小エネルギー
                if "Minimum energy:" in line:
                    # 例: "Minimum energy: -0.977581425733 Ha"
                    parts = line.split()
                    try:
                        min_energy = float(parts[2])
                    except (ValueError, IndexError):
                        logger.warning("Could not parse minimum energy from line: %s in %s", line, filename)

                if "computed force at optimized geometry:" in line:
                    # 例: "computed force at optimized geometry: 0.0123456789 Ha/Angstrom"
                    parts = line.split()
                    try:
                        force = float(parts[5])
                    except (ValueError, IndexError):
                        logger.warning("Could not parse force from line: %s in %s", line, filename)

        if min_energy is None:
            logger.warning("No minimum energy found in %s", filename)
            continue
        if hf_energy is None:
            logger.warning("No HF (first) energy found in %s", filename)
            continue

        data.append((distance, min_energy, hf_energy, force))

    # 距離でソート
    data.sort(key=lambda x: x[0])
    return data

def _key(p: np.ndarray) -> bytes:
    arr = np.asarray(p, dtype=np.float64).ravel()
    return arr.tobytes()

def compute_h2_energy_classical(
    distance_angstrom: float,
    *,
    basis: str = "sto-3g",
    conv_tol: float = 1e-12,
) -> float:
    """Return the Full-CI energy of H2 at the given bond length.

    Args:
        distance_angstrom: H-H distance in Ångströms.
        basis: Gaussian basis set name.
        conv_tol: SCF convergence tolerance.

    Returns:
        Total electronic energy in Hartree.
    """
    mol = _build_h2_molecule(distance_angstrom, basis)
    mf = scf.RHF(mol)
    mf.conv_tol = conv_tol
    mf.kernel()
    mf.scf()

    # Run CASCI (equivalent to Full CI for H2/sto-6g with ncas=2)
    # H2 has 2 electrons and 2 spatial orbitals in sto-6g.
    mx: Any = mcscf.CASCI(mf, ncas=2, nelecas=(1, 1))
    e_tot = mx.kernel()
    if isinstance(e_tot, tuple):
        e_tot = e_tot[0]
    return float(e_tot)

def _build_h2_force_operator(
    mol: gto.Mole,
    mo_coeff: np.ndarray,
    ncas: int = 2,
    cholesky_tol: float = 1e-8,
    delta: float = 1e-3,  # h (Angstrom)
) -> SparsePauliOp:
    """Build the force operator -dH/dR via finite differences of AO integrals.

    Uses a 4th-order central difference scheme for higher accuracy.

    Args:
        mol: PySCF molecule object at the reference geometry.
        mo_coeff: MO coefficient matrix from the CASCI calculation.
        ncas: Number of active-space orbitals.
        cholesky_tol: Tolerance for Cholesky decomposition of the 2e integrals.
        delta: Finite-difference step size in Ångströms.

    Returns:
        The force operator as a SparsePauliOp (in units of Ha/Å).
    """
    coords0 = mol.atom_coords(unit="Angstrom")

    def build_mol_with_shift(shift: float) -> gto.Mole:
        coords = coords0.copy()
        coords[1, 2] += shift

        m = gto.Mole()
        m.build(
            atom=[[mol.atom_symbol(i), coords[i]] for i in range(len(coords0))],
            unit="Angstrom",
            basis=mol.basis,
            charge=mol.charge,
            spin=mol.spin,
            verbose=0,
        )
        return m

    # R ± h, R ± 2h
    mol_p1 = build_mol_with_shift(+delta)
    mol_m1 = build_mol_with_shift(-delta)
    mol_p2 = build_mol_with_shift(+2 * delta)
    mol_m2 = build_mol_with_shift(-2 * delta)

    # AO積分（hcore, ERI）と核反発
    h1_p1 = scf.hf.get_hcore(mol_p1)
    h1_m1 = scf.hf.get_hcore(mol_m1)
    h1_p2 = scf.hf.get_hcore(mol_p2)
    h1_m2 = scf.hf.get_hcore(mol_m2)

    h2_p1 = mol_p1.intor("int2e")
    h2_m1 = mol_m1.intor("int2e")
    h2_p2 = mol_p2.intor("int2e")
    h2_m2 = mol_m2.intor("int2e")

    e_p1 = mol_p1.energy_nuc()
    e_m1 = mol_m1.energy_nuc()
    e_p2 = mol_p2.energy_nuc()
    e_m2 = mol_m2.energy_nuc()

    # 4次中央差分（d/dR）
    h = delta
    coef = 1.0 / (12.0 * h)

    h1_deriv_ao = (-h1_p2 + 8.0 * h1_p1 - 8.0 * h1_m1 + h1_m2) * coef
    h2_deriv_ao = (-h2_p2 + 8.0 * h2_p1 - 8.0 * h2_m1 + h2_m2) * coef
    e_nuc_deriv = (-e_p2 + 8.0 * e_p1 - 8.0 * e_m1 + e_m2) * coef

    ncore = mol.nelectron // 2 - ncas // 2
    mo_cas = mo_coeff[:, ncore : ncore + ncas]

    h1_mo_deriv = mo_cas.T @ h1_deriv_ao @ mo_cas

    h2_mo_deriv_flat = ao2mo.incore.general(
        h2_deriv_ao, (mo_cas, mo_cas, mo_cas, mo_cas)
    )
    h2_mo_deriv = ao2mo.restore(1, h2_mo_deriv_flat, ncas)

    force_op = _build_sparse_pauli_hamiltonian(
        -1.0 * e_nuc_deriv,
        -1.0 * h1_mo_deriv,
        -1.0 * h2_mo_deriv,
        cholesky_tol,
    )
    return force_op
