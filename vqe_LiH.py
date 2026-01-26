import numpy as np
from pyscf import ao2mo, gto, mcscf, scf
from qiskit.quantum_info import SparsePauliOp
from matplotlib import pyplot as plt
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper

# To run on hardware, select the backend with the fewest number of jobs in the queue
from qiskit_ibm_runtime import QiskitRuntimeService
 
service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)

from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes import (
    ALAPScheduleAnalysis,
    PadDynamicalDecoupling,
    ConstrainedReschedule,
)
from qiskit.circuit.library import XGate

# We will start by using a local simulator
from qiskit_aer import AerSimulator
 
# Import an estimator, this time from qiskit (we will import from Runtime for real hardware)
from qiskit.primitives import BackendEstimatorV2
from qiskit_ibm_runtime.options import EstimatorOptions

# generate a simulator that mimics the real quantum system
backend_sim = AerSimulator.from_backend(backend)
estimator = BackendEstimatorV2(backend=backend_sim)

# Define LiH molecule

distance = 1.56
mol = gto.Mole()
mol.build(
    verbose=0,
    atom=[["Li", (0, 0, 0)], ["H", (0, 0, distance)]],
    basis="sto-6g",
    spin=0,
    charge=0,
    symmetry="Coov",
)
mf = scf.RHF(mol)
mf.scf()
 
print(
    mf.energy_nuc(),
    mf.energy_elec()[0],
    mf.energy_tot(),
    mf.energy_tot() - mol.energy_nuc(),
)

active_space = range(mol.nelectron // 2 - 1, mol.nelectron // 2 + 1)

# create Fermion Hamiltonian

E1 = mf.kernel()
mx = mcscf.CASCI(mf, ncas=5, nelecas=(1, 1))
cas_space_symmetry = {"A1": 3, "E1x": 1, "E1y": 1}
mo = mcscf.sort_mo_by_irrep(mx, mf.mo_coeff, cas_space_symmetry)
E2 = mx.kernel(mo)[:2]

h1e, ecore = mx.get_h1eff()
h2e = ao2mo.restore(1, mx.get_h2eff(), mx.ncas)

def cholesky(V, eps):
    # see https://arxiv.org/pdf/1711.02242.pdf section B2
    # see https://arxiv.org/abs/1808.02625
    # see https://arxiv.org/abs/2104.08957
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
    print(
        "accuracy of Cholesky decomposition ",
        np.abs(np.einsum("prg,qsg->prqs", L, L) - V).max(),
    )
    return L, ng

def identity(n):
    return SparsePauliOp.from_list([("I" * n, 1)])
 
 
def creators_destructors(n, mapping="jordan_wigner"):
    c_list = []
    if mapping == "jordan_wigner":
        for p in range(n):
            if p == 0:
                ell, r = "I" * (n - 1), ""
            elif p == n - 1:
                ell, r = "", "Z" * (n - 1)
            else:
                ell, r = "I" * (n - p - 1), "Z" * p
            cp = SparsePauliOp.from_list([(ell + "X" + r, 0.5), (ell + "Y" + r, 0.5j)])
            c_list.append(cp)
    else:
        raise ValueError("Unsupported mapping.")
    d_list = [cp.adjoint() for cp in c_list]
    return c_list, d_list

def build_hamiltonian(ecore: float, h1e: np.ndarray, h2e: np.ndarray) -> SparsePauliOp:
    ncas, _ = h1e.shape
 
    C, D = creators_destructors(2 * ncas, mapping="jordan_wigner")
    Exc = []
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
 
    # low-rank decomposition of the Hamiltonian
    Lop, ng = cholesky(h2e, 1e-6)
    t1e = h1e - 0.5 * np.einsum("pxxr->pr", h2e)
 
    H = ecore * identity(2 * ncas)
    # one-body term
    for p in range(ncas):
        for r in range(p, ncas):
            H += t1e[p, r] * Exc[p][r - p]
    # two-body term
    for g in range(ng):
        Lg = 0 * identity(2 * ncas)
        for p in range(ncas):
            for r in range(p, ncas):
                Lg += Lop[p, r, g] * Exc[p][r - p]
        H += 0.5 * Lg @ Lg
 
    return H.chop().simplify()

H = build_hamiltonian(ecore, h1e, h2e)

nuclear_repulsion = mf.energy_nuc()
print("Nuclear repulsion:", nuclear_repulsion)
print("Theoretical energy (Hartree):", mf.energy_tot())

# Pre-defined ansatz circuit
from qiskit.circuit.library import efficient_su2
 
# SciPy minimizer routine
from scipy.optimize import minimize
 
# Plotting functions
 
# Random initial state and efficient_su2 ansatz
ansatz = efficient_su2(H.num_qubits, su2_gates=["ry"], entanglement="linear", reps=2)
# x0 = np.zeros(ansatz.num_parameters)
x0 = [0,0,1,1,1,0,0,1,1,1]
x0.extend([0]*(ansatz.num_parameters-10))
print(len(active_space))
hf_state = HartreeFock(
    num_spatial_orbitals=H.num_qubits // 2,
    num_particles=(len(active_space), len(active_space)),
    qubit_mapper=JordanWignerMapper(),
)
hf_state.decompose().draw("mpl")
print(ansatz.decompose().depth())
ansatz.decompose().draw("mpl")

def cost_func(params, ansatz, H, estimator):
    pub = (ansatz, [H], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]
    return energy


 
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
 
 
# Use the pass manager and draw the resulting circuit
ansatz_isa = pm.run(ansatz)
ansatz_isa.draw(output="mpl", idle_wires=False, style="iqp")

hamiltonian_isa = H.apply_layout(ansatz_isa.layout)



# 時間計測
import time
start_time = time.time()
 


# res = minimize(
#     cost_func,
#     x0,
#     args=(ansatz_isa, hamiltonian_isa, estimator),
#     method="cobyla",
#     options={"maxiter": 300, "disp": True,},
#     tol=1e-3
# )
 
# print(getattr(res, "fun") - nuclear_repulsion)
# print(res)

# record energies per iteration using a callback, then plot energy vs step with a horizontal line at the true energy
energies = []

def _callback(xk):
    energies.append(cost_func(xk, ansatz_isa, hamiltonian_isa, estimator))
    time_elapsed = time.time() - start_time
    print(f"Step {len(energies)}: Energy = {energies[-1]}, Time Elapsed = {time_elapsed:.2f} seconds")

res_trace = minimize(
    cost_func,
    x0,
    args=(ansatz_isa, hamiltonian_isa, estimator),
    method="cobyla",
    callback=_callback,
    options={"maxiter": 300, "disp": False},
    tol=1e-3,
)

# ensure at least one point (fallback to final result)
if not energies and hasattr(res_trace, "x"):
    energies.append(cost_func(res_trace.x, ansatz_isa, hamiltonian_isa, estimator))

true_energy = mf.energy_tot()
steps = list(range(len(energies)))
# energiesをtxtファイルに保存　ファイル名は"vqe_energies_yymmddhhmm.txt"を新規作成
np.savetxt("vqe_energies_{}.txt".format(time.strftime("%y%m%d%H%M")), energies)

plt.figure(figsize=(8, 5))
plt.plot(steps, energies, marker="o", label="VQE energy")
plt.axhline(true_energy, color="red", linestyle="--", label=f"True energy = {true_energy}")
plt.xlabel("Step")
plt.ylabel("Energy(Hartree)")
plt.title("Energy vs Step")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(getattr(res_trace, "fun") - nuclear_repulsion)
print(res_trace)

# 時間計測終了

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")