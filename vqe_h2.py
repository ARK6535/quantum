# General imports
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from matplotlib import pyplot as plt
from qiskit.quantum_info import Statevector
 
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
# Hamiltonian obtained from a previous lesson
 
H = SparsePauliOp(
    [
        "IIII",
        "IIIZ",
        "IZII",
        "IIZI",
        "ZIII",
        "IZIZ",
        "IIZZ",
        "ZIIZ",
        "IZZI",
        "ZZII",
        "ZIZI",
        "YYYY",
        "XXYY",
        "YYXX",
        "XXXX",
    ],
    coeffs=[
        -0.09820182 + 0.0j,
        -0.1740751 + 0.0j,
        -0.1740751 + 0.0j,
        0.2242933 + 0.0j,
        0.2242933 + 0.0j,
        0.16891402 + 0.0j,
        0.1210099 + 0.0j,
        0.16631441 + 0.0j,
        0.16631441 + 0.0j,
        0.1210099 + 0.0j,
        0.17504456 + 0.0j,
        0.04530451 + 0.0j,
        0.04530451 + 0.0j,
        0.04530451 + 0.0j,
        0.04530451 + 0.0j,
    ],
)
 
nuclear_repulsion = 0.7199689944489797

# Pre-defined ansatz circuit
from qiskit.circuit.library import efficient_su2
 
# SciPy minimizer routine
from scipy.optimize import minimize
 
# Plotting functions
 
# Random initial state and efficient_su2 ansatz
ansatz = efficient_su2(H.num_qubits, su2_gates=["ry"], entanglement="linear", reps=4)
ansatz.draw(output="mpl")
plt.show()
x0 = 2 * np.pi * np.random.random(ansatz.num_parameters)
print(ansatz.decompose().depth())
print(ansatz.num_parameters)

def cost_func(params, ansatz, H, estimator):
    pub = (ansatz, [H], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]
    return energy
 
 
# def cost_func_sim(params, ansatz, H, estimator):
#    energy = estimator.run(ansatz, H, parameter_values=params).result().values[0]
#    return energy

# To run on hardware, select the backend with the fewest number of jobs in the queue
from qiskit_ibm_runtime import QiskitRuntimeService
 
service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)
print("Using backend:", backend.name)

from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes import (
    ALAPScheduleAnalysis,
    PadDynamicalDecoupling,
    ConstrainedReschedule,
)
from qiskit.circuit.library import XGate
 
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
plt.show()

hamiltonian_isa = H.apply_layout(ansatz_isa.layout)

# We will start by using a local simulator
from qiskit_aer import AerSimulator
 
# Import an estimator, this time from qiskit (we will import from Runtime for real hardware)
from qiskit.primitives import BackendEstimatorV2
from qiskit_ibm_runtime.options import EstimatorOptions

# 時間計測
import time
start_time = time.time()
 
# generate a simulator that mimics the real quantum system
backend_sim = AerSimulator.from_backend(backend)
estimator = BackendEstimatorV2(backend=backend_sim)

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

# Plot (no Japanese). Draw horizontal line at true energy = -1.137306
true_energy = -1.137306
steps = list(range(len(energies)))

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



# from qiskit_ibm_runtime import QiskitRuntimeService, Session
# from qiskit_ibm_runtime import EstimatorV2 as Estimator
# from qiskit_ibm_runtime.options import EstimatorOptions

# estimator_options = EstimatorOptions(resilience_level=0, default_shots=2000)
# with Session(backend=backend) as session:
#     estimator = Estimator(mode=session, options=estimator_options)
 
#     res = minimize(
#         cost_func,
#         x0,
#         args=(ansatz_isa, hamiltonian_isa, estimator),
#         method="cobyla",
#         options={"maxiter": 500, "disp": True,},
#     )

# print(getattr(res, "fun") - nuclear_repulsion)
# print(res)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

