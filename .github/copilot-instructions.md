# Copilot instructions for this repo

This repo contains small, script-first VQE experiments targeting IBM Quantum backends using Qiskit 2.x. The code alternates between local simulation (Aer) and real hardware via Qiskit Runtime.

## Repository map
- `vqe_h2.py`: 4-qubit H2 Hamiltonian (hard-coded SparsePauliOp). Builds an EfficientSU2 ansatz, maps to hardware ISA with a custom pass manager (scheduling + DD), evaluates energy with EstimatorV2 and optimizes with SciPy COBYLA. Plots energy trace vs a known reference.
- `vqe_LiH.py`: Builds LiH Hamiltonian on-the-fly with PySCF + low-rank (Cholesky) factorization, then runs the same VQE stack. Saves energies to `vqe_energies_YYMMDDHHMM.txt`.
- `plotter.py`: Minimal PySCF + CASCI example for H2, computes exact eigenvalues of the qubit Hamiltonian.
- `helloworld.py`: Small EstimatorV2 example that submits a job and prints a job ID.
- `output.py`: Fetches a past job by ID and plots observable expectations.
- `requirements.txt`: Core deps (Qiskit 2.x, matplotlib, numpy, scipy, etc.). Optional deps are not pinned here (see below).
- `setup.py`/`apikey.json`: One-off IBM Quantum account setup; contains secrets. Do NOT commit real tokens.

## How things fit together
- VQE flow: define qubit Hamiltonian (SparsePauliOp) → build ansatz (`efficient_su2` RY, linear, reps=3) → hardware-aware transpile to ISA (`generate_preset_pass_manager` + scheduling + `PadDynamicalDecoupling`) → apply layout to H via `H.apply_layout(ansatz_isa.layout)` → evaluate energy with `EstimatorV2` → minimize with COBYLA.
- Estimation uses the new EstimatorV2 API with “pubs”: each pub is `(circuit, [hamiltonian], [params])`. Results are read from `result[0].data.evs[0]`.
- For fast iteration, scripts build `AerSimulator.from_backend(backend)` to mimic the selected real backend’s target/noise.

## Conventions and patterns
- Always transform the Hamiltonian after ISA mapping: `hamiltonian_isa = H.apply_layout(ansatz_isa.layout)`. Skipping this yields wrong energies.
- Optimizer: SciPy COBYLA with an energy-collecting callback (`energies.append(cost_func(...))`). `vqe_LiH.py` persists these to `vqe_energies_*.txt` and plots traces.
- The H2 script includes a state reconstruction helper expecting `res.x`; the actual optimizer variable is `res_trace.x`. If you reuse that utility, pass `res_trace.x` or rename accordingly.
- Ansatz defaults: `efficient_su2(..., su2_gates=["ry"], entanglement="linear", reps=3)`. `vqe_LiH.py` also constructs a Hartree–Fock bitstring, but it’s not currently wired into the ansatz.

## Dependencies and environment
- **Virtual Environment**: Always activate the virtual environment before running any scripts or installing packages. Use `source /Users/mac/Desktop/quantum/.venv/bin/activate`.
- Python: prefer 3.10–3.12. Qiskit Aer build may fail on 3.14 on macOS/Apple Silicon (see `log.txt`). If Aer fails to build, use a supported Python version or conda-forge wheels.
- Optional packages for some scripts (not in `requirements.txt`):
  - `pyscf` and `qiskit-nature` for `vqe_LiH.py` and `plotter.py`.
- IBM Quantum access: save an account once with `QiskitRuntimeService.save_account(...)`. Don’t hardcode tokens in tracked files—prefer environment variables or local-only files ignored by Git.

## Typical workflows
- Local VQE (H2): run `vqe_h2.py` to see COBYLA steps and a plot vs reference energy (`true_energy = -1.137306`).
- LiH VQE: run `vqe_LiH.py` to generate `vqe_energies_YYMMDDHHMM.txt`; it uses PySCF to build the Hamiltonian and the same optimizer loop.
- Real hardware: both VQE scripts pick `service.least_busy(operational=True, simulator=False)`; swap to a named backend or use a `Session` + Runtime `EstimatorV2` (the commented block in `vqe_h2.py` shows the intended pattern).

## Gotchas and troubleshooting
- Aer install/build errors (macOS/py3.14): use Python ≤3.12 or conda; ensure `qiskit-aer` installs from wheels. See `log.txt` for a failure example.
- Missing deps: installing `pyscf`/`qiskit-nature` is required for LiH/H2 PySCF examples.
- Results API: EstimatorV2 returns a list of results; energy is under `.data.evs[0]`, standard deviation under `.data.stds[0]`.
- Secrets: `setup.py` and `apikey.json` include tokens for convenience—treat them as sensitive; rotate and move to env vars if committed accidentally.
