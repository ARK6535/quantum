# Copilot Instructions

## Project Overview

VQE (Variational Quantum Eigensolver) and simple molecular dynamics simulation
scripts, currently focused on the hydrogen molecule (H₂).  The codebase is
designed to be extensible to other diatomic molecules (e.g. LiH) in the future.

| Layer | Stack |
|---|---|
| Language | Python 3.11 (macOS recommended — PySCF constraint) |
| Quantum computing | Qiskit 2.x, qiskit-aer, qiskit-ibm-runtime |
| Quantum chemistry | PySCF (RHF → CASCI → Cholesky → Jordan-Wigner) |
| Numerical | NumPy, SciPy (`optimize`, `linalg`) |
| Visualization | Matplotlib |
| Parallelism | `concurrent.futures.ProcessPoolExecutor` |

Default basis set is `sto-3g`.  A migration to a higher-quality basis set is
planned for the future; keep helper APIs basis-agnostic where possible.

---

## Module Structure

> **Target layout after refactoring** — new code must follow this structure.

```
h2_helpers.py            # Shared constants, quantum-chemistry utilities, log parsers
h2_dynamics.py           # Molecular dynamics helpers: Velocity Verlet, force wrappers, parallel workers
h2_energy.py             # Noisy-backend VQE energy (+ force) computation
h2_energy_statevector.py # Noiseless statevector VQE energy (+ force) computation
h2_energy_demo.py        # Main entry-point: MD simulation, force-curve comparison
h2_energy_distribution.py# Statistical sampling of VQE energies
vqe_h2.py               # Single-shot noisy VQE for H2 (educational / experimental)
vqe_LiH.py              # Single-shot noisy VQE for LiH (educational / experimental)
plot_energy_vs_distance.py # Visualization: energy vs distance from logs
plot_force_from_log.py     # Visualization: force vs distance from logs
read_csv.py                # Visualization: MD trajectory analysis from CSV
```

### Rules

1. **Single source of truth** — Shared functions (`compute_h2_energy_classical`,
   `_find_lowest_det_occupation`, `_key`, `parse_log_files`, physical constants,
   molecule builders, Hamiltonian builders, force-operator builders) live
   exclusively in `h2_helpers.py`.  Other modules import from there.
   Re-exporting or re-defining the same function in another module is
   **prohibited**.
2. **`__all__` in every module** — Define `__all__` to make the public API
   explicit.  Private helpers use a leading underscore (`_build_h2_molecule`).
3. **Entry-point pattern** — Every runnable script must define a `main()`
   function and guard it with `if __name__ == "__main__": main()`.
   CLI argument parsing must be extracted into a dedicated `parse_args()`
   function that returns an `argparse.Namespace`.
4. **Separation of concerns** — Visualization scripts (`plot_*.py`, `read_csv.py`)
   must not contain VQE or MD computation logic.  Computation modules must not
   contain plotting code (except optional inline debug plots behind a flag).

---

## Coding Conventions

### Imports

```python
from __future__ import annotations          # Always first — every file

import os                                    # 1. stdlib
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np                           # 2. Third-party
from pyscf import gto, scf
from qiskit.quantum_info import SparsePauliOp

from h2_helpers import (                     # 3. Local
    BOHR_TO_ANGSTROM,
    compute_h2_energy_classical,
)
```

- Follow **isort** ordering: stdlib → third-party → local, separated by blank
  lines.
- **No mid-script imports.** All imports at the top of the file.

### Type Hints

- Use **built-in lowercase generics** (`list`, `dict`, `tuple`, `set`) — not
  `typing.List`, `typing.Dict`, etc.
- Use **PEP 604 union** syntax: `X | None` — not `Optional[X]`.
- Use `numpy.typing.NDArray` for array annotations where appropriate.
- `from __future__ import annotations` enables all of the above on Python 3.11.

```python
# Good
def compute(distances: list[float], *, basis: str = "sto-3g") -> tuple[float, np.ndarray]:
    ...

# Bad — do not use
from typing import List, Optional, Tuple
def compute(distances: List[float], basis: Optional[str] = "sto-3g") -> Tuple[float, np.ndarray]:
    ...
```

### Naming

| Kind | Convention | Example |
|---|---|---|
| Function | `snake_case` | `compute_h2_energy_quantum` |
| Private function | `_snake_case` | `_build_h2_molecule` |
| Constant | `ALL_CAPS` | `BOHR_TO_ANGSTROM` |
| Physical variable | unit suffix | `distance_angstrom`, `energy_hartree`, `force_au`, `time_fs` |

### Docstrings

- **English**, **Google style**, required on every **public** function.
- Include `Args` and `Returns` sections.  `Raises` only when non-obvious.

```python
def compute_h2_energy_classical(
    distance_angstrom: float,
    *,
    basis: str = "sto-3g",
) -> float:
    """Return the Full-CI energy of H2 at the given bond length.

    Args:
        distance_angstrom: H-H distance in Ångströms.
        basis: Gaussian basis set name.

    Returns:
        Total electronic energy in Hartree.
    """
```

### Comments

- **Inline comments may be written in Japanese** (lab project convention).
- Docstrings are always English.

---

## Logging & Experiment Tracking

Use the standard `logging` module for **experiment-parameter recording** and
**progress/debug output**.  This replaces ad-hoc `print()` calls.

### Setup pattern

```python
import logging

logger = logging.getLogger(__name__)

def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info(
        "Starting VQE run: distance=%.4f Å, basis=%s, backend=%s, maxiter=%d",
        args.distance, args.basis, args.backend, args.maxiter,
    )
    ...
```

### What to log

| Level | Content |
|---|---|
| `logger.info` | Experiment parameters (initial distance, basis set, backend type — statevector vs noisy, maxiter, timestamp), major milestones (optimizer switch, run complete) |
| `logger.debug` | Per-step energy values, intermediate timings, cache hits |
| `print()` | **Final user-facing results only** (e.g. summary table printed to terminal) |

### VQE trace files

The per-step CSV trace written to `h2_energy_quantum_{distance:.2f}.txt` uses
a **custom format** (step,energy lines with `#` comment lines for optimizer
switches).  Keep this as **manual `open`/`write`** — do not route through
`logging.FileHandler`.

---

## Error Handling

- **Let library exceptions propagate.**  Do not wrap Qiskit / PySCF / NumPy
  calls in `try`/`except` — the raw traceback is more useful for debugging
  (and for pasting into an AI assistant).
- Use `try`/`except` only for **non-critical parse failures** where the loop
  should continue (e.g. `parse_log_files` skipping a malformed line).
- Do not add speculative input validation (`assert distance > 0`, etc.).
  If a library rejects bad input, its own error message suffices.

---

## Physics & Units

| Quantity | External (I/O, CLI, filenames) | Internal (computation) |
|---|---|---|
| Distance | Ångström (Å) | Bohr |
| Energy | — | Hartree (Ha) |
| Force | Ha/Å (in logs) | Ha/Bohr (in integrator) |
| Time | femtoseconds (fs) | atomic units |
| Mass | — | atomic units (electron mass = 1) |

- **Always** convert through the named constants in `h2_helpers.py`
  (`BOHR_TO_ANGSTROM`, `ANGSTROM_TO_BOHR`, `T_AU_FS`, etc.).
  Never hard-code conversion factors elsewhere.
- The reduced mass `MU` and hydrogen mass `M_H` are defined in `h2_helpers.py`.

---

## VQE & Quantum Computing Patterns

### Two-stage optimization

1. **COBYLA** (derivative-free, broad search) — global exploration phase.
2. **L-BFGS-B** (gradient-based, tight convergence) — refinement phase.

Both stages share an **`eval_cache`** dictionary (parameter bytes → energy)
to avoid redundant evaluations.  The cache key is produced by `_key()` in
`h2_helpers.py`.

> **Note:** The variable is historically misspelled as `eval_chache` in some
> files.  Fix this to `eval_cache` whenever touching those files.

### Ansatz

- **UCCSD** (`qiskit_nature.second_q.circuit.library.UCCSD`) with a
  Hartree-Fock reference circuit (`HartreeFock`).
- Qubit mapping: **Jordan-Wigner**.
- Initial parameters: zero vector (default) or values carried over from a
  neighbouring geometry.

### IBM Runtime authentication

- Use `QiskitRuntimeService()` (reads credentials from environment or
  `~/.qiskit/`).
- **Never hard-code API keys or backend names** in source files.
  `apikey.json` exists as a local reference only and is git-ignored.

---

## Files & Log Convention

### Directory layout

```
logs/<YYMMDDHHmm>/
    h2_energy_quantum_0.74.txt   # VQE trace for R = 0.74 Å
    h2_energy_quantum_0.76.txt
    ...
    _energy_vs_distance_<ts>.txt # Summary table
    _h2_dynamics_data.txt        # MD trajectory (CSV)
```

- Timestamp format: `YYMMDDHHmm` (e.g. `2602091345`).
- VQE log filename: **`h2_energy_quantum_{distance:.2f}.txt`** (canonical).
  The legacy `{distance}_h2_energy_quantum.txt` format is accepted by
  `parse_log_files` but should not be produced by new code.

### Git-ignored files

The following are listed in `.gitignore` and must never be committed:

- `apikey.json`, `.env`, `*.token`, `*.secret`
- `logs/` (entire directory)
- `*.png`, `*.pdf` (generated plots)
- `vqe_energies_*.txt` (legacy VQE outputs)
- `__pycache__/`, `.venv/`

---

## Reminders for AI Assistants

- This repository is a **research lab project**; inline comments in Japanese
  are expected and acceptable.
- When suggesting code changes, follow the conventions above — especially
  import ordering, type-hint style, and the single-source-of-truth rule.
- Prefer small, focused diffs over large rewrites.
- When the user pastes a traceback, work from the **raw library error** rather
  than suggesting additional `try`/`except` wrappers.

---

## AI Writing Rules — Content (厳守)

- Do not fabricate content or add unsourced specifics.  Never introduce
  numbers, proper nouns, or examples that are absent from the original text.
- Leave ambiguous parts ambiguous.  Restructure for readability, but do not
  resolve uncertainty by inventing detail.
- Do not ask the reader questions or request confirmation (no follow-up
  questions).
- Skip preamble declarations such as "結論から言うと", "本記事では",
  "以下で解説します".  Start directly with the substance.
- Remove **safety-cushion phrases** ("一般的に", "多くの場合",
  "状況によって異なります", "一概には言えませんが") by default.
  If a caveat is genuinely needed, compress it to the minimum.
- Do not lean on abstract adjectives alone ("重要", "効果的", "最適", "本質",
  "メリット").  Within the scope of the original text, use verb-centric
  concrete expressions that convey *what happens*.
- Do not chain synonyms for emphasis (e.g. 重要・大切・欠かせない).
  Say it **once**.
- Cut redundant summary restating ("まとめると", "要するに", "総じて") and
  any re-iteration of the same point.
- Vary sentence rhythm.  Mix short and long sentences; avoid repeating the
  same pattern (assertion → reason, conclusion → supplement).  Use
  connectives sparingly.
- Keep the narrative voice consistent.  If using first person, stick to one
  form — do not mix 私 / 筆者 / 私たち.

## AI Writing Rules — Formatting (厳守・最重要)

- **Do not use Markdown formatting** (`**bold**`, `##` headings, bullet
  markers, decorative markup) in prose output.
- **Minimise 「」.**  Remove quoting brackets used merely for emphasis or
  definition-like framing; dissolve them into the sentence flow.  Use 「」
  only for actual quotations or proper names, and sparingly.
- **Minimise ().**  Do not offload supplementary info into parentheses.
  Work it into the sentence naturally.  Explain a term in running text at
  first mention — at most once, and preferably without parentheses.
- Avoid the colon（：）as a rule.  When unavoidable, **never place a
  half-width space after it** (「： 」is forbidden).  Do not use
  label-style enumerations like "目的：背景：結論：".
- Do not use slashes（／）to juxtapose concepts.  Avoid arrows（→）and
  pseudo-code notation; write prose instead.
- No closing boilerplate ("参考になれば幸いです", "まずは小さく始めましょう",
  etc.).  If a closing sentence is needed, make it specific to the content
  and understated.
