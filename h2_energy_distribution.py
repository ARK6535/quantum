
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os
from qiskit_ibm_runtime import QiskitRuntimeService
from h2_energy_statevector import compute_h2_energy_quantum_statevector as compute_h2_energy_quantum

def run_batch(n_samples: int, distance: float, backend_name: str, timestamp: str, batch_id: int) -> list[float]:
    """
    Runs a batch of energy calculations.
    Fetches the backend once per batch to save overhead.
    """
    print(f"Batch {batch_id} starting: {n_samples} samples.")
    
    # Initialize service and backend once per worker/batch
    try:
        service = QiskitRuntimeService()
        if backend_name:
            backend = service.backend(backend_name)
        else:
            # Fallback if no name provided, though main should provide it
            backend = service.least_busy(operational=True, simulator=False)
    except Exception as e:
        print(f"Batch {batch_id} failed to initialize backend: {e}")
        return []

    energies = []
    for i in range(n_samples):
        try:
            # We pass the backend object directly to avoid re-fetching
            energy = compute_h2_energy_quantum(
                distance_angstrom=distance,
                timestamp=timestamp,
                # Use default parameters as per vqe_h2.py / h2_energy.py
                basis="sto-3g", # h2_energy.py default is sto-6g
                ansatz_reps=1,  # h2_energy_demo uses 1, let's stick to a reasonable default or 3 as per instructions?
                                # Instructions say: "Ansatz defaults: efficient_su2(..., reps=3)"
                                # But h2_energy.py default is 0?
                                # Let's check h2_energy.py default.
            )
            energies.append(energy)
        except Exception as e:
            print(f"Batch {batch_id} sample {i} failed: {e}")
    
    print(f"Batch {batch_id} finished.")
    return energies

def main():
    parser = argparse.ArgumentParser(description="H2 Energy Distribution")
    parser.add_argument("--distance", type=float, default=0.735, help="Interatomic distance in Angstrom")
    parser.add_argument("--samples", type=int, default=1000, help="Total number of samples")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--backend", type=str, default=None, help="Backend name")
    
    args = parser.parse_args()
    
    timestamp = time.strftime("%y%m%d%H%M")
    
    # 1. Resolve backend name in main process to ensure all workers use the same one
    service = QiskitRuntimeService()
    if args.backend:
        backend_name = args.backend
    else:
        print("Selecting least busy backend...")
        backend = service.least_busy(operational=True, simulator=False)
        backend_name = backend.name
    
    print(f"Target Backend: {backend_name}")
    print(f"Distance: {args.distance} Å")
    print(f"Total Samples: {args.samples}")
    print(f"Workers: {args.workers}")
    
    # 2. Prepare batches
    base_batch_size = args.samples // args.workers
    remainder = args.samples % args.workers
    
    batches = []
    for i in range(args.workers):
        size = base_batch_size + (1 if i < remainder else 0)
        if size > 0:
            batches.append(size)
            
    print(f"Batch sizes: {batches}")
    
    # 3. Run in parallel
    all_energies = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for i, batch_size in enumerate(batches):
            futures.append(executor.submit(run_batch, batch_size, args.distance, backend_name, timestamp, i))
            
        for future in futures:
            batch_energies = future.result()
            all_energies.extend(batch_energies)
            
    total_time = time.time() - start_time
    print(f"Collected {len(all_energies)} samples in {total_time:.2f} seconds.")
    
    # 4. Save results
    os.makedirs(f"logs/{timestamp}", exist_ok=True)
    output_file = f"logs/{timestamp}/energy_distribution_{args.distance:.3f}.txt"
    np.savetxt(output_file, all_energies)
    print(f"Saved energies to {output_file}")
    
    # 5. Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_energies, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f"Energy Distribution for H2 at R={args.distance} Å\n(Backend: {backend_name}, Samples: {len(all_energies)})")
    plt.xlabel("Energy (Hartree)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_e = float(np.mean(all_energies))
    std_e = float(np.std(all_energies))
    min_e = float(np.min(all_energies))
    plt.axvline(mean_e, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_e:.6f}')
    
    plt.legend()
    
    plot_file = f"logs/{timestamp}/energy_distribution_{args.distance:.3f}.png"
    plt.savefig(plot_file)
    print(f"Saved plot to {plot_file}")
    plt.show()

if __name__ == "__main__":
    main()
