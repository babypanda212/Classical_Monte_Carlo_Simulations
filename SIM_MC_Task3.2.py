import SIM_MC_functions as mc
import numpy as np 
import matplotlib.pyplot as plt

# Define the temperature grid
n_values = np.arange(-12, 29)
T_values = 2**(n_values/4)

# Define linear system sizes
L_values = [10, 20, 40, 80]

# Number of Monte Carlo steps for each temperature
N_MC = 100

# Dictionary to store generated samples
samples = {}

# Generate Monte Carlo samples for each temperature and system size
for L in L_values:
    samples[L] = {}
    for T in T_values:
        sample = mc.generate_sample(size=L, temperature=T, J=1.0, B=0.0, N=N_MC)
        samples[L][T] = sample
        print(f"Generated {N_MC} samples for size {L} and temperature {T}")

print("Generated Monte Carlo samples")

# Binning analysis for each temperature and system size
kmax = 20
error_estimates = {}

for L in L_values:
    error_estimates[L] = {}
    for T in T_values:
        energies = [mc.calculate_total_energy(sample, J=1.0, B=0.0) for sample in samples[L][T]]
        error_est = mc.binning_analysis(energies, kmax)
        error_estimates[L][T] = error_est

# Plot the results
for L in L_values:
    plt.figure(figsize=(10, 6))
    for T in T_values:
        plt.plot(range(1, kmax + 1), error_estimates[L][T], label=f'T={T:.4f}')
    
    plt.title(f'Binning Analysis - System Size L={L}')
    plt.xlabel('Binning Level')
    plt.ylabel('Error Estimate')
    plt.legend()
    plt.savefig(f'Binning_Analysis_L_{L}.png')
    plt.show()

print("done")