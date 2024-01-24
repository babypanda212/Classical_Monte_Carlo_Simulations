import SIM_MC_functions as mc
import numpy as np

# TASKS 
# 3.1

import matplotlib.pyplot as plt

def magnetization_squared(lattice):
    # Calculate the squared magnetization
    return (np.sum(lattice) / lattice.size)**2

def generate_samples(size, temperature, num_samples, J):
    lattice = mc.initialize_lattice(size)
    samples = []

    for _ in range(num_samples):
        mc.metropolis_update(lattice, temperature, J)
        samples.append(magnetization_squared(lattice))

    return np.array(samples)

# System size
lattice_sizes = [10, 20, 40, 80]
# Temperature range
temperature_range = np.linspace(1, 6, 20)
# Number of samples
num_samples = 20000

plt.figure(figsize=(10, 6))

for size in lattice_sizes:
    mean_magnetization_squared = []

    for temperature in temperature_range:
        samples = generate_samples(size, temperature, num_samples, J=1.0)
        mean_magnetization_squared.append(np.mean(samples))

    plt.plot(temperature_range, mean_magnetization_squared, label=f'L={size}')

plt.title('Expectation Value of Squared Magnetization')
plt.xlabel('Temperature (T/J)')
plt.ylabel('M^2')
plt.legend()
plt.show()