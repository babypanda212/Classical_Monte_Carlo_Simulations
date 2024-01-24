import SIM_MC_functions as mc
import numpy as np

# see paper about binning analysis, very interesting and 
# relevant to SIM
# understand """ what Xl^(k) is ! ask Yoshito for help
# """

size = 10
temperature = 1.0
J = 1.0
B = 0.0
N = 10

parameters = (temperature,J,B,N)

lattice = mc.initialize_lattice(size)
lattice, energies, wts = mc.update_sweep(lattice,*parameters)

# calculate estimation of expectation value of energy
energy_EV = np.sum(energies)/N

# calculate estimation of the expectation value of energy squared
energy_squared_EV = np.sum(np.square(energies)) / N

# specific heat
specific_heat = (energy_squared_EV - energy_EV**2) / temperature

# calculate variance
variance = np.sum(*((energies-energy_EV)**2))

# calculating error estimate for energy expectation value
error = np.sqrt(variance/len(lattice))

# calculate entropy
entropy = np.log(2) - specific_heat*np.log(temperature)