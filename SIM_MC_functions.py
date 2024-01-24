import numpy as np

def initialize_lattice(size):
    # Initialize a triangular lattice with random spins (+1 or -1)
    lattice = np.random.choice([-1, 1], size=(size, size))
    return lattice


def energy_difference(lattice, i, j, J, B):
    # Calculate the energy difference when a single spin at (i, j) is flipped

    # Indices of neighboring spins
    neighbors = [
        ((i+1)%len(lattice), j),
        ((i-1)%len(lattice), j),
        (i, (j+1)%len(lattice[i])),
        (i, (j-1)%len(lattice[i])),
        ((i+1)%len(lattice), (j+1)%len(lattice[i])),
        ((i-1)%len(lattice), (j-1)%len(lattice[i]))
    ]

    neighbor_sum = 0

    # Sum of spins at neighbouring sites
    neighbor_sum = sum(lattice[x, y] for x, y in neighbors)
    
    energy_diff = -2.0*lattice[i,j]*neighbor_sum - 2*lattice[i,j]*B

    # Return the energy difference
    return energy_diff


def calculate_total_energy(lattice, J, B):
    # Calculate the total energy of the lattice for the Ising model with an external magnetic field
    energy = 0
    l = len(lattice)

    for i in range(l):
        for j in range(l):
            spin = lattice[i, j]

            # Indices of neighboring spins
            neighbors = [((i+1)%l,j),(i,(j+1)%l),((i+1)%l,(j+1)%l)
                                     ]

            neighbor_sum = sum(lattice[x, y] for x, y in neighbors)
            energy += J * spin * neighbor_sum - B * spin

    return energy


def metropolis_update(lattice, temperature, J, B):
    # Perform a Metropolis update for each spin in the lattice
    l = len(lattice)

    # Random choice of lattice site
    i = np.random.randint(0,l)
    j = np.random.randint(0,l)
  
    energy_diff = energy_difference(lattice, i, j, J, B)
            
    # Metropolis acceptance criterion
    if energy_diff < 0 or np.random.rand() < np.exp(-energy_diff / temperature):
        lattice[i, j] *= -1
    return lattice


def update_sweep(lattice, temperature, J, B, N):
    # Perform an update sweep with a given number of Markov steps

    # initialising variables needed inside the loop
    energies = []

    # The Markov steps
    for _ in range(N):
        metropolis_update(lattice, temperature, J, B)
        energy = calculate_total_energy(lattice, J, B)
        energies.append(energy)

    # get array of energies
    energies = np.array(energies)

    return lattice, energies

def generate_sample(size, temperature, J, B, N):
    # Initialize the lattice
    lattice = np.random.choice([-1, 1], size=(size, size))

    # Initialize an array to store configurations
    sample = np.zeros((N + 1, size, size))
    sample[0] = lattice.copy()

    # Run the Markov chain and store configurations
    for sweep in range(1, N + 1):
        lattice, _ = update_sweep(lattice, temperature, J, B, 1)
        sample[sweep] = lattice.copy()

    return sample

def binning_analysis(X, kmax):
    M = len(X)
    error_est = np.zeros(kmax)

    for k in range(1, kmax + 1):
        Mk = M // k
        Xk = np.mean(np.reshape(X, (k, Mk)), axis=0)
        
        # Compute coarse grained sequence
        error_est[k-1] = np.std(Xk) / np.sqrt(Mk-1)
    
    return error_est

size = 10
temperature = 1.0
J = 1.0
B = 0.0
N = 10

parameters = (temperature,J,B,N)

lattice = initialize_lattice(size)
lattice, energies, wts = update_sweep(lattice,*parameters)

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