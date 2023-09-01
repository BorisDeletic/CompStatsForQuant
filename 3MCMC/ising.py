import random
import math
import numpy as np
import matplotlib.pyplot as plt

# onsager analytic result
CRITICAL_TEMP = 2.26918


# $$ H = \beta \sum_i $$

def magnetisation(lattice):
    return np.sum(lattice) / lattice.size


# return i in the lattice with periodic boundary conditions
def boundary_conditions(n, i):
    i = n - 1 if i < 0 else i
    i = 0 if i > n - 1 else i

    return i


def delta_energy(lattice, i, j):
    flipped_spin = -1 * lattice[i][j]

    idx_above = boundary_conditions(n, j + 1)
    idx_below = boundary_conditions(n, j - 1)
    idx_left = boundary_conditions(n, i - 1)
    idx_right = boundary_conditions(n, i + 1)

    neighbour_sum = lattice[i][idx_above] + lattice[i][idx_below] + lattice[idx_left][j] + lattice[idx_right][j]

    init_energy = - lattice[i][j] * neighbour_sum
    new_energy = - flipped_spin * neighbour_sum

    return new_energy - init_energy


def metropolis_update(lattice, temperature):
    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):

            if math.exp(-delta_energy(lattice, i, j) / temperature) > random.random():
                # flip the spin
                lattice[i][j] *= -1


steps = 100
n = 30
temp = 1.2

lattice = np.ones((n, n))
mags = []


# for t in range(steps):
#     for _ in range(n ** 2):
#         i = random.randint(0, n - 1)
#         j = random.randint(0, n - 1)
#         metropolis_update(lattice, temp, i, j)
#
#     mags.append(magnetisation(lattice))

# print(np.mean(mags))

plt.plot(mags)
plt.axhline(np.mean(mags), color='tab:orange', label='mean magnetisation')
plt.xlabel('iteration')
plt.ylabel('magnetisation')
plt.title('magnetisation vs markov-chain step T={}'.format(temp))
plt.legend(loc='upper right')
plt.show()

steps = 1000
n = 10

temperatures = np.arange(0.1, 4, 0.2)
mean_magnetisations = []

for temp in temperatures:
    print(temp)

    lattice = np.ones((n, n))
    mags = []

    for t in range(steps):
        metropolis_update(lattice, temp)
        mags.append(magnetisation(lattice))

    mean_mag = np.abs(np.mean(mags))
    mean_magnetisations.append(mean_mag)

plt.plot(temperatures, mean_magnetisations)
# plt.axhline(np.mean(mags), color='tab:orange', label='mean magnetisation')
plt.xlabel('temperature')
plt.ylabel('mean magnetisation')
plt.title('mean magnetisation vs temperature')
# plt.legend(loc='upper right')
plt.show()


class Ising:
    def __init__(self, n, temperature):
        # n must be even
        self.n = n if n % 2 == 0 else n - 1
        self.temperature = temperature

        self.lattice1 = np.ones((self.n, self.n // 2))
        self.lattice2 = np.ones((self.n, self.n // 2))

    def magnetisation(self):
        return np.sum(self.lattice1 + self.lattice2) / self.n ** 2

    def delta_energy(self):
        neighbour_sum = np.copy(self.lattice2)
        neighbour_sum += np.roll(self.lattice2, (-1, 0), axis=(1, 0))
        neighbour_sum += np.roll(self.lattice2, (0, 1), axis=(1, 0))
        neighbour_sum += np.roll(self.lattice2, (0, -1), axis=(1, 0))

        init_energy = - self.lattice1 * neighbour_sum
        new_energy = self.lattice1 * neighbour_sum

        return new_energy - init_energy

    def grid_metropolis_update(self):
        # lattice1 half-update
        threshold = np.random.random((self.n, self.n // 2))
        flip_spin = np.exp(-self.delta_energy() / self.temperature) > threshold
        self.lattice1 = np.where(flip_spin, -self.lattice1, self.lattice1)

        switch = self.lattice1
        self.lattice1 = self.lattice2
        self.lattice2 = switch


steps = 1000
temperatures = np.arange(0.1, 4, 0.2)

mean_mags = []
for temperature in temperatures:
    ising = Ising(n=100, temperature=temperature)
    mags = []
    for t in range(steps):
        ising.grid_metropolis_update()

        mags.append(ising.magnetisation())

    mean_mags.append(np.mean(np.abs(mags)))

plt.plot(temperatures, mean_mags)
plt.axvline(CRITICAL_TEMP, linestyle='--', color='tab:orange')
plt.show()
