# first we will code a naive example
# we take a quarter of a circle and count the number in and out of the circle

import random
import numpy as np
import matplotlib.pyplot as plt

# define variables
n_samples = 100000

def estimate_pi_naive(n_samples: int) -> float:

    n_circ = 0

    for _ in range(n_samples):
        # generate coords
        x = random.random()
        y = random.random()

        if x**2 + y**2 < 1:
            n_circ += 1

    return 4 * n_circ / n_samples

# print(estimate_pi_naive(n_samples=10000))

def estimate_pi_vectorised(n_samples: int) -> np.array(float):

    x = np.random.uniform(0,1,n_samples)
    y = np.random.uniform(0,1,n_samples)

    hit_list = np.where(x**2 + y**2 < 1, 1, 0)
    cum_samples = np.arange(1, n_samples + 1)
    cum_estimate = 4 * np.cumsum(hit_list) / cum_samples

    return cum_estimate

cum_estimate = estimate_pi_vectorised(n_samples=n_samples)

# now lets create a rolling mean

cum_samples = np.arange(1, n_samples + 1)

# plt.plot(cum_samples, cum_estimate)
# plt.plot(cum_samples, np.repeat(np.pi, n_samples))
# plt.ylim(3,3.3)
# plt.xlabel("no. of samples")
# plt.ylabel("estimate of pi")
# plt.show()

# plt.plot(cum_samples, cum_estimate - np.pi)
# plt.show()

# we can look at how the speed differs


# for n_samples in np.geomspace(1e0,1e5, 5+1):
#     naive = %timeit estimate_pi_naive(int(n_samples))
#     vectorised = %timeit estimate_pi_vectorised((int(n_samples)))

# plot the above results here








