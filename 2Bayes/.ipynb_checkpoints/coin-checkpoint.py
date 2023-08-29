''' Structure of the nb

- Start off by explaining the conditional prob via Bayes' rules for two coins out of a bag
- Then explain how you can view this in a probablistic way too
- We then show that we can use multiple flips to inform this distribution
- Show that it doesn't matter how we enter this data (i.e. one at a time or all at once)
- for TH and HH example show the how probability distribution change if we get HH or TH
- Then we change to a ~continuous distribution by using 100 coins in a bag with p between 0 and 1
'''

import numpy as np
from math import factorial as fact
import matplotlib.pyplot as plt

from typing import List

# two coin flips at once

def gaussian(mu, sigma, x):
    pdf = np.exp( - (x - mu)**2 / (2*sigma**2))

    return pdf / np.sum(pdf)

def nCr(n: int, r: int) -> int:
    return fact(n) // (fact(n - r) * fact(r))

def likelihood(data: np.array(bool), theta: float) -> float:

    flips = len(data)
    heads = sum(data)

    return nCr(flips, heads) * pow(theta, heads) * pow(1-theta, flips - heads)


data = np.array([True, True])
p = np.array([0.5, 1])

def posterior(data: np.array(bool), theta:float, prior: np.array(float)) -> np.array(float):
    post = likelihood(data=data, theta=theta) * prior
    return post / np.sum(post)

# plt.scatter(
#     p, posterior(data,p)
# )
# plt.xlim(-0.1,1.1)
# plt.ylim(-0.1,1.1)
# plt.show()

n = 500
true_p = 0.2
prior = np.ones(n) / n # uniform
print(prior)
# prior = np.random.normal(0,1,)
p = np.linspace(0, 0.99, n)
prior = gaussian(0.8, 0.1, p)
data = np.array(np.random.random(n) < true_p)

# first pick a coin and then flip it!
#data = np.array(np.random.random(n) < 0.5*np.random.random(n))

# do batching of data i.e. list[np.array]

dataset = np.split(data, 10)

def update(dataset: list[np.array(bool)], theta: np.array(float), prior: np.array(float)) -> np.array(float):

    post = prior

    plt.plot(
        p, post, label='prior'
    )

    i = 0
    for data in dataset:

        post = posterior(data, theta, post)

        plt.plot(
            p, post, label='{}'.format(i)
        )
        i += 1

    plt.axvline(true_p)
    plt.legend(loc='upper right')
    return post

# plt.plot(
#     p, update(dataset, p, prior)
# )

# plt.xlim(-0.1, 1.1)
# plt.ylim(-0.1, 1.1)

print(update(
    dataset, p, prior
))




plt.show()



