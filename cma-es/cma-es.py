import numpy as np
import matplotlib.pyplot as plt

def rastrigin(x, n):
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

n = 2

# Param settings
lambda_ = int(4 + np.floor(3 * np.log(n)))
mu = lambda_ // 2

weights_prime = np.array([np.log((lambda_ + 1) / 2) - np.log(i) 
    for i in range(1, lambda_)])
weights_prime_pos = weights_prime[:mu]
weights_prime_neg = weights_prime[mu:]
mueff = np.sum(weights_prime_pos) ** 2 \
    / np.sum(weights_prime_pos ** 2)
mueff_neg = np.sum(weights_prime_neg) ** 2 \
    / np.sum(weights_prime_neg ** 2)

alpha_cov = 2
cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
c1 = alpha_cov / ((n + 1.3) ** 2 + mueff)
cmu = min(1 - c1, alpha_cov * (mueff - 2 + 1 / mueff) / \
    ((n + 2) ** 2 + alpha_cov * mueff / 2))

alpha_mu_minus = 1 + c1 / cmu
alpha_mueff_minus = 1 + 2 * mueff_neg / (mueff + 2)
alpha_posdef_minus = (1 - c1 - cmu) / (n * cmu)
weights_pos = 1 / np.sum(weights_prime_pos) * weights_prime_pos
weights_neg = min(alpha_mu_minus, alpha_mueff_minus, alpha_posdef_minus) \
    / np.sum(np.abs(weights_prime_neg)) * weights_prime_neg

cm = 1

csigma = (mueff + 2) / (n + mueff + 5)
dsigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + csigma


# Initialization
evolution_path_cov = np.zeros(n)
evolution_path_sigma = np.zeros(n)
cov = np.identity(n)
g = 0
