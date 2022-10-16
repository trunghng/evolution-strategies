import numpy as np
import matplotlib.pyplot as plt

def rastrigin(x, n):
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

n = 2

# Param settings
lambda_ = int(4 + np.floor(3 * np.log(n)))
mu = lambda_ // 2

weights_prime = np.array([np.log((lambda_ + 1) / 2) - np.log(i) 
    for i in range(1, lambda_ + 1)])
weights_prime_pos = weights_prime[:mu]
weights_prime_neg = weights_prime[mu:]
mueff = np.sum(weights_prime_pos)**2 \
    / np.sum(weights_prime_pos**2)
mueff_neg = np.sum(weights_prime_neg)**2 \
    / np.sum(weights_prime_neg**2)

alpha_cov = 2
cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
c1 = alpha_cov / ((n + 1.3)**2 + mueff)
cmu = min(1 - c1, alpha_cov * (mueff - 2 + 1 / mueff) / \
    ((n + 2)**2 + alpha_cov * mueff / 2))

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
pc = np.zeros(n)
psigma = np.zeros(n)
B = np.identity(n)
D = np.identity(n)
C = B.dot(D**2).dot(B.T)
m = np.random.multivariate_normal(np.zeros(n), np.identity(n))
sigma = 0.5
stop_fitness = 1e-10
stop_eval = 1e3 * n**2
count_eval = 0

while count_eval < stop_eval:

    # Sample new population of search pts
    Z = np.random.randn(n, lambda_)
    Y = B.dot(D).dot(Z)
    X = (m + (sigma * Y).T).T
    count_eval += lambda_

    # Selection & recombination
    fitness_arr = np.array([rastrigin(X[:, i], n) for i in range(X.shape[1])])
    ordered_indices = np.argsort(fitness_arr)
    elite_indices = ordered_indices[:mu]
    non_elite_indices = ordered_indices[mu:]
    elites = Y[:, elite_indices]
    non_elites = Y[:, non_elite_indices]
    print(weights_pos.shape, elites.shape)
    elite_ws = np.sum(weights_pos * elites, axis=1)
    m += cm * sigma * elite_ws

    # Step-size control
    psigma = (1 - csigma) * psigma + np.sqrt(csigma * (2 - csigma) \
        * mueff * np.linalg.inv(C)).dot(elite_ws)
    psigma_len = np.linalg.norm(psigma)
    psigma_explen = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
    sigma = sigma * np.exp(csigma / dsigma * (psigma_len / psigma_explen - 1))

    # Covariance matrix adaption
    g = count_eval / lambda_
    hsigma = (psigma_len / np.sqrt(1 - (1 - csigma)**(2 * g)) / psigma_explen) < 1.4 + 2 / (n + 1)
    delta_hsigma = (1 - hsigma) * cc * (2 - cc)
    pc = (1 - cc) * pc + hsigma * np.sqrt(cc * (2 - cc) * mueff) * elite_ws
    rank_one_upd = c1 * pc.reshape(n, 1).dot(pc.reshape(1, n))
    elite_cov = np.sum(np.array([weights_pos[i] * elites[:, i].reshape(n, 1).dot(elites[:, i].reshape(1, n)) \
            for i in range(len(weights_pos))]), axis=0)
    non_elite_cov = np.sum(np.array([weights_neg[i] * n / (np.linalg.norm(np.sqrt(np.linalg.inv(C)) \
            .dot(non_elites[:, i]))**2) * non_elites[:, i].reshape(n, 1).dot(non_elites[:, i].reshape(1, n)) \
        for i in range(len(weights_neg))]), axis=0)
    rank_mu_upd = elite_cov + non_elite_cov
    C = (1 + c1 * delta_hsigma - c1 - cmu * (np.sum(weights_pos) + np.sum(weights_neg))) * C \
        + rank_one_upd + rank_mu_upd
    C = np.triu(C) + np.triu(C, 1).T
    







