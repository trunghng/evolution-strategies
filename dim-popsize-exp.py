import es
from test_functions import rastrigin
import numpy as np
import matplotlib.pyplot as plt
from plot import Plot
from tqdm import trange


if __name__ == '__main__':
    np.random.seed(2)
    n_runs = 100
    ns = [2, 5, 10]
    popsizes = [5, 7, 10, 14, 20, 32, 50, int(50 * np.sqrt(2)), 100]
    colors = ['blue', 'red', 'green']
    markers = ['o', 'x', 'v']

    record = np.zeros((len(ns), len(popsizes)))

    for ni, n in enumerate(ns):
        for pi, popsize in enumerate(popsizes):
            fstop_count = 0
            for _ in trange(n_runs):
                xstart = np.random.uniform(1, 5, n)
                sigma = 2
                ftarget = 1e-10
                f = rastrigin
                es_ = es.CMAES(xstart, sigma, ftarget=ftarget, popsize=popsize)

                while True:
                    X = es_.ask()
                    fitness_list = np.array([f(X[:, i]) for i in range(X.shape[1])])
                    es_.tell(X, fitness_list)
                    result = es_.result()

                    if es_.stop():
                        if result['best_val'] <= ftarget:
                            fstop_count += 1
                        break

            record[ni, pi] = fstop_count
            print(f'n={n}, popsize={popsize}, fstop_count={fstop_count}')

record /= n_runs
plt.figure(figsize=(10, 10))
for i, n in enumerate(ns):
    plt.scatter(popsizes, record[i, :], marker=markers[i], c=colors[i])
    plt.plot(popsizes, record[i, :], linestyle='dashed', color=colors[i], label=f'n={n}')
plt.xlabel('population size')
plt.ylabel('success probability')
plt.xticks([0, 10, 50, 100])
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.legend(loc='upper left')
plt.savefig('./images/dim-popsize-exp.png')
plt.close()
