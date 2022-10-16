import es
from test_functions import rastrigin, rosenbrock
import numpy as np


if __name__ == '__main__':
    n = 2
    xstart = np.array([0.5] * n)
    sigma = 0.5
    es_ = es.CMAES(xstart, sigma)
    while True:
        X = es_.ask()
        fitness_list = np.array([rastrigin(X[:, i]) for i in range(X.shape[1])])
        es_.tell(X, fitness_list)

        if es_.stop():
            print(es_.stop())
            break
    print(es_.result())
    # np.random.seed(1)
    # x = np.random.randn(1)
    # print(x)