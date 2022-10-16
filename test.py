import es
from test_functions import rastrigin, rosenbrock, schaffer
import numpy as np
import matplotlib.pyplot as plt
from plot import Plot


if __name__ == '__main__':
    n = 2
    xstart = np.random.randn(n)
    sigma = 0.5
    ftarget = 1e-10
    figsize = (10, 10)
    img_basename = './images/cmaes-schaffer'
    x1 = np.linspace(-5, 5, num=100)
    x2 = np.linspace(-5, 5, num=100)
    f = schaffer
    plot = Plot(figsize, img_basename, x1, x2, f)
    es_ = es.CMAES(xstart, sigma, ftarget=ftarget)

    plot.contour()
    plot.point(xstart, 'black')
    plot.save('search-space')

    while True:
        X = es_.ask()
        fitness_list = np.array([f(X[:, i]) for i in range(X.shape[1])])
        es_.tell(X, fitness_list)
        result = es_.result()
        current_iter = result['count_iter']
        if current_iter % 2 == 0:
            plot.point(result['best_sol'], 'red')
            plot.save(f'iter{current_iter}')

        if es_.stop():
            print('Terminated due to:', es_.stop())
            print(es_.result())
            plot.gif()
            plot.close()
            break