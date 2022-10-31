import es
from test_functions import rastrigin
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def test():
    n_runs = 100
    # n_runs = 1
    np.random.seed(13)
    # ns = [2, 5, 10, 20]
    ns = [2, 5]
    # popsizes = []
    popsizes = [5, 7, 10, 14, 20, 32, 50]

    np.set_printoptions(precision=16)
    record = dict()

    for n in ns:
        for popsize in popsizes:
            fstop_count = 0
            for _ in trange(n_runs):
                mu = np.random.uniform(1, 5, n)
                A = np.identity(n)
                ftarget = 1e-8
                # figsize = (10, 10)
                # img_basename = './images/cmaes-rastrigin'
                # x1 = np.linspace(-500, 500, num=100)
                # x2 = np.linspace(-500, 500, num=100)
                f = rastrigin
                # plot = Plot(figsize, img_basename, x1, x2, f)
                es_ = es.xNES(mu, A, ftarget=ftarget, popsize=popsize)
                # es_ = es.CMAES(xstart, sigma)


                while True:
                    X = es_.ask()
                    fitness_list = np.array([f(X[:, i]) for i in range(X.shape[1])])
                    es_.tell(fitness_list)
                    result = es_.result()
                    # current_iter = result['count_iter']
                    # if current_iter % 5 == 0:    
                    #     plot.contour()
                    #     plot.point(xstart, 'black')
                    #     plot.point(result['best_sol'], 'red')
                    #     plot.point(result['solutions'], 'blue', alpha=0.3)
                    #     plot.save(f'iter{current_iter}')
                    #     plot.clf()

                    if es_.stop():
                        if result['best_val'] <= ftarget:
                            fstop_count += 1
                        # print()
                        # print(result['best_sol'])
                        # print('best_val:', -result['best_val'])
                        # print(result['solutions'])
                        # print('count_eval:', result['count_eval'])
                        print('Terminated due to:', es_.stop())
                        # print(es_.result())
                        # plot.contour()
                        # plot.point(xstart, 'black')
                        # plot.point(result['best_sol'], 'red')
                        # plot.point(result['solutions'], 'blue', alpha=0.5)
                        # plot.gif('end')
                        # plot.close()
                        break

            print(f'n={n}, popsize={popsize}, fstop_count={fstop_count}')
            record[f'n={n},popsize={popsize}'] = fstop_count

    print(record)


def dim_popsize() -> None:
    n_runs = 100
    ns = [2, 5, 10]
    popsizes = [5, 7, 10, 14, 20, 32, 50, int(50 * np.sqrt(2)), 100, int(100 * np.sqrt(2)), 200]
    colors = ['blue', 'red', 'green']
    markers = ['o', 'x', 'v']
    f = rastrigin
    ftarget = 1e-10

    methods = [
        {
            'es': es.CMAES,
            'name': 'CMA-ES',
            'image-path': './images/cmaes-rastrigin.png'
        },
        {
            'es': es.xNES,
            'name': 'xNES',
            'image-path': './images/nes-rastrigin.png'
        }
    ]

    for method in methods:
        record = np.zeros((len(ns), len(popsizes)))
        name = method['name']

        for ni, n in enumerate(ns):
            for ps_i, popsize in enumerate(popsizes):
                fstop_count = 0

                for _ in trange(n_runs):
                    if name == 'CMA-ES':
                        xstart = np.random.uniform(1, 5, n)
                        sigma = 2
                        es_ = es.CMAES(xstart, sigma, ftarget=ftarget, popsize=popsize)
                    else:
                        mu = np.random.uniform(1, 5, n)
                        A = np.identity(n)
                        es_ = es.xNES(mu, A, ftarget=ftarget, popsize=popsize)

                    while True:
                        X = es_.ask()
                        fitness_list = np.array([f(X[:, i]) for i in range(X.shape[1])])
                        es_.tell(fitness_list)
                        result = es_.result()

                        if es_.stop():
                            if result['best_val'] <= ftarget:
                                fstop_count += 1
                            break

                record[ni, ps_i] = fstop_count
                print(f'{name}, n={n}, popsize={popsize}, fstop_count={fstop_count}')

        record /= n_runs
        plt.figure(figsize=(10, 10))
        for i, n in enumerate(ns):
            plt.scatter(popsizes, record[i, :], marker=markers[i], c=colors[i])
            plt.plot(popsizes, record[i, :], linestyle='dashed', color=colors[i], label=f'n={n}')
        plt.xlabel('population size')
        plt.ylabel('success probability')
        plt.xticks([5, 10, 50, 100, 200])
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.legend(loc='upper left')
        plt.savefig(method['image-path'])
        plt.close()


if __name__ == '__main__':
    np.random.seed(13)
    dim_popsize()
    
