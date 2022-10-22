import numpy as np


def rastrigin(x: np.ndarray) -> float:
    '''
    Rastrigin function
    '''
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

# def rastrigin(x):
#     x = np.asarray(x)
#     """Rastrigin test objective function"""
#     if not np.isscalar(x[0]):
#         N = len(x[0])
#         return [10 * N + sum(xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x]
#         # return 10*N + sum(x**2 - 10*np.cos(2*np.pi*x), axis=1)
#     N = len(x)
#     return 10 * N + sum(x**2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x: np.ndarray) -> float:
    '''
    Rosenbrock function
    '''
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def schaffer(x: np.ndarray) -> float:
    '''
    Schaffer function
    '''
    assert len(x) == 2, 'Schaffer is a 2D function'
    return 0.5 + (np.sin(x[0]**2 - x[1]**2)**2 - 0.5) \
        / ((1 + 0.001 * (x[0]**2 + x[1]**2))**2)