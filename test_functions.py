import numpy as np


def rastrigin(x: np.ndarray) -> float:
    '''
    Rastrigin function
    '''
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x: np.ndarray) -> float:
    '''
    Rosenbrock function
    '''
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)