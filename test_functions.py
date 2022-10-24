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


def schaffer(x: np.ndarray) -> float:
    '''
    Schaffer function
    '''
    assert len(x) == 2, 'Schaffer is a 2D function'
    return 0.5 + (np.sin(x[0]**2 - x[1]**2)**2 - 0.5) \
        / ((1 + 0.001 * (x[0]**2 + x[1]**2))**2)
