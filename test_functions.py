import numpy as np


def rastrigin(x: np.ndarray, n: int) -> float:
    '''
    Rastrigin function
    '''
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))