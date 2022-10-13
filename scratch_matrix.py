from typing import List
from copy import deepcopy


def transpose(A):
    B = deepcopy(A)
    for i in range(len(A)):
        for j in range(len(A)):
            B[i][j] = A[j][i]
    return B


def vt_sc_mul(x: List[float], a: float) -> List[float]:
    y = deepcopy(x)
    for i in range(len(y)):
        y[i] *= a
    return y


def mtx_sc_mul(A: List[List[float]], a: float) -> List[List[float]]:
    prod = deepcopy(A)
    for i in range(len(A)):
        prod[i] = vt_sc_mul(prod[i], a)
    return prod


def dot_prod(x: List[float], y: List[float]) -> float:
    assert len(x) == len(y), f"Vectors' size not matched: \
        (1, {len(x)}), (1, {len(y)})"
    prod = 0
    for xi, yi in zip(x, y):
        prod += xi * yi
    return prod


def mtx_vt_mul(A: List[List[float]], x: List[int]) -> List[float]:
    assert len(x) == len(A[0]), f"Vector's size (1, {len(x)}) \
        not matched with the matrix's ({len(A)},{len(A[0])})"
    prod = []
    for i, row in enumerate(A):
        prod.append(dot_prod(row, x))
    return prod


def mtx_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    assert len(A) == len(B[0]) and len(B) == len(A[0]), f"Matrices' size \
        not matched: ({len(A)}, {len(A[0])}), ({len(B)}, {len(B[0])})"
    B_ = transpose(B)
    prod = deepcopy(A)

    for i, Ai in enumerate(A):
        for j, Bj in enumerate(B_):
            prod[i][j] = dot_prod(Ai, Bj)
    return prod


def vt_add(x: List[float], y: List[float]) -> List[float]:
    assert len(x) == len(y), f"Vectors' size not matched: \
        (1, {len(x)}), (1, {len(y)})"
    sum_ = []
    for xi, yi in zip(x, y):
        sum_.append(xi + yi)
    return sum_


def mtx_add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    assert len(A) == len(B) and len(B[0]) == len(A[0]), f"Matrices' size \
        not matched: ({len(A)}, {len(A[0])}), ({len(B)}, {len(B[0])})"
    sum_ = []
    for Ai, Bi in zip(A, B):
        sum_.append(vt_add(Ai, Bi))
    return sum_


class SquareMatrix(list):
    '''
    Square matrix class
    '''

    def __init__(self, dim: int):
        self += [(i - 1) * [0] + [1] + (dim - i) * [0] 
                for i in range(1, dim + 1)]
        self._dim = dim


    @property
    def dim(self):
        return self._dim


    @property
    def diag(self):
        return [self[i][i] for i in range(self.dim)]


    def transpose(self) -> List[List[float]]:
        self = transpose(self)
        return self


    def sc_mul(self, a: float) -> List[List[float]]:
        self = mtx_sc_mul(self, a)
        return self


    def vt_mul(self, x: List[int]) -> List[float]:
        self = mtx_vt_mul(self, x)
        return self


    def mtx_mul(self, A: List[List[float]]) -> List[List[float]]:
        self = mtx_mul(self, A)
        return self


    def add(self, A: List[List[float]]) -> List[List[float]]:
        self = mtx_add(self, A)
        return self


class MatrixDiagonalization(SquareMatrix):
    '''
    Diagonalizing-square-matrices class
    '''


    def __init__(self, A: SquareMatrix) -> None:
        '''
        A: square matrix
        '''








