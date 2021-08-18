"""A module of utility functions for AGF computations"""
import os
import numpy as np
from typing import List


def read_fc(force_constants_filename: str) -> np.ndarray:
    """
    Read force constants from text file into a numpy array.
    The returned array has shape (n,n,3,3), i.e. each matrix element is a 3x3 matrix
    """
    force_constants_filename = os.path.expanduser(force_constants_filename)
    with open(force_constants_filename) as f:
        line = f.readline()

        # outermost matrix size
        n, m = [int(x) for x in line.split()]

        # actual matrix with explicit size
        force_constants = np.zeros((n, m, 3, 3), dtype=np.double)

        # extract interactions
        line = f.readline()
        atom_i, atom_j = [int(x) - 1 for x in line.split()]
        for i in range(3):
            line = f.readline()
            force_constants[atom_i][atom_j][i] = [float(x) for x in line.split()]
        line = f.readline()

        while line:
            atom_i, atom_j = [int(x) - 1 for x in line.split()]
            for i in range(3):
                line = f.readline()
                force_constants[atom_i][atom_j][i] = [float(x) for x in line.split()]
            line = f.readline()

    return unfold_matrix(force_constants)


def flatten_list(nested_list: List[list]) -> list:
    """return a 1D list from an arbitrarily nested list"""

    def _flatten_list_helper(some_list: List, _flat_list: list) -> list:
        """helper function for `flatten_list` recursion"""
        for _item in some_list:
            if type(_item) is list:
                _flatten_list_helper(_item, _flat_list)
            else:
                _flat_list.append(_item)
        return _flat_list

    flat_list = []
    for item in nested_list:
        flat_list = _flatten_list_helper(item, flat_list)
    return flat_list


def fold_matrix(arr: np.ndarray, p: int, q: int) -> np.ndarray:
    """fold a shape (M,N) matrix to shape (m,n,p,q) where m=M/p and n=N/q"""
    M, N = arr.shape
    if not (M % p == 0 and N % q == 0):
        raise ValueError(
            f"can fold matrix of shape ({M},{N}) to shape (n,m,{p},{q})"
        )
    new_arr = np.zeros((M // p, N // q, p, q))
    for i in range(M):
        for j in range(N):
            val = arr[i][j]
            k = i % p
            l = j % q
            i0 = i // p
            j0 = j // q
            new_arr[i0, j0, k, l] = val

    return new_arr


def unfold_matrix(arr: np.ndarray) -> np.ndarray:
    """unfold a shape (m,n,p,q) matrix to shape (M,N) where M=p*m and N=q*n"""
    m, n, p, q = arr.shape
    new_arr = np.zeros((m * p, n * q))

    for i in range(m):
        i0 = p * i
        for j in range(n):
            j0 = q * j
            for k in range(p):
                for l in range(q):
                    val = arr[i, j, k, l]
                    new_arr[i0 + k, j0 + l] = val

    return new_arr


def index_nonzero(arr: np.ndarray) -> tuple:
    """
    Extract indices a,b,c,d for arr[a:b, c:d], the largest submatrix of arr
    with nonzero rows and columns.
    """
    x, y = np.nonzero(arr)
    a = x.min()
    b = x.max() + 1
    c = y.min()
    d = y.max() + 1
    return a, b, c, d


def nonzero_submatrix(arr: np.ndarray) -> np.ndarray:
    """extract the largest submatrix with nonzero rows and columns"""
    a, b, c, d = index_nonzero(arr)
    return arr[a:b, c:d]


def slice_top_right(arr: np.ndarray, m: int, n: int) -> np.ndarray:
    """return the top right shape (m,n) matrix inside arr"""
    return arr[:n, -m:]


def slice_bottom_left(arr: np.ndarray, m: int, n: int) -> np.ndarray:
    """return the bottom left shape (m,n) matrix inside arr"""
    return arr[-n:, :m]
