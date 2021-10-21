"""Misc. useful functions for Atomistic Green's Function computations."""
from numba import njit
from typing import Sequence
import numpy as np


@njit(parallel=True)
def fold_matrix(arr: np.ndarray, p: int, q: int) -> np.ndarray:
    """fold a shape (M,N) matrix to shape (m,n,p,q) where m=M/p and n=N/q"""
    M, N = arr.shape
    new_arr = np.zeros((M // p, N // q, p, q), dtype=arr.dtype)
    for i in range(M):
        for j in range(N):
            val = arr[i][j]
            k = i % p
            l = j % q
            i0 = i // p
            j0 = j // q
            new_arr[i0, j0, k, l] = val

    return new_arr


@njit(parallel=True)
def unfold_matrix(arr: np.ndarray) -> np.ndarray:
    """unfold a shape (m,n,p,q) matrix to shape (M,N) where M=p*m and N=q*n"""
    m, n, p, q = arr.shape
    new_arr = np.zeros((m * p, n * q), dtype=arr.dtype)

    for i in range(m):
        i0 = p * i
        for j in range(n):
            j0 = q * j
            for k in range(p):
                for l in range(q):
                    val = arr[i, j, k, l]
                    new_arr[i0 + k, j0 + l] = val

    return new_arr


@njit(parallel=True)
def extract_matrix(row_index: np.ndarray, col_index: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """extract all elements in row_index x col_index, from arr into a new array"""
    m = row_index.shape[0]
    n = col_index.shape[0]
    d = arr.shape[2]
    matrix = np.zeros((m, n, d, d), dtype=arr.dtype)

    i0 = 0
    for i in row_index:
        j0 = 0
        for j in col_index:
            matrix[i0][j0] = arr[i][j]
            j0 += 1
        i0 += 1

    return matrix


def get_zhang_delta(omegas: Sequence[float],
                    c1: float = 1e-3,
                    c2: float = 0.0,
                    max_val: float = None) -> np.ndarray:
    """
    A frequency broadening function, see:

        Zhang et al. Numerical Heat Transfer, Part B: Fundamentals 51.4 (2007): 333-349.
        Eq. 39
    """
    if max_val is None:
        max_val = max(omegas)
    omegas = np.asarray(omegas)
    return c1 * ((1.0 + c2) - omegas / max_val) * omegas**2
