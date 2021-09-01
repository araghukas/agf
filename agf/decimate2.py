"""Implements the decimation technique for re-normalizing (ωI - H) matrices"""
import numpy as np
from typing import Tuple


class _BlockMatrix:
    """wraps a 2D array and provides easy access to blocks"""

    @property
    def k(self) -> int:
        """block matrix size"""
        return self._k

    @k.setter
    def k(self, _k):
        self._k = _k
        self._update_dimensions(_k)

    @property
    def Np(self) -> int:
        """side length (in blocks) of complete block matrix"""
        return self._Np

    @property
    def Nc(self) -> int:
        """side length (in blocks) of non-surface block matrix"""
        return self._Np - 1

    @property
    def N(self) -> int:
        """side length of the flat 2D array"""
        return self._N

    def _update_dimensions(self, _k: int) -> None:
        # check `_k` against the array shape
        M = self.arr.shape[0]
        if M % _k != 0:
            raise ValueError(f"can not divide {M}x{M} array into {_k}x{_k} blocks.")

        self._Np = self._N // _k
        self._Nc = self._Np - 1

    def __init__(self, arr: np.ndarray, k: int):
        if arr.ndim != 2:
            raise ValueError(f"input array has {arr.ndim}-dimensional, not 2.")
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(f"input array shape {arr.shape[0]}x{arr.shape[1]} is not square.")

        self._N = len(arr)
        self.arr = arr
        self.arr_prev = self.arr.copy()  # the previous surface array
        self.k = k

    def get_block(self, i: int, j: int) -> np.ndarray:
        """
        Returns a block assuming the the non-surface elements repeat indefinitely.
        For example, if Ws = W00 and Wb is [[W11, ...] [..., W22]],

        then we're sampling the image matrix:
         __________________________
        |W00 ...
        |... W11 ...
        |   ... W22 ...
        |        ... W11 ...
        |            ... W22 ...
        |                ... W11 ...
        |                       .
        |                          .
        """

        if (0 <= i * self._k < self._N) and (0 <= j * self._k < self._N):
            a, b, c, d = self._as_principal_slice(i, j)
        else:
            a, b, c, d = self._remap_block_index(i, j)
        return np.ascontiguousarray(self.arr[a:b, c:d])

    def set_block(self, i: int, j: int, new_block: np.ndarray) -> None:
        """assign new values to block[i][j]"""
        a, b, c, d = self._get_slice_numbers(i, j)
        self.arr[a:b, c:d] = new_block

    def get_previous_block(self, i: int, j: int) -> np.ndarray:
        """
        Same as `get_block`, but a block from the previous matrix is returned
        """
        a, b, c, d = self._get_slice_numbers(i, j)
        return np.ascontiguousarray(self.arr_prev[a:b, c:d])

    def set_previous_block(self, i: int, j: int, new_block: np.ndarray) -> None:
        """
        Same as `set_block`, but for a block from the previous matrix
        """
        a, b, c, d = self._get_slice_numbers(i, j)
        self.arr_prev[a:b, c:d] = new_block

    def advance_previous(self) -> None:
        """copy main array into previous array, except the top left corner"""
        W_0_0_previous = self.get_previous_block(0, 0)
        self.arr_prev = self.arr.copy()
        self.set_previous_block(0, 0, W_0_0_previous)

    def _get_slice_numbers(self, i: int, j: int) -> np.ndarray:
        """
        Returns array of 4 integers a,b,c,d for accessing
        block_ij = arr[a:b, c:d]
        """
        if (0 <= i * self._k < self._N) and (0 <= j * self._k < self._N):
            return self._as_principal_slice(i, j)
        return self._remap_block_index(i, j)

    def _as_principal_slice(self, i: int, j: int) -> np.ndarray:
        """returns slice index for principal `arr`"""
        idx = _BlockMatrix._as_slice_numbers(i, j, self._k)
        self._validate_index(idx)
        return idx

    def _remap_block_index(self, i: int, j: int) -> np.ndarray:
        """returns slice index for `_Wb_cache`"""
        _i, _j = self._reduce_diagonal(i, j)
        idx = _BlockMatrix._as_slice_numbers(_i, _j, self._k)
        self._validate_index(idx)
        return idx

    def _reduce_diagonal(self, _i: int, _j: int) -> Tuple[int, int]:
        """
        Determine root element of periodic image.
        Result will be nonsensical, unless |i - j| == 1 or 0
        """
        delta = _i - _j
        if delta < 0:
            j = (_j - self._Np) % self._Nc + 1
            i = j + delta
        else:
            i = (_i - self._Np) % self._Nc + 1
            j = i - delta
        return i, j

    def _validate_index(self, idx: np.ndarray) -> None:
        """to avoid getting empty slices"""
        if np.any(idx < 0) or np.any(idx > self._N):
            raise IndexError(f"slice [{idx[0]}:{idx[1]},{idx[2]}:{idx[3]}] is outside "
                             f"the {self._N}x{self._N} principal array.")

    @staticmethod
    def _as_slice_numbers(i: int, j: int, k: int) -> np.ndarray:
        """generic slice indexer for block matrix of k x k blocks"""
        idx = np.array([
            i * k, (i + 1) * k,  # a, b
            j * k, (j + 1) * k  # c, d
        ])
        return idx


class Decimator(_BlockMatrix):
    pass


def decimate(arr: np.ndarray,
             k: int,
             max_tol: float = 1e-8,
             max_iter: int = 100,
             in_place: bool = False,
             flip: bool = False) -> np.ndarray:
    """
    Guinea et al. decimation technique for the surface & bulk Green's functions.

    Consider an array of size M x M that we split into k x k blocks.

    Let:
        W{i} -> block[i][i]
        g{i} -> W{i}^-1
        p{i} -> block[i][i+1]
        m{i} -> block[i-1][i]

    Then, at iteration n = 0, 1, 2... update the array as follows:

        W{0} -> W{0} - p{0} @ g{2^n} @ m{2^n}
        p{0} -> -p{0} @ g{2^n} @ p{2^n}

        W{i} -> W{i} - m{0} @ g{i-2^n} @ p{i-2^n} - p{0} @ g{i+2^n} @ m{i+2^n}
        m{i} -> -m{i} @ g{i-2^n} @ m{i-2^n}
        p{i} -> -p{i} @ g{i+2^n} @ p{i+2^n}

    where

    W{i} is modified to include its diagonal (2^n)th neighbours,
    m{i} is modified to connect {i} to i-2^n}
    p{i} is modified to connect {i} to {i+2^n}

    Eventually, m{i} and p{i} should tend to zero for all i. At this point W{i}
    does not change between iterations.

    :param arr: matrix to decimate
    :param k: largest k such that kth nearest neighbor interaction is non-zero
    :param max_tol: convergence tolerance, smaller -> more work
    :param max_iter: maximum number of iterations
    :param in_place: if False, decimation is done on a copy of the input array
    :param flip: flip input array, compute, and flip back before returning
    :return: the array after a convergent decimation
    """
    if not in_place:
        arr = arr.copy()
    if flip:
        arr = np.flip(arr)

    tol = np.inf
    i = 0
    bm = _BlockMatrix(arr, k)
    while tol > max_tol and i < max_iter:
        _decimate_once(bm, 2**i)

        i += 1
        tol = np.linalg.norm(bm.arr - bm.arr_prev) / np.linalg.norm(bm.arr_prev)
        bm.advance_previous()

    if flip:
        return np.flip(bm.arr)
    return bm.arr