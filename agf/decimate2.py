"""Implements the decimation technique for re-normalizing (ωI - H) matrices"""
import numpy as np
from math import log2
from typing import Tuple
from agf.utility import get_block_and_index


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

    def _update_dimensions(self, _k: int) -> None:
        # check `_k` against the array shape
        M = self.arr.shape[0]
        if M % _k != 0:
            raise ValueError(f"can not divide {M}x{M} array into {_k}x{_k} blocks.")

        self._Np = self.N // _k  # side length (in blocks) of complete block matrix
        self._Nc = self._Np - 1  # side length (in blocks) of non-surface block matrix

    def __init__(self, arr: np.ndarray, k: int):
        if arr.ndim != 2:
            raise ValueError(f"input array has {arr.ndim}-dimensional, not 2.")
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(f"input array shape {arr.shape[0]}x{arr.shape[1]} is not square.")

        self.arr = arr
        self.arr_prev = self.arr.copy()  # the previous surface array
        self.N = len(arr)
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

        if (0 <= i * self._k < self.N) and (0 <= j * self._k < self.N):
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

    def _get_slice_numbers(self, i: int, j: int) -> np.ndarray:
        """
        Returns array of 4 integers a,b,c,d for accessing
        block_ij = arr[a:b, c:d]
        """
        if (0 <= i * self._k < self.N) and (0 <= j * self._k < self.N):
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
        if delta >= 0:
            i = (_i - self._Np) % self._Nc + 1
            j = i - delta
        else:
            j = (_j - self._Np) % self._Nc + 1
            i = j + delta
        return i, j

    def _validate_index(self, idx: np.ndarray) -> None:
        """to avoid getting empty slices"""
        if np.any(idx < 0) or np.any(idx > self.N):
            raise IndexError(f"slice [{idx[0]}:{idx[1]},{idx[2]}:{idx[3]}] is outside "
                             f"the {self.N}x{self.N} principal array.")

    @staticmethod
    def _as_slice_numbers(i: int, j: int, k: int) -> np.ndarray:
        """generic slice indexer for block matrix of k x k blocks"""
        idx = np.array([
            i * k, (i + 1) * k,  # a, b
            j * k, (j + 1) * k  # c, d
        ])
        return idx


def decimate(arr: np.ndarray,
             k: int,
             max_tol: float = 1e-8,
             max_iter: int = 100,
             semi_infinite: bool = True,
             in_place: bool = False,
             flip: bool = False) -> np.ndarray:
    """
    Guinea et al. decimation technique for the surface & bulk Green's functions.

    :param arr: matrix to decimate
    :param k: largest k such that kth nearest neighbor interaction is non-zero
    :param max_tol: convergence tolerance, smaller -> more work
    :param max_iter: maximum number of iterations
    :param semi_infinite: assume non-surface layers repeat indefinitely
    :param in_place: if False, decimation is done on a copy of the input array
    :param flip: flip input array, compute, and flip back before returning
    :return: the array after a convergent decimation
    """
    if not in_place:
        arr = arr.copy()
    if flip:
        arr = np.flip(arr)

    # decimation strategy
    if semi_infinite:
        _decimate = _decimate_once_infinite
    else:
        _decimate = _decimate_once_finite

    # TODO: these checks are no longer necessary -> instantiate _BlockMatrix
    M, N = arr.shape
    if M != N:
        raise ValueError(f"input array shape {M}x{N} is not square.")
    if M % k != 0:
        raise ValueError(f"can not divide {M}x{M} array into {k}x{k} blocks.")
    m = M // k

    max_doubling = int(log2(m) - 1)
    if max_doubling < 0:
        raise ValueError(f"block matrix is too small for decimation on {k}x{k} blocks.")

    old_arr = arr.copy()
    d = 0  # iteration counter and doubling exponent parameter
    tol = np.inf  # for change of effective hamiltonian between iterations
    while tol > max_tol and d < max_doubling and d < max_iter:
        s = 2**(d + 1)  # inter-layer stride size

        arr = _decimate_once_finite(arr, k, s, m)

        tol = np.linalg.norm(arr - old_arr) / np.linalg.norm(old_arr)
        d += 1
        old_arr = arr.copy()

    if flip:
        return np.flip(arr)
    return arr


def _decimate_once_finite(arr: np.ndarray, k: int, s: int, m: int) -> np.ndarray:
    """The actual algorithm for one iteration of decimating `arr`."""
    r = s // 2

    # update upper left edge
    # ----------------------
    W_00, iW = get_block_and_index(arr, k, 0, 0, safe=True)
    t_0_0p, i1 = get_block_and_index(arr, k, 0, r, safe=True)

    t_0p_0 = get_block_and_index(arr, k, r, 0, safe=True)[0]
    t_0p_0pp = get_block_and_index(arr, k, r, 2 * r, safe=True)[0]
    W_0p_0p = get_block_and_index(arr, k, r, r, safe=True)[0]

    g_11 = np.linalg.inv(W_0p_0p)
    arr[iW[0]:iW[1], iW[2]:iW[3]] = W_00 - t_0_0p @ g_11 @ t_0p_0  # new W_00
    arr[i1[0]:i1[1], i1[2]:i1[3]] = -t_0_0p @ g_11 @ t_0p_0pp  # new t_0_0p

    # interior
    # --------
    j = 1
    while s * (j + 2) < m:
        i = s * j
        # indices name legend:
        # im -> i-1
        # ip -> i+1
        # imm -> i-2
        # ipp -> i+2

        W_ii, iW = get_block_and_index(arr, k, i, i, safe=True)
        t_i_im, i0 = get_block_and_index(arr, k, i, i - r, safe=True)
        t_i_ip, i1 = get_block_and_index(arr, k, i, i + r, safe=True)

        W_im_im = get_block_and_index(arr, k, i - r, i - r, safe=True)[0]
        W_ip_ip = get_block_and_index(arr, k, i + r, i + r, safe=True)[0]
        t_im_i = get_block_and_index(arr, k, i - r, i, safe=True)[0]
        t_ip_i = get_block_and_index(arr, k, i + r, i, safe=True)[0]
        t_im_imm = get_block_and_index(arr, k, i - r, i - 2 * r, safe=True)[0]
        t_ip_ipp = get_block_and_index(arr, k, i + r, i + 2 * r, safe=True)[0]

        g_im_im = np.linalg.inv(W_im_im)
        g_ip_ip = np.linalg.inv(W_ip_ip)
        arr[iW[0]:iW[1], iW[2]:iW[3]] = (
                W_ii - t_i_im @ g_im_im @ t_im_i - t_i_ip @ g_ip_ip @ t_ip_i  # new W_ii
        )
        arr[i0[0]:i0[1], i0[2]:i0[3]] = -t_i_im @ g_im_im @ t_im_imm
        arr[i1[0]:i1[1], i1[2]:i1[3]] = -t_i_ip @ g_ip_ip @ t_ip_ipp

        j += 1

    return arr


def _decimate_once_infinite(arr: np.ndarray, k: int, s: int, m: int) -> np.ndarray:
    """The actual algorithm for one iteration of decimating `arr`."""
    r = s // 2

    # update upper left edge
    # ----------------------
    W_00, iW = get_block_and_index(arr, k, 0, 0, safe=True)
    t_0_0p, i1 = get_block_and_index(arr, k, 0, r, safe=True)

    t_0p_0 = get_block_and_index(arr, k, r, 0, safe=True)[0]
    t_0p_0pp = get_block_and_index(arr, k, r, 2 * r, safe=True)[0]
    W_0p_0p = get_block_and_index(arr, k, r, r, safe=True)[0]

    g_11 = np.linalg.inv(W_0p_0p)
    arr[iW[0]:iW[1], iW[2]:iW[3]] = W_00 - t_0_0p @ g_11 @ t_0p_0  # new W_00
    arr[i1[0]:i1[1], i1[2]:i1[3]] = -t_0_0p @ g_11 @ t_0p_0pp  # new t_0_0p

    # interior
    # --------
    j = 1
    while s * (j + 2) < m:
        i = s * j
        # indices name legend:
        # im -> i-1
        # ip -> i+1
        # imm -> i-2
        # ipp -> i+2

        W_ii, iW = get_block_and_index(arr, k, i, i, safe=True)
        t_i_im, i0 = get_block_and_index(arr, k, i, i - r, safe=True)
        t_i_ip, i1 = get_block_and_index(arr, k, i, i + r, safe=True)

        W_im_im = get_block_and_index(arr, k, i - r, i - r, safe=True)[0]
        W_ip_ip = get_block_and_index(arr, k, i + r, i + r, safe=True)[0]
        t_im_i = get_block_and_index(arr, k, i - r, i, safe=True)[0]
        t_ip_i = get_block_and_index(arr, k, i + r, i, safe=True)[0]
        t_im_imm = get_block_and_index(arr, k, i - r, i - 2 * r, safe=True)[0]
        t_ip_ipp = get_block_and_index(arr, k, i + r, i + 2 * r, safe=True)[0]

        g_im_im = np.linalg.inv(W_im_im)
        g_ip_ip = np.linalg.inv(W_ip_ip)
        arr[iW[0]:iW[1], iW[2]:iW[3]] = (
                W_ii - t_i_im @ g_im_im @ t_im_i - t_i_ip @ g_ip_ip @ t_ip_i  # new W_ii
        )
        arr[i0[0]:i0[1], i0[2]:i0[3]] = -t_i_im @ g_im_im @ t_im_imm
        arr[i1[0]:i1[1], i1[2]:i1[3]] = -t_i_ip @ g_ip_ip @ t_ip_ipp

        j += 1

    return arr
