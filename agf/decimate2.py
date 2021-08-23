"""Implements the decimation technique for re-normalizing (ωI - H) matrices"""
import numpy as np
from math import log2
from typing import Tuple
from agf.utility import get_block_and_index


class _BlockMatrix:
    """wraps a 2D array and provides easy access to blocks"""

    @property
    def Ws(self):
        return np.ascontiguousarray(self._Ws)

    @property
    def t_sb(self):
        return np.ascontiguousarray(self._t_sb)

    @property
    def t_bs(self):
        return np.ascontiguousarray(self._t_bs)

    @property
    def k(self) -> int:
        """block matrix size"""
        return self._k

    @k.setter
    def k(self, _k):
        self._k = _k
        self._assign_matrices(_k)

    def _assign_matrices(self, _k: int) -> None:
        # assign principal matrices
        a, b, c, d = _BlockMatrix._as_slice_numbers(0, 1, _k)
        self._t_sb = self.arr[a:b, c:d]

        a, b, c, d = _BlockMatrix._as_slice_numbers(1, 0, _k)
        self._t_bs = self.arr[a:b, c:d]

        a, b, c, d = _BlockMatrix._as_slice_numbers(0, 0, _k)
        self._Ws = self.arr[a:b, c:d]  # the surface array
        self._Wc = self.arr.copy()  # the image array

        self._Np = self.N // _k  # number of principal blocks
        self._Nc = self._Np - 1  # number of image cache blocks

    def __init__(self, arr: np.ndarray, k: int, s: int):
        self.arr = arr
        self.N = len(arr)
        self.k = k
        self.s = s

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
            # return a principal block
            a, b, c, d = self._as_principal_slice(i, j)
            print("principal: ", a, b, c, d)
            return np.ascontiguousarray(self.arr[a:b, c:d])

        # return a 'fictitious' block
        a, b, c, d = self._remap_block_index(i, j)
        print("chached: ", a, b, c, d)
        return np.ascontiguousarray(self._Wc[a:b, c:d])

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

    def _reduce_diagonal(self, i: int, j: int) -> Tuple[int, int]:
        """determine root element of periodic image"""
        _i = i
        _j = j
        while _i >= self._Np or _j >= self._Np:
            _i -= self._Nc
            _j = _i - (i - j)
        return _i, _j

    def _validate_index(self, idx: np.ndarray) -> None:
        """to avoid getting empty slices"""
        if any(i < 0 for i in idx) or any(i > self.N for i in idx):
            raise IndexError(f"slice [{idx[0]}:{idx[1]},{idx[2]}:{idx[3]}] is outside "
                             f"the {self.N}x{self.N} array.")

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
