"""Implements the decimation technique for re-normalizing (ωI - H) matrices"""
import numpy as np
from agf.utility import get_block


def decimate(arr: np.ndarray,
             k: int,
             max_tol: float = 1e-8,
             max_iter: int = 100,
             in_place: bool = False,
             flip: bool = False) -> np.ndarray:
    """
    Implements Guinea et al. decimation technique for surface Green's function.
    Note: the input array is decimated IN PLACE.

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

    M, N = arr.shape
    if M != N:
        raise ValueError(f"input array shape {M}x{N} is not square.")
    if M % k != 0:
        raise ValueError(f"can not divide {M}x{M} array into {k}x{k} blocks.")

    m = M // k
    i = 0
    tol = np.inf
    old_arr = arr.copy()
    while tol > max_tol and i < max_iter:
        arr = _decimate_once(arr, k, m)
        tol = np.linalg.norm(arr - old_arr) / np.linalg.norm(old_arr)
        i += 1
        old_arr = arr.copy()

    if flip:
        return np.flip(arr)
    return arr


def _decimate_once(arr: np.ndarray, k: int, m: int) -> np.ndarray:
    """decimate a (k*m) x (k*m) array made of k x k blocks"""

    # upper left edge case (i = 0)
    # get refs. to initial values
    W_00, iW = get_block(arr, k, 0, 0)
    t_01, i1 = get_block(arr, k, 0, 1)

    # adjacent blocks for updating values
    W_11 = get_block(arr, k, 1, 1)[0]
    t_10 = get_block(arr, k, 1, 0)[0]
    t_12 = get_block(arr, k, 1, 2)[0]

    # compute inversions
    g_11 = np.linalg.inv(W_11)

    # update values at refs.
    arr[iW[0]:iW[1], iW[2]:iW[3]] = W_00 - t_01 @ g_11 @ t_10  # W_00
    arr[i1[0]:i1[1], i1[2]:i1[3]] = t_01 @ -g_11 @ t_12  # t_01

    # repeat for interior blocks
    for i in range(1, m - 1):
        # indices name legend:
        # im -> i-1
        # ip -> i+1
        # imm -> i-2
        # ipp -> i+2

        # principal blocks at this iteration
        W_ii, iW = get_block(arr, k, i, i)
        t_im_i, i0 = get_block(arr, k, i - 1, i)
        t_ip_i, i1 = get_block(arr, k, i + 1, i)

        # adjacent blocks for updating values
        W_im_im = get_block(arr, k, i - 1, i - 1)[0]
        W_ip_ip = get_block(arr, k, i + 1, i + 1)[0]

        t_i_im = get_block(arr, k, i, i - 1)[0]
        t_i_ip = get_block(arr, k, i, i + 1)[0]

        t_ip_ipp = get_block(arr, k, i + 1, i + 2)[0]
        t_im_imm = get_block(arr, k, i - 1, i - 2)[0]

        # compute inversions
        g_ip_ip = np.linalg.inv(W_ip_ip)
        g_im_im = np.linalg.inv(W_im_im)

        # update values at refs.
        arr[iW[0]:iW[1], iW[2]:iW[3]] = (  # new W_ii
                W_ii - t_i_im @ g_im_im @ t_im_i + t_i_ip @ g_ip_ip @ t_ip_i
        )
        arr[i0[0]:i0[1], i0[2]:i0[3]] = -t_i_im @ g_im_im @ t_im_imm  # new t_im_i
        arr[i1[0]:i1[1], i1[2]:i1[3]] = -t_i_ip @ g_ip_ip @ t_ip_ipp  # new t_ip_i

    return arr
