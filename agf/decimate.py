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

    # upper left edge (i = 0)
    # -----------------------
    W_00, iW = get_block(arr, k, 0, 0)
    t_0_0p, i1 = get_block(arr, k, 0, 1)

    t_0p_0 = get_block(arr, k, 1, 0)[0]
    t_0p_0pp = get_block(arr, k, 1, 2)[0]
    W_0p_0p = get_block(arr, k, 1, 1)[0]

    g_11 = np.linalg.inv(W_0p_0p)
    arr[iW[0]:iW[1], iW[2]:iW[3]] = W_00 - t_0_0p @ g_11 @ t_0p_0  # new W_00
    arr[i1[0]:i1[1], i1[2]:i1[3]] = -t_0_0p @ g_11 @ t_0p_0pp  # new t_0_0p

    # repeat for interior blocks
    # --------------------------
    for i in range(1, m - 1):
        # indices name legend:
        # im -> i-1
        # ip -> i+1
        # imm -> i-2
        # ipp -> i+2
        W_ii, iW = get_block(arr, k, i, i)
        t_i_im, i0 = get_block(arr, k, i, i - 1)
        t_i_ip, i1 = get_block(arr, k, i, i + 1)

        W_im_im = get_block(arr, k, i - 1, i - 1)[0]
        W_ip_ip = get_block(arr, k, i + 1, i + 1)[0]
        t_im_i = get_block(arr, k, i - 1, i)[0]
        t_ip_i = get_block(arr, k, i + 1, i)[0]
        t_im_imm = get_block(arr, k, i - 1, i - 2)[0]
        t_ip_ipp = get_block(arr, k, i + 1, i + 2)[0]

        g_im_im = np.linalg.inv(W_im_im)
        g_ip_ip = np.linalg.inv(W_ip_ip)
        arr[iW[0]:iW[1], iW[2]:iW[3]] = (
                W_ii - t_i_im @ g_im_im @ t_im_i - t_i_ip @ g_ip_ip @ t_ip_i  # new W_ii
        )
        arr[i0[0]:i0[1], i0[2]:i0[3]] = -t_i_im @ g_im_im @ t_im_imm
        arr[i1[0]:i1[1], i1[2]:i1[3]] = -t_i_ip @ g_ip_ip @ t_ip_ipp
        print(arr)

    # lower right edge (i = m - 1 = f)
    # --------------------------------
    f = m - 1
    W_ff, iW = get_block(arr, k, f, f)
    t_f_fm, i0 = get_block(arr, k, f, f - 1)

    W_fm_fm = get_block(arr, k, f - 1, f - 1)[0]
    t_fm_f = get_block(arr, k, f - 1, f)[0]
    t_fm_fmm = get_block(arr, k, f - 1, f - 2)[0]

    g_fm_fm = np.linalg.inv(W_fm_fm)
    arr[iW[0]:iW[1], iW[2]:iW[3]] = W_ff - t_f_fm @ g_fm_fm @ t_fm_f
    arr[i0[0]:i0[1], i0[2]:i0[3]] = -t_f_fm @ g_fm_fm @ t_fm_fmm

    return arr
