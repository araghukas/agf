""""
The Sancho-Rubio decimation technique is implemented here.

See:
    "Highly convergent schemes for the calculation of bulk and surface Green functions"
    Sancho, M. P. L. et al., Journal of Physics F: Metal Physics 15 (4): 851–58 (1985)
"""
import numpy as np
from numpy.linalg import norm, inv
from typing import Tuple

from agf.utility import fold_matrix


def decimate(arr: np.ndarray,
             omega: float,
             delta: float,
             d: int,
             abs_tol: float = 1e-6,
             homogeneous: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements the decimation algorithm described by Sancho-Rubio.

    :param arr: assumed to be (Ιω**2 - H) in the eq. (Ιω**2 - H)G = I.
    :param omega: angular frequency
    :param delta: frequency broadening
    :param d: the block size (number of degrees of freedom).
    :param abs_tol: convergence tolerance for norm of connection matrix (default 1e-6)
    :param homogeneous: assume the matrix has repeating elements (i.e. layers).
    :return: the effective surface and bulk matrices (Iw**2 - H)
    """
    if not arr.ndim == 2:
        raise ValueError("input array is not 2-dimensional")

    M = arr.shape[0]
    if not arr.shape[1] == M:
        raise ValueError("input array is not square.")
    if arr.shape[0] % d != 0:
        raise ValueError(f"can not divide {M}x{M} array into {d}x{d} blocks.")

    arr = fold_matrix(arr, d, d)  # returns a new array of blocks

    if homogeneous:
        w2I = (omega**2 + 1.j * delta) * np.eye(d)
        a = arr[0][1]
        b = a.conj().T
        eps = arr[0][0]
        W0, Wb = _homogeneous_decimation(w2I, a, b, eps, abs_tol)
    else:
        raise NotImplementedError(
            "sorry, can't decimate for inhomogeneous contacts yet.")

    return W0, Wb


def _homogeneous_decimation(omega: np.ndarray,
                            a: np.ndarray,
                            b: np.ndarray,
                            eps: np.ndarray,
                            max_tol: float) -> Tuple[np.ndarray, np.ndarray]:
    tol = np.inf
    eps_s = eps.copy()
    while tol > max_tol:
        g = inv(omega - eps)

        # update diagonal elements
        agb = a @ g @ b  # no need to compute twice
        eps_s = eps_s + agb
        eps = eps + agb + b @ g @ a

        # update connection elements
        b = b @ g @ b
        a = a @ g @ a

        # update tolerance value
        tol = norm(a)

    W0 = omega - eps_s
    Wb = omega - eps
    return W0, Wb
