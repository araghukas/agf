""""
The Sancho-Rubio decimation technique is implemented here.

See:
    "Highly convergent schemes for the calculation of bulk and surface Green functions"
    Sancho, M. P. L. et al., Journal of Physics F: Metal Physics 15 (4): 851–58 (1985)
"""
import numpy as np

from typing import Tuple
from dataclasses import dataclass

from agf.utility import fold_matrix


@dataclass(frozen=True)
class DecimationResult:
    Ws: np.ndarray  # effective matrix (Ιω**2 - Hs) for surface
    Wb: np.ndarray  # ... for bulk
    a: np.ndarray  # surface connection matrix
    b: np.ndarray  # conjugate surface connection matrix


def decimate(arr: np.ndarray,
             omega: float,
             delta: float,
             layer_size: int,
             abs_tol: float = 1e-6,
             homogeneous: bool = True,
             flip: bool = False) -> DecimationResult:
    """
    Implements the decimation algorithm described by Sancho-Rubio.

    :param arr: assumed to be (Ιω**2 - H) in the eq. (Ιω**2 - H)G = I.
    :param omega: angular frequency
    :param delta: frequency broadening
    :param layer_size: number of atoms per layers times the number of degrees of freedom.
    :param abs_tol: convergence tolerance for norms of connection matrices (default 1e-6)
    :param homogeneous: assume the matrix has repeating elements (i.e. layers).
    :param flip: flip the matrix before decimation
    :return: the effective surface and bulk matrices (Iw**2 - H)
    """
    if not arr.ndim == 2:
        raise ValueError("input array is not 2-dimensional")

    M = arr.shape[0]
    if not arr.shape[1] == M:
        raise ValueError("input array is not square.")
    if arr.shape[0] % layer_size != 0:
        raise ValueError(
            f"can not divide {M}x{M} array into {layer_size}x{layer_size} blocks.")

    arr = fold_matrix(arr, layer_size, layer_size)  # returns a new array of blocks

    if homogeneous:
        w2I = (omega**2 + 1.j * delta) * np.eye(layer_size)
        a = np.ascontiguousarray(arr[-1][-2] if flip else arr[0][1])
        b = np.ascontiguousarray(a.conj().T)
        eps = np.ascontiguousarray(arr[-1][-1] if flip else arr[0][0])
        Ws, Wb, a, b = _homogeneous_decimation(w2I, a, b, eps, abs_tol)
    else:
        raise NotImplementedError(
            "sorry, can't decimate for inhomogeneous contacts yet.")

    return DecimationResult(Ws, Wb, a, b)


def _homogeneous_decimation(omega: np.ndarray,
                            a: np.ndarray,
                            b: np.ndarray,
                            eps: np.ndarray,
                            max_tol: float) -> Tuple[np.ndarray,
                                                     np.ndarray,
                                                     np.ndarray,
                                                     np.ndarray]:
    tol = np.inf
    eps_s = eps.copy()
    while tol > max_tol:
        g = np.linalg.inv(omega - eps)

        # update diagonal elements
        agb = a @ g @ b  # no need to compute twice
        eps_s = eps_s + agb
        eps = eps + agb + b @ g @ a

        # update connection elements
        b = b @ g @ b
        a = a @ g @ a

        # update tolerance value
        tol = np.linalg.norm(a) + np.linalg.norm(b)

    Ws = omega - eps_s
    Wb = omega - eps
    return Ws, Wb, a, b
