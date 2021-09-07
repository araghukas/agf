import numpy as np
from typing import Iterable, Union
from dataclasses import dataclass

from agf.utility import slice_bottom_left, slice_top_right, get_block_and_index
from agf.base import _HarmonicMatrices
from agf.sancho_rubio import decimate

from agf.structure import Layer, StructureSystem

__version__ = "18Aug2021"


class AGF(_HarmonicMatrices):
    """
    Atomistic Green's Function method calculator.

    See:
        Zhang et al. Numerical Heat Transfer, Part B: Fundamentals 51.4 (2007): 333-349.
    """

    @property
    def max_iter(self) -> int:
        """maximum number of iterations before gs is returned"""
        return self._max_iter

    @max_iter.setter
    def max_iter(self, i):
        self._max_iter = int(i)

    @property
    def iter_tol(self) -> float:
        """minimum value of the change between iterations before gs is returned"""
        return self._iter_tol

    @iter_tol.setter
    def iter_tol(self, t: float):
        if t < 0:
            raise ValueError("tolerance must be non-negative")
        self._iter_tol = t

    def __init__(self,
                 struct: StructureSystem,
                 force_constants: Union[str, np.ndarray],
                 sort_force_constants: bool = True,
                 n_dof: int = 3):
        super().__init__(struct, force_constants, sort_force_constants, n_dof)

        # property defaults
        self._iter_tol = 1e-6
        self._max_iter = 100

    def compute(self, omega: float, delta: float) -> 'AGFResult':
        """
        Calculates uncoupled Greens functions using the decimation technique.

        :param omega: frequency at which to compute
        :param delta: broadening factor
        """

        # decimation for contact surface Green's functions
        nB = len(self._struct.contact1[0].ids)
        db1 = decimate(self._H_LCB, omega, delta, nB, flip=True)
        db2 = decimate(self._H_RCB, omega, delta, nB)
        g_LCB = np.linalg.inv(db1.Ws)
        g_RCB = np.linalg.inv(db2.Ws)

        w2I1 = (omega**2 + 1.j * delta) * np.ones(g_LCB.shape[0])
        g1s = np.linalg.inv(
            w2I1 - self._H_LC - self._tau_LC_LCB @ g_LCB @ self._tau_LC_LCB.conj().T
        )
        w2I2 = (omega**2 + 1.j * delta) * np.ones(g_RCB.shape[0])
        g2s = np.linalg.inv(
            w2I2 - self._H_RC - self._tau_RC_RCB @ g_RCB @ self.tau_RC_RCB.conj().T
        )

        # compute self-energy matrices
        nd = self._Hd.shape[0]
        se1 = np.zeros((nd, nd), dtype=complex)
        se2 = np.zeros((nd, nd), dtype=complex)
        se1[-self._ns1:, -self._ns1:] = self._tau_LC_LD @ g1s @ self._tau_LC_LD_H
        se2[0:self._ns2, 0:self._ns2] = self._tau_RC_RD @ g2s @ self._tau_RC_RD_H

        # compute the device Green's function
        G = np.linalg.inv(omega**2 * np.eye(nd) - self._Hd - se1 - se2)

        # compute derived quantities
        Gamma1 = 1.j * (se1 - se1.conj().T)
        Gamma2 = 1.j * (se2 - se2.conj().T)

        return AGFResult(omega, delta, G, Gamma1, Gamma2)


@dataclass(frozen=True)
class AGFResult:
    """Results of an AGF computation at some frequency `omega`"""
    omega: float  # frequency
    delta: float  # broadening
    G: np.ndarray  # green's function matrix of the device
    Gamma1: np.ndarray  # Zhang eq. (16)
    Gamma2: np.ndarray

    @property
    def transmission(self) -> float:
        """transmission function at omega"""
        trans = np.trace(self.Gamma1 @ self.G @ self.Gamma2 @ self.G.conj().T)
        return float(trans.real)

    @property
    def dos(self) -> np.ndarray:
        return 1.j * self.omega / np.pi * (self.G - self.G.conj().T)


def get_zhang_delta(omegas: Iterable[float],
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
