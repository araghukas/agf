import numpy as np
from typing import Iterable, Union
from dataclasses import dataclass

from agf.utility import slice_bottom_left, slice_top_right, get_block_and_index
from agf.base import _HarmonicMatrices
from agf.decimate import decimate

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
        self._iter_tol = 1e-8
        self._max_iter = 100

    def compute(self, omega: float, delta: float) -> 'AGFResult':
        """
        Calculates uncoupled Greens functions using the decimation technique.

        :param omega: frequency at which to compute
        :param delta: broadening factor
        """

        # decimation for contact surface Green's functions
        ns1 = self._struct.contact1[-1].N
        ns2 = self._struct.contact2[0].N
        w2I1 = (omega**2 + 1j * delta) * np.eye(self._struct.n1)
        w2I2 = (omega**2 + 1j * delta) * np.eye(self._struct.n2)
        W1 = decimate(w2I1 - self._H1, self._n_dof)
        W2 = decimate(w2I2 - self._H2, self._n_dof)
        print(W2)
        g1s = np.linalg.inv(get_block_and_index(W1, ns1, 0, 0)[0])
        g2s = np.linalg.inv(get_block_and_index(W2, ns2, 0, 0)[0])

        # TODO: can make smaller, some atoms in one layer don't interact with any in the next
        # extract non-zero sub-matrices
        t1 = slice_top_right(self._tau1, ns1, ns1)
        t1_H = t1.conj().T
        t2 = slice_bottom_left(self._tau2, ns2, ns2)
        t2_H = t2.conj().T

        # compute self-energy matrices
        se1 = t1 @ g1s @ t1_H
        se2 = t2 @ g2s @ t2_H

        # pad the self energy matrices if necessary
        nd = self._struct.nd
        if ns1 != nd or ns1 != nd:
            se1 = np.pad(se1, ((0, nd - ns1), (0, nd - ns1)), constant_values=0.j)
        if ns2 != nd or ns2 != nd:
            se2 = np.pad(se2, ((nd - ns2, 0), (nd - ns2, 0)), constant_values=0.j)

        # compute the device Green's function
        w2Id = (omega**2 + 1j * delta) * np.eye(nd)
        G = np.linalg.inv(w2Id - self._Hd - se1 - se2)

        # compute derived quantities
        A1 = 1j * (g1s - g1s.conj().T)
        A2 = 1j * (g2s - g2s.conj().T)
        Gamma1 = t1 @ A1 @ t1_H
        Gamma2 = t2 @ A2 @ t2_H

        return AGFResult(omega, delta, G, Gamma1, Gamma2, A1, A2)


@dataclass(frozen=True)
class AGFResult:
    """Results of an AGF computation at some frequency `omega`"""
    omega: float  # frequency
    delta: float  # broadening
    G: np.ndarray  # green's function matrix of the device
    Gamma1: np.ndarray  # Zhang eq. (16)
    Gamma2: np.ndarray
    A1: np.ndarray  # Zhang eq. (15)
    A2: np.ndarray

    @property
    def transmission(self) -> float:
        """transmission function at omega"""
        try:
            trans = np.trace(self.Gamma1 @ self.G @ self.Gamma2 @ self.G.conj().T)
        except ValueError:
            trans = np.trace(self.Gamma1 * self.G * self.Gamma2 * self.G.conj().T)
        return float(trans.real)


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
