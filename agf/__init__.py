import numpy as np
from warnings import warn
from typing import List, Iterable, Union
from dataclasses import dataclass

from agf.utility import slice_bottom_left, slice_top_right
from agf.base import _HarmonicMatrices

from agf.structure import Layer, StructureSystem

__version__ = "18Aug2021"


class Decimation:
    """
    Calculates surface and bulk Green's functions given a sorted harmonic matrix,
    see:

        Guinea, F., et al. Physical Review B, vol. 28, no. 8, Oct. 1983, pp. 4397–402.
        Sancho et al. Journal of Physics F: Metal Physics, vol. 15, no. 4, Apr. 1985, pp. 851–58.
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
                 harmonic_matrix: np.ndarray,
                 layers: List[Layer],
                 surface_index: int,
                 n_dof: int = 3):

        self._harmonic_matrix = harmonic_matrix
        self._layers = layers
        self._ns = [len(layer.ids) for layer in self._layers]
        if n_dof > 0:
            self._n_dof = n_dof
        else:
            raise ValueError("number of degrees of freedom must be a positive integer.")

        # property defaults
        self._iter_tol = np.finfo(float).eps
        self._max_iter = 100
        self._inverse_order = False

        # set matrices Hs, tau1, tau2
        N_layers = len(self._layers)
        surface_index %= N_layers
        if not (surface_index == 0 or surface_index == N_layers - 1):
            raise ValueError("surface index does not correspond to and edge layer.")

        if surface_index != 0:
            # flip everything around for second contact
            self._harmonic_matrix = np.flip(self._harmonic_matrix)
            self._layers.reverse()
            self._inverse_order = True

        # assign starting values
        self._Hs = self._get_H_submatrix(0, 0)
        self._t1 = self._get_H_submatrix(0, 1)
        self._t2 = self._get_H_submatrix(1, 0)

    def __call__(self, omega: float, delta: float) -> np.ndarray:
        gs = self._calculate_gs_at_omega(omega, delta)
        if self._inverse_order:
            return np.flip(gs)
        return gs

    def _calculate_gs_at_omega(self, omega: float, delta: float) -> np.ndarray:
        """compute surface Green's function at given frequency with given broadening"""
        w2I = (omega**2 + 1j * delta) * np.eye(self._n_dof * self._ns[0])

        # input values
        Ws = Wb = w2I - self._Hs  # TODO: simplifying symmetry rules assumed
        t1 = -self._t1
        t2 = -self._t2

        # termination/convergence conditions
        Ws_change = np.inf
        Ws_change_min = self._iter_tol
        count = 0
        count_max = self._max_iter
        while Ws_change > Ws_change_min and count < count_max:
            # this loop implements Guinea eqs. 15
            Gb = np.linalg.inv(Wb)
            Gb12 = t1 @ Gb @ t2
            Gb21 = t2 @ Gb @ t1
            Ws_prev = Ws

            # update values
            Ws = Ws - Gb12
            Wb = Wb - Gb12 - Gb21
            t1 = -t1 @ Gb @ t1
            t2 = -t2 @ Gb @ t2

            Ws_change = np.linalg.norm(Ws - Ws_prev) / np.linalg.norm(Ws_prev)
            count += 1

        if count > count_max:
            warn(f"exceeded max number of iterations in decimation at freq. {omega}")

        G00 = np.linalg.inv(Ws)
        return G00

    def _get_H_submatrix(self, i: int, j: int) -> np.ndarray:
        """get the harmonic matrix between layers i, j"""
        a = sum(self._ns[:i])
        b = sum(self._ns[:j])
        c = a + self._ns[i]
        d = b + self._ns[j]
        a *= self._n_dof
        b *= self._n_dof
        c *= self._n_dof
        d *= self._n_dof
        return self._harmonic_matrix[a: c, b: d]


class AGF(_HarmonicMatrices):
    """
    Atomistic Green's Function method calculator.

    See:
        Zhang et al. Numerical Heat Transfer, Part B: Fundamentals 51.4 (2007): 333-349.
    """

    def __init__(self,
                 struct: StructureSystem,
                 force_constants: Union[str, np.ndarray],
                 sort_force_constants: bool = True,
                 n_dof: int = 3):
        super().__init__(struct, force_constants, sort_force_constants, n_dof)
        self._decimate_g1 = Decimation(self._H1, self.structure.contact1, 0, self._n_dof)
        self._decimate_g2 = Decimation(self._H2, self.structure.contact2, -1, self._n_dof)

    def set_iteration_limits(self, iter_tol: float = None, max_iter: int = None) -> None:
        """
        Set tolerance and maximum number of iterations for decimations

        :param iter_tol: minimum value of change between iterations
        :param max_iter: maximum number of iterations
        """
        if iter_tol is not None:
            self._decimate_g1.iter_tol = iter_tol
            self._decimate_g2.iter_tol = iter_tol
        if max_iter is not None:
            self._decimate_g1.max_iter = max_iter
            self._decimate_g2.max_iter = max_iter

    def compute(self, omega: float, delta: float) -> 'AGFResult':
        """
        Calculates uncoupled Greens functions using the decimation technique.

        :param omega: frequency at which to compute
        :param delta: broadening factor
        """

        # decimation for contact surface Green's functions
        g1s = self._decimate_g1(omega, delta)
        g2s = self._decimate_g2(omega, delta)
        m1, n1 = g1s.shape
        m2, n2 = g2s.shape
        md = self._Hd.shape[0]

        # TODO: can make smaller, some atoms in one layer don't interact with any in the next
        # extract non-zero sub-matrices
        t1 = slice_top_right(self._tau1, m1, n1)
        t1_H = t1.conj().T
        t2 = slice_bottom_left(self._tau2, m2, n2)
        t2_H = t2.conj().T

        # compute self-energy matrices
        se1 = t1 @ g1s @ t1_H
        se2 = t2 @ g2s @ t2_H

        # pad the self energy matrices if necessary
        if m1 != md or n1 != md:
            se1 = np.pad(t1 @ g1s @ t1_H, ((0, md - m1), (0, md - n1)), constant_values=0.j)
        if m2 != md or n2 != md:
            se2 = np.pad(t2 @ g2s @ t2_H, ((md - m2, 0), (md - n2, 0)), constant_values=0.j)

        # compute the device Green's function
        w2I = (omega**2 + 1j * delta) * np.eye(md)
        G = np.linalg.inv(w2I - self._Hd - se1 - se2)

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
        return float(trans)


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
