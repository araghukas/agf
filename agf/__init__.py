import numpy as np
from typing import List, Iterable, Dict, Union, Tuple
from dataclasses import dataclass

from agf.utility import (read_fc,
                         slice_top_right,
                         slice_bottom_left,
                         nonzero_submatrix,
                         index_nonzero,
                         fold_matrix,
                         unfold_matrix,
                         flatten_list)

from agf.structure import Layer, StructureSystem


@dataclass
class HarmonicMatrix:
    """
    Represents the sub-matrices of the total harmonic matrix
    for a contact(1)-device(d)-contact(2) structure:

                | H1    tau1†  0    |
        H_tot = | tau1  Hd     tau2 |
                | 0     tau2†  H2   |

    If the harmonic sub-matrices have shapes:

        H1 ~ c1 x c1
        Hd ~ cd x cd
        H2 ~ c2 x c2

    then, the connection matrices have shapes:

        tau1 ~ cd x c1
        tau2 ~ cd x c2

    """

    _H1: np.ndarray
    _tau1: np.ndarray
    _Hd: np.ndarray
    _tau2: np.ndarray
    _H2: np.ndarray

    @property
    def H1(self):
        """harmonic matrix of first contact; H_tot[0,0]"""
        return self._H1

    @property
    def tau1(self):
        """connection matrix between first contact and device; H_tot[1,0]"""
        return self._tau1

    @property
    def Hd(self):
        """harmonic matrix of device region; H_tot[1,1]"""
        return self._Hd

    @property
    def tau2(self):
        """connection matrix between second contact and device; H_tot[1,2]"""
        return self._tau2

    @property
    def H2(self):
        """harmonic matrix of second contact; H_tot[2,2]"""
        return self._H2


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
    def transmission(self):
        try:
            return np.trace(self.Gamma1 @ self.G @ self.Gamma2 @ self.G.conj().T)
        except ValueError:
            return np.trace(self.Gamma1 * self.G * self.Gamma2 * self.G.conj().T)


class AGF(HarmonicMatrix):
    """
    Atomistic Green's Function method calculator.

    See:
        Zhang et al. Numerical Heat Transfer, Part B: Fundamentals 51.4 (2007): 333-349.
    """

    @property
    def force_constants(self) -> np.ndarray:
        """array of all force constants between every pair of atoms [eV/Angstrom]"""
        return self._force_constants

    @property
    def index_map(self) -> Dict[int, int]:
        """a dictionary {atom_id: index in force_constants}"""
        return self._index_map

    @property
    def structure(self) -> StructureSystem:
        """a contact-device-contact structure outlining layers and constituent atoms"""
        return self._struct

    def __init__(self,
                 struct: StructureSystem,
                 force_constants: Union[str, np.ndarray],
                 sort_force_constants: bool = True,
                 n_dof: int = 3):
        """
        Sort force constants and prepare harmonic matrices.
        """

        self._struct = struct
        if type(force_constants) is str:
            force_constants = read_fc(force_constants)

        if sort_force_constants:
            self._force_constants, self._index_map = self._sort_fcs(force_constants)
        else:
            self._force_constants = force_constants
            self._index_map = {i + 1: i for i in range(force_constants.shape[0])}

        self._set_harmonic_matrices(n_dof)

    def _sort_fcs(self, fcs: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
        """sort force constants matrix to correspond to ordering in structure"""
        rearrange = []
        index_map = {}
        layers_id_sequence = flatten_list(self.structure.section_ids)
        for layers_index, id_ in enumerate(layers_id_sequence):
            index = self.structure.locate_atom(id_).index

            rearrange.append(index)
            index_map[id_] = layers_index

        # fold (reshape) force constants so outer elements are atom-wise matrices
        fcs = fold_matrix(fcs, 3, 3)

        # rearrange rows
        rearrange = np.asarray(rearrange)
        fcs = fcs[rearrange]

        # rearrange columns
        for j in range(fcs.shape[0]):
            fcs[j, :] = fcs[j, :][rearrange]

        # unfold into original shape
        fcs = unfold_matrix(fcs)

        return fcs, index_map

    def _set_harmonic_matrices(self, n_dof: int = 3) -> None:
        """assign harmonic and connection matrices according to AGF formulation"""
        ids = self.structure.section_ids

        n1 = sum(len(layer_ids) for layer_ids in ids[0])
        n2 = sum(len(layer_ids) for layer_ids in ids[1])
        n3 = sum(len(layer_ids) for layer_ids in ids[2])

        folded_fcs = fold_matrix(self._force_constants, n_dof, n_dof)

        N = n1 + n2 + n3
        if folded_fcs.shape[:2] != (N, N):
            raise RuntimeError("force constants shape does not match structure")

        # first contact
        a11 = 0
        b11 = n1
        self._H1 = unfold_matrix(folded_fcs[a11:b11, a11:b11])

        # first connection matrix (tau1)
        a1d = n1
        b1d = n1 + n2
        c1d = 0
        d1d = n1
        self._tau1 = unfold_matrix(folded_fcs[a1d:b1d, c1d:d1d])

        # device region
        add = n1
        bdd = n1 + n2
        self._Hd = unfold_matrix(folded_fcs[add:bdd, add:bdd])

        # second connection matrix (tau2)
        ad2 = n1
        bd2 = n1 + n2
        cd2 = n1 + n2
        dd2 = N
        self._tau2 = unfold_matrix(folded_fcs[ad2:bd2, cd2:dd2])

        # second contact
        a22 = n1 + n2
        b22 = N
        self._H2 = unfold_matrix(folded_fcs[a22:b22, a22:b22])

        # decimators for computing uncoupled contact Green's functions
        self._decimate_g1 = Decimation(self._H1, self.structure.contact1, n_dof)
        self._decimate_g2 = Decimation(self._H2, self.structure.contact2, n_dof)

    def compute(self, omega: float, delta: float) -> AGFResult:
        """
        Calculates uncoupled Greens functions using the decimation technique.

        :param omega: frequency at which to compute
        :param delta: broadening factor
        """

        # decimation for contact surface Green's functions
        g1s = self._decimate_g1(omega, delta)
        m1, n1 = g1s.shape
        g2s = self._decimate_g2(omega, delta)
        m2, n2 = g2s.shape

        # TODO: can make smaller, some atoms in one layer don't interact with any in the next
        # extract non-zero sub-matrices
        t1 = slice_top_right(self._tau1, m1, n1)
        t1_H = t1.conj().T
        t2 = slice_bottom_left(self._tau2, m2, n2)
        t2_H = t2.conj().T

        # compute self-energy matrices
        S1 = t1 @ g1s @ t1_H
        S2 = t2 @ g2s @ t2_H

        """final matrix inversion for Green's function"""
        # compute the device Green's function
        Iw = omega * np.eye(self._Hd.shape[0])
        G = np.linalg.inv(Iw - self._Hd - S1 - S2)

        # compute derived quantities
        A1 = 1j * (g1s - g1s.conj().T)
        A2 = 1j * (g2s - g2s.conj().T)
        Gamma1 = t1 @ A1 @ t1_H
        Gamma2 = t2 @ A2 @ t2_H

        return AGFResult(omega, delta, G, Gamma1, Gamma2, A1, A2)


class Decimation:
    """
    Calculates surface and bulk Green's functions given a sorted harmonic matrix, see:

        Sancho et al. Journal of Physics F: Metal Physics, vol. 15, no. 4, Apr. 1985, pp. 851–58.
    """

    @property
    def max_iter(self) -> int:
        """maximum number of iterations before G00 is returned"""
        return self._max_iter

    @max_iter.setter
    def max_iter(self, i):
        self._max_iter = int(i)

    @property
    def iter_tol(self) -> float:
        """minimum value of the change between iterations before G00 is returned"""
        return self._iter_tol

    @iter_tol.setter
    def iter_tol(self, t: float):
        if t < 0:
            raise ValueError("tolerance must be non-negative")
        self._iter_tol = t

    def __init__(self,
                 harmonic_matrix: np.ndarray,
                 layers: List[Layer],
                 n_dof: int = 3):
        self.harmonic_matrix = harmonic_matrix
        self.layers = layers
        self.n_dof = n_dof

        self._es_difference = np.inf
        self._ns = [len(layer.ids) for layer in self.layers]

        # property defaults
        self._iter_tol = np.finfo(float).eps
        self._max_iter = 100

    def __call__(self, omega: float, delta: float) -> np.ndarray:
        return self._calculate_G00_at_omega(omega, delta)

    def _calculate_G00_at_omega(self, omega: float, delta: float) -> np.ndarray:
        """compute surface Green's function at given frequency with given broadening"""
        omega += 1j * delta
        h0 = self._get_H_matrix(0, 0)
        h0s = h1s = h0
        a0 = self._get_H_matrix(1, 0)
        b0 = a0.conj().T
        Iw = omega * np.eye(h0.shape[0])

        """main calculation loop"""
        hs_change = np.inf
        count = 0
        while hs_change > self._iter_tol and count < self._max_iter:
            # some variables to avoid redundancy
            G_bulk = np.linalg.inv(Iw - h0)
            a0_G = a0 @ G_bulk
            b0_G = b0 @ G_bulk
            a0_G_b0 = a0_G @ b0

            # Sancho eq. 11
            a1 = a0_G @ a0
            b1 = b0_G @ b0
            h1 = h0 + a0_G_b0 + b0_G @ a0
            h1s = h0s + a0_G_b0

            hs_change = ((h1s - h0s)**2).mean(axis=None)
            count += 1

            # prepare next iteration
            a0 = a1
            b0 = b1
            h0 = h1
            h0s = h1s

        G00 = np.linalg.inv(omega - h1s)
        return G00

    def _get_H_matrix(self, i: int, j: int) -> np.ndarray:
        """get the harmonic matrix between layers i, j"""
        a = sum(self._ns[:i])
        b = sum(self._ns[:j])
        c = a + self._ns[i]
        d = b + self._ns[j]
        a *= self.n_dof
        b *= self.n_dof
        c *= self.n_dof
        d *= self.n_dof
        return self.harmonic_matrix[a: c, b: d]


def get_zhang_delta(omegas: Iterable[float],
                    c1: float = 1e-3,
                    c2: float = 1e-3,
                    max_val: float = None) -> np.ndarray:
    """
    Frequency broadening function, see:

        Zhang et al. Numerical Heat Transfer, Part B: Fundamentals 51.4 (2007): 333-349.
        Eq. 39
    """
    if max_val is None:
        max_val = max(omegas)
    omegas = np.asarray(omegas)
    return c1 * ((1.0 + c2) - omegas / max_val) * omegas**2


__version__ = "18Aug2021"
