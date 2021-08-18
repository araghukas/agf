from typing import List, Iterable
from dataclasses import dataclass

from agf.utility import (read_fc,
                         slice_top_right,
                         slice_bottom_left,
                         nonzero_submatrix,
                         index_nonzero,
                         fold_matrix,
                         unfold_matrix,
                         flatten_list)

from agf.structure import Layer, CDCStructure, StructureError

import numpy as np


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
        return np.trace(self.Gamma1 @ self.G @ self.Gamma2 @ self.G.conj().T)


class AGF(HarmonicMatrix):
    """
    Atomistic Green's Function method calculator.

    See:
        Zhang et al. Numerical Heat Transfer, Part B: Fundamentals 51.4 (2007): 333-349.
    """

    @property
    def force_constants(self):
        """array of all force constants between every pair of atoms"""
        return self._force_constants

    @property
    def index_map(self):
        """a dictionary {atom_id: index in force_constants}"""
        return self._index_map

    @property
    def structure(self):
        """a contact-device-contact structure outlining layers and constituent atoms"""
        return self._struct

    def __init__(self,
                 struct: CDCStructure,
                 force_constants_file: str):
        """
        Sort force constants and prepare harmonic matrices.
        """

        self._struct = struct
        self._sort_fcs(force_constants_file)
        self._set_harmonic_matrices()

    def _sort_fcs(self, force_constants_file: str) -> None:
        """sort force constants matrix to correspond to layer ordering"""
        fcs = read_fc(force_constants_file)

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

        self._index_map = index_map
        self._force_constants = fcs

    def _set_harmonic_matrices(self) -> None:
        """Assign Harmonic and connection matrices according to AGF formulation"""
        ids = self.structure.section_ids

        n1 = sum(len(layer_ids) for layer_ids in ids[0])
        n2 = sum(len(layer_ids) for layer_ids in ids[1])
        n3 = sum(len(layer_ids) for layer_ids in ids[2])

        folded_fcs = fold_matrix(self._force_constants, 3, 3)

        N = n1 + n2 + n3
        if not (folded_fcs.shape[:2] == (N, N)):
            raise StructureError("force constants shape does not match structure")

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
        self._decimate_g1 = SanchoDecimation(self._H1, self.structure.contact1)
        self._decimate_g2 = SanchoDecimation(self._H2, self.structure.contact2)

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

        # TODO: can slice reduce Hd as well to make this even smaller
        # extract non-zero sub-matrices
        t1 = slice_top_right(self._tau1, m1, n1)
        t1_H = t1.conj().T
        t2 = slice_bottom_left(self._tau2, m2, n2)
        t2_H = t2.conj().T

        # compute self-energy matrices
        S1 = t1 @ g1s @ t1_H
        S2 = t2 @ g2s @ t2_H

        # compute the device Green's function
        Iw = omega * np.eye(self._Hd.shape[0])
        G = np.linalg.inv(Iw - self._Hd - S1 - S2)

        # compute derived quantities
        A1 = 1j * (g1s - g1s.conj().T)
        A2 = 1j * (g2s - g2s.conj().T)
        Gamma1 = t1 @ A1 @ t1_H
        Gamma2 = t2 @ A2 @ t2_H

        return AGFResult(omega, delta, G, Gamma1, Gamma2, A1, A2)


class SanchoDecimation:
    """
    Calculates surface and bulk Green's functions given a sorted harmonic matrix
    """

    @property
    def iter(self) -> int:
        """iteration counter"""
        return self._iter

    @property
    def ns(self) -> List[int]:
        """number of atoms in each layer; used to index sorted harmonic matrix"""
        return self._ns

    def __init__(self,
                 harmonic_matrix: np.ndarray,
                 layers: List[Layer]):
        self.harmonic_matrix = harmonic_matrix
        self.layers = layers
        self._iter = 0
        self._es_difference = np.inf
        self._ns = [len(layer.ids) for layer in self.layers]

    def __post_init__(self):
        m, n, k, l = self.harmonic_matrix.shape
        if not m == n:
            raise ValueError("harmonic matrix is not square")
        if not k == l == 3:
            raise ValueError("harmonic matrix elements must be 3x3 matrices")

        all_ids = []
        for layer in self.layers:
            all_ids += layer.ids
        unique_ids = np.unique(all_ids)
        if len(unique_ids) != m:
            raise ValueError("not enough unique ids for harmonic matrix ({:d}x{:d})"
                             .format(m, n))

    def __call__(self, omega: float, delta: float) -> np.ndarray:
        return self._calculate_G00_at_omega(omega, delta)

    def _calculate_G00_at_omega(self, omega: float, delta: float) -> np.ndarray:
        """compute surface Green's function at given frequency with given broadening"""
        omega += 1j * delta
        e0 = self._get_H_matrix(0, 0)
        es0 = es1 = e0
        a0 = self._get_H_matrix(1, 0)
        b0 = a0.conj().T
        omega *= np.eye(e0.shape[0])

        min_es = np.finfo(float).eps
        while self._es_difference > min_es:
            Gb = np.linalg.inv(omega - e0)

            # to avoid redundancy
            A = a0 @ Gb
            B = b0 @ Gb
            C = A @ b0
            D = B @ a0
            # -------------------

            a1 = A @ a0
            b1 = B @ b0
            e1 = e0 + C + D
            es1 = es0 + C

            self._es_difference = ((es1 - es0)**2).mean(axis=None)

            a0 = a1
            b0 = b1
            e0 = e1
            es0 = es1

        G00 = np.linalg.inv(omega - es1)
        return G00

    def _get_H_matrix(self, i: int, j: int) -> np.ndarray:
        """get the harmonic matrix between layers i, j"""
        H = fold_matrix(self.harmonic_matrix, 3, 3)
        a = sum(self._ns[:i])
        b = sum(self._ns[:j])
        c = a + self._ns[i]
        d = b + self._ns[j]
        sub_matrix = H[a:c, b:d]
        return unfold_matrix(sub_matrix)


def get_zhang_delta(omegas: Iterable[float]) -> np.ndarray:
    """
    Frequency broadening function, see:

        Zhang et al. Numerical Heat Transfer, Part B: Fundamentals 51.4 (2007): 333-349.
        Eq. 39
    """
    omega_max = max(omegas)
    omegas = np.asarray(omegas)
    return 0.001 * (1 - omegas / omega_max) * omegas**2


__version__ = "18Aug2021"
