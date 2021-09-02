"""
Base classes for setting up calculations, but not running them.
"""
import numpy as np
from typing import Tuple, Dict, Union

from agf.structure import StructureSystem
from agf.utility import (fold_matrix,
                         unfold_matrix,
                         read_fc,
                         flatten_list,
                         slice_top_right,
                         slice_bottom_left)


class _HarmonicMatrices:
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

    @property
    def H_tot(self):
        """total harmonic matrix"""
        return self._force_constants

    @property
    def H1(self):
        """harmonic matrix of first contact; H_tot[0,0]"""
        return self._H1.copy()

    @property
    def H_LCB(self):
        """... of first contact bulk"""
        return self._H_LCB.copy()

    @property
    def H_LC(self):
        """... of first contact surface"""
        return self._H_LC.copy()

    @property
    def tau1(self):
        """connection matrix between first contact and device; H_tot[1,0]"""
        return self._tau1.copy()

    @property
    def tau_LC_LCB(self):
        """... between surface and bulk of the first contact"""
        return self._tau_LC_LCB.copy()

    @property
    def Hd(self):
        """harmonic matrix of device region; H_tot[1,1]"""
        return self._Hd.copy()

    @property
    def tau2(self):
        """connection matrix between second contact and device; H_tot[1,2]"""
        return self._tau2.copy()

    @property
    def tau_RC_RCB(self):
        """... between surface and bulk of second contact"""
        return self._tau_RC_RCB.copy()

    @property
    def H2(self):
        """harmonic matrix of second contact; H_tot[2,2]"""
        return self._H2.copy()

    @property
    def H_RC(self):
        """... of second contact surface"""
        return self._H_RC.copy()

    @property
    def H_RCB(self):
        """... of second contact bulk"""
        return self._H_RCB.copy()

    @property
    def force_constants(self) -> np.ndarray:
        """array of all force constants between every pair of atoms [eV/Angstrom]"""
        return self._force_constants.copy()

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
        Build harmonic sub-matrices given a structure and all the force constants

        :param struct: StructureSystem object containing layer/atom information
        :param force_constants: a matrix of harmonic constants for the system
        :param sort_force_constants: if True, sort force constants in structure order
        :param n_dof: number of degrees of freedom
        """
        self._n_dof = n_dof
        self._struct = struct
        if type(force_constants) is str:
            force_constants = read_fc(force_constants)

        if sort_force_constants:
            self._force_constants, self._index_map = self._sort_fcs(force_constants)
        else:
            self._force_constants = force_constants
            self._index_map = {i + 1: i for i in range(force_constants.shape[0])}

        self._force_constants = self._force_constants.astype(np.complex128)

        # some counting for matrix slicing
        section_ids = self.structure.section_ids
        nc1 = sum(len(layer_ids) for layer_ids in section_ids[0])
        nd = sum(len(layer_ids) for layer_ids in section_ids[1])
        nc2 = sum(len(layer_ids) for layer_ids in section_ids[2])
        nLCB = sum(len(layer_ids) for layer_ids in section_ids[0][:-1])
        nRCB = sum(len(layer_ids) for layer_ids in section_ids[-1][1:])
        ns1 = self._struct.contact1[-1].N
        ns2 = self._struct.contact2[0].N

        # fold the matrix for easier indexing
        folded_fcs = fold_matrix(self._force_constants, self._n_dof, self._n_dof)
        N = nc1 + nd + nc2
        if folded_fcs.shape[:2] != (N, N):
            raise RuntimeError("force constants shape does not match structure")

        # first contact
        b11 = nc1
        self._H1 = unfold_matrix(folded_fcs[0:b11, 0:b11])

        # first connection matrix (tau1)
        a1d = nc1
        b1d = nc1 + nd
        d1d = nc1
        self._tau1 = unfold_matrix(folded_fcs[a1d:b1d, 0:d1d])
        self._tau1s = slice_top_right(self._tau1, ns1, ns1)
        self._tau1s_H = self._tau1s.conj().T

        # device region
        add = nc1
        bdd = nc1 + nd
        self._Hd = unfold_matrix(folded_fcs[add:bdd, add:bdd])

        # second connection matrix (tau2)
        ad2 = nc1
        bd2 = nc1 + nd
        cd2 = nc1 + nd
        dd2 = N
        self._tau2 = unfold_matrix(folded_fcs[ad2:bd2, cd2:dd2])
        self._tau2s = slice_bottom_left(self._tau2, ns2, ns2)
        self._tau2s_H = self._tau2s.conj().T

        # second contact
        a22 = nc1 + nd
        b22 = N
        self._H2 = unfold_matrix(folded_fcs[a22:b22, a22:b22])

        # inside first contact
        self._H_LC = unfold_matrix(folded_fcs[nLCB:nc1, nLCB:nc1])
        self._H_LCB = unfold_matrix(folded_fcs[0:nLCB, 0:nLCB])
        self._tau_LC_LCB = unfold_matrix(folded_fcs[nLCB:nc1, 0:nLCB])

        # inside second contact
        self._H_RC = unfold_matrix(folded_fcs[a22:-nRCB, a22:-nRCB])
        self._tau_RC_RCB = unfold_matrix(folded_fcs[a22:-nRCB, -nRCB:])
        self._H_RCB = unfold_matrix(folded_fcs[-nRCB:, -nRCB:])

        # to avoid counting again
        self._ns1 = ns1
        self._ns2 = ns2

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
        fcs = fold_matrix(fcs, self._n_dof, self._n_dof)

        # rearrange rows
        rearrange = np.asarray(rearrange)
        fcs = fcs[rearrange]

        # rearrange columns
        for j in range(fcs.shape[0]):
            fcs[j, :] = fcs[j, :][rearrange]

        # unfold into original shape
        fcs = unfold_matrix(fcs)

        return fcs, index_map
