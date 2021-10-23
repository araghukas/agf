"""A module for calculating Atomistic Green's Function using the `AGF` solver."""
from dataclasses import dataclass
from typing import Union, Sequence, Dict
import numpy as np
import warnings

from agf.hm import HarmonicMatrix
from agf.structure import Section
from agf.utility import unfold_matrix
from agf.sancho_rubio import decimate


class AssignmentError(Exception):
    """For invalid layer assignments on AGF instances."""


class AGF:
    """
    An Atomistic Greens Function method calculator.

    Determines the Green's function of the 'device' portion, at a given frequency
    via the `compute(omega, delta)` method.
    """

    PRINT_LOG = True

    @dataclass(frozen=True)
    class _ComputeConstants:
        """constant values container for AGF instances"""
        H_D: np.ndarray  # unfolded (i.e. strictly 2D) D harmonic matrix
        H_LCB: np.ndarray  # unfolded LCB harmonic matrix
        H_RCB: np.ndarray
        H_LC: np.ndarray  # unfolded LC harmonic matrix
        H_RC: np.ndarray

        t_LC_LCBs: np.ndarray  # connection matrix between LC and LCB surface
        t_LCBs_LC: np.ndarray
        t_RC_RCBs: np.ndarray
        t_RCBs_RC: np.ndarray

        t_LCs_D: np.ndarray  # connection matrix between LC surface and D surface
        t_D_LCs: np.ndarray
        t_RCs_D: np.ndarray
        t_D_RCs: np.ndarray

        I_D: np.ndarray  # identity matrix for unfolded D
        I_LCs: np.ndarray  # identity matrix for unfolded LCB
        I_RCs: np.ndarray

        n_LCBs: int  # unfolded LCB surface matrix size
        n_RCBs: int

        n_DsL: int  # size of device left surface
        n_DsR: int  # size of device right surface

    @property
    def layer_assignments(self) -> Dict[Section, Sequence[int]]:
        """dictionary matching one or more layers to a specific Section"""
        return self._lass

    @layer_assignments.setter
    def layer_assignments(self,
                          section_layers_map: Dict[Union[int, Section], Sequence[int]]):
        self.print("assigning layers")

        # convert all keys to Sections
        section_layers_map = {Section(k): v for k, v in section_layers_map.items()}

        # check all Sections specified
        for k in Section:
            if k not in section_layers_map:
                raise AssignmentError(f"missing assignment for section {k.name} ({k}).")

        # convert all scalar values into length-1 sequences
        self._lass = section_layers_map
        for k, v in self._lass.items():
            if type(v) is int:
                self._lass[k] = [v]

    def __init__(self,
                 harmonic_matrix: HarmonicMatrix,
                 layer_assignments: Dict[Union[int, Section], Sequence[int]]):
        self._hm = harmonic_matrix
        self.layer_assignments = layer_assignments

        self._validate_assignments()
        self._const = self._get_compute_constants()

    def _validate_assignments(self) -> None:
        """check matrix/section assignments for incompatible values"""
        self.print("validating sections and matrices")
        # check that the LCB and RCB contain more than one layer
        # check that the layers lists are non-empty
        layers_L = [self._hm.layers[i] for i in self._lass[Section.LCB]]
        if len(layers_L) < 2:
            warnings.warn("section LCB has only one layer.")
        elif np.any(np.diff([len(layer) for layer in layers_L]) != 0):
            raise ValueError("unequal layers lengths in the LCB.")

        layers_R = [self._hm.layers[i] for i in self._lass[Section.RCB]]
        if len(layers_R) < 2:
            warnings.warn("section RCB has only one layer.")
        elif np.any(np.diff([len(layer) for layer in layers_R]) != 0):
            raise ValueError("unequal layers lengths in RCB.")

        # check assumptions about Sections hold
        H_LCB_D = self._get_matrix(Section.LCB, Section.D)
        if not np.linalg.norm(H_LCB_D) == 0.0:
            warnings.warn("LCB-D interaction is non-zero.")
        H_D_LCB = self._get_matrix(Section.D, Section.LCB)
        if not np.linalg.norm(H_D_LCB) == 0.0:
            warnings.warn("D-LCB interaction is non-zero.")
        H_RCB_D = self._get_matrix(Section.RCB, Section.D)
        if not np.linalg.norm(H_RCB_D) == 0.0:
            warnings.warn("RCB-D interaction is non-zero.")
        H_D_RCB = self._get_matrix(Section.D, Section.RCB)
        if not np.linalg.norm(H_D_RCB) == 0.0:
            warnings.warn("D-RCB interaction is non-zero.")

    def _get_compute_constants(self) -> _ComputeConstants:
        """pre-compute constant values for efficiency"""
        self.print("pre-computing constant values")
        n_dof = self._hm.harmonic_constants.shape[-1]
        layers_LCB = [self._hm.layers[i] for i in self._lass[Section.LCB]]
        layers_RCB = [self._hm.layers[i] for i in self._lass[Section.RCB]]
        n_LCBs = n_dof * len(layers_LCB[-1])
        n_RCBs = n_dof * len(layers_RCB[0])

        layers_LC = [self._hm.layers[i] for i in self._lass[Section.LC]]
        layers_RC = [self._hm.layers[i] for i in self._lass[Section.RC]]
        layers_D = [self._hm.layers[i] for i in self._lass[Section.D]]
        n_LCs = n_dof * len(layers_LC[-1])
        n_RCs = n_dof * len(layers_RC[0])

        H_D = self._get_matrix(Section.D)
        n_D = H_D.shape[0] * n_dof
        n_DsL = n_dof * len(layers_D[0])
        n_DsR = n_dof * len(layers_D[-1])

        t_LCBs_LC = self._hm.get_interaction(self._lass[Section.LCB][-1],
                                             self._lass[Section.LC][0])

        t_LC_LCBs = self._hm.get_interaction(self._lass[Section.LC][0],
                                             self._lass[Section.LCB][-1])

        t_RCBs_RC = self._hm.get_interaction(self._lass[Section.RCB][0],
                                             self._lass[Section.RC][-1])

        t_RC_RCBs = self._hm.get_interaction(self._lass[Section.RC][-1],
                                             self._lass[Section.RCB][0])

        t_LCs_D = self._hm.get_interaction(self._lass[Section.LC][-1],
                                           self._lass[Section.D][0])

        t_D_LCs = self._hm.get_interaction(self._lass[Section.D][0],
                                           self._lass[Section.LC][-1])

        t_RCs_D = self._hm.get_interaction(self._lass[Section.RC][0],
                                           self._lass[Section.D][-1])

        t_D_RCs = self._hm.get_interaction(self._lass[Section.D][-1],
                                           self._lass[Section.RC][0])

        consts = AGF._ComputeConstants(
            H_D=unfold_matrix(self._get_matrix(Section.D)),
            H_LCB=unfold_matrix(self._get_matrix(Section.LCB)),
            H_RCB=unfold_matrix(self._get_matrix(Section.RCB)),
            H_LC=unfold_matrix(self._get_matrix(Section.LC)),
            H_RC=unfold_matrix(self._get_matrix(Section.RC)),
            t_LC_LCBs=unfold_matrix(t_LC_LCBs),
            t_LCBs_LC=unfold_matrix(t_LCBs_LC),
            t_RC_RCBs=unfold_matrix(t_RC_RCBs),
            t_RCBs_RC=unfold_matrix(t_RCBs_RC),
            t_LCs_D=unfold_matrix(t_LCs_D),
            t_D_LCs=unfold_matrix(t_D_LCs),
            t_RCs_D=unfold_matrix(t_RCs_D),
            t_D_RCs=unfold_matrix(t_D_RCs),
            I_D=np.eye(n_D, dtype=np.complex128),
            I_LCs=np.eye(n_LCs, dtype=np.complex128),
            I_RCs=np.eye(n_RCs, dtype=np.complex128),
            n_LCBs=n_LCBs,
            n_RCBs=n_RCBs,
            n_DsL=n_DsL,
            n_DsR=n_DsR
        )
        del self._hm  # to save memory, no longer needed
        return consts

    def _get_matrix(self, *key):
        """get the interaction matrix between any two Sections"""
        n_keys = len(key)
        if n_keys == 0 or n_keys > 2:
            raise ValueError("invalid matrix key(s).")
        if n_keys == 1:
            key += key

        layer_indices_1 = self._lass[key[0]]
        layer_indices_2 = self._lass[key[1]]
        return self._hm.get_interaction(layer_indices_1, layer_indices_2)

    def compute(self,
                omega: float,
                delta: float,
                decimation_tol: float = 1e-6) -> 'GreensFunctionMatrix':

        # run decimation on homogeneous contacts
        dec_L = decimate(self._const.H_LCB,
                         omega,
                         delta,
                         self._const.n_LCBs,
                         decimation_tol)
        dec_R = decimate(self._const.H_RCB,
                         omega,
                         delta,
                         self._const.n_RCBs,
                         decimation_tol,
                         flip=True)

        # calculate uncoupled Green's functions
        g_LCBs = np.linalg.inv(dec_L.Ws)
        w2I_LCs = (omega**2 + 1.j * delta) * self._const.I_LCs
        H_LC = self._const.H_LC
        t_LC_LCBs = self._const.t_LC_LCBs
        t_LCBs_LC = self._const.t_LCBs_LC
        gLs = np.linalg.inv(w2I_LCs - H_LC - t_LC_LCBs @ g_LCBs @ t_LCBs_LC)

        g_RCBs = np.linalg.inv(dec_R.Ws)
        w2I_RCs = (omega**2 + 1.j * delta) * self._const.I_RCs
        H_RC = self._const.H_RC
        t_RC_RCBs = self._const.t_RC_RCBs
        t_RCBs_RC = self._const.t_RCBs_RC
        gRs = np.linalg.inv(w2I_RCs - H_RC - t_RC_RCBs @ g_RCBs @ t_RCBs_RC)

        # calculate self energy matrices
        I_D = self._const.I_D

        n_DsL = self._const.n_DsL
        seL = I_D.copy()
        t_LCs_D = self._const.t_LCs_D
        t_D_LCs = self._const.t_D_LCs
        seL[-n_DsL:, -n_DsL:] = t_D_LCs @ gLs @ t_LCs_D

        n_DsR = self._const.n_DsR
        seR = I_D.copy()
        t_RCs_D = self._const.t_RCs_D
        t_D_RCs = self._const.t_D_RCs
        seR[:n_DsR, :n_DsR] = t_D_RCs @ gRs @ t_RCs_D

        # compute the 'device' Green's function
        w2I_D = omega**2 * I_D
        H_D = self._const.H_D
        G = np.linalg.solve(w2I_D - H_D - seL - seR, self._const.I_D)

        M1 = 1.j * (seL - seL.conj().T)
        M2 = 1.j * (seR - seR.conj().T)

        return GreensFunctionMatrix(omega, delta, G, M1, M2)

    @staticmethod
    def print(s: str) -> None:
        """optional logging of progress"""
        if AGF.PRINT_LOG:
            print(s)


@dataclass(frozen=True)
class GreensFunctionMatrix:
    """Results of an AGF computation at some frequency `omega`"""
    omega: float  # frequency
    delta: float  # broadening
    G: np.ndarray  # green's function matrix of the device
    M1: np.ndarray  # Zhang eq. (16)
    M2: np.ndarray

    @property
    def transmission(self) -> float:
        """transmission function at omega"""
        trans = np.trace(self.M1 @ self.G @ self.M2 @ self.G.conj().T)
        return float(trans.real)

    @property
    def dos(self) -> np.ndarray:
        return 1.j * self.omega / np.pi * (self.G - self.G.conj().T)
