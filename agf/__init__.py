from dataclasses import dataclass
from typing import Union, Sequence, Dict
import numpy as np
import warnings

from agf.hm import HarmonicMatrix
from agf.structure import Section
from agf.utility import unfold_matrix, get_zhang_delta
from agf.sancho_rubio import decimate


class AssignmentError(Exception):
    """For invalid layer assignments on AGF instances."""


class AGF:
    """
    An Atomistic Greens Function method calculator.

    Determines the Green's function of the 'device' portion, at a given frequency
    via the `compute(omega, delta)` method.
    """

    @dataclass(frozen=True)
    class _ComputeConstants:
        """constant values container for AGF instances"""
        H_D: np.ndarray  # unfolded D harmonic matrix
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
        I_LCBs: np.ndarray  # identity matrix for unfolded LCB
        I_RCBs: np.ndarray

        n_LCBs: int  # unfolded LCB surface matrix size
        n_RCBs: int

        n_LCs: int  # LC surface size
        n_RCs: int

    @property
    def harmonic_matrix(self) -> HarmonicMatrix:
        """
        Atom-wise matrix of d-dimensional harmonic interactions between pairs,
        a.k.a. the 'force constants' matrix.
        """
        return self._hm

    @harmonic_matrix.setter
    def harmonic_matrix(self, _hm: HarmonicMatrix):
        if not isinstance(_hm, HarmonicMatrix):
            raise TypeError(f"can not assign {type(_hm)} to harmonic matrix.")
        self._hm = _hm

    @property
    def layer_assignments(self) -> Dict[Section, Sequence[int]]:
        """dictionary matching one or more layers to a specific Section"""
        return self._lass

    @layer_assignments.setter
    def layer_assignments(self,
                          section_layers_map: Dict[Union[int, Section], Sequence[int]]):

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
        self._sections = {s: None for s in Section}
        self.harmonic_matrix = harmonic_matrix
        self.layer_assignments = layer_assignments

        self._validate_assignments()
        self._const = self._get_compute_constants()

    def _validate_assignments(self) -> None:
        """check matrix/section assignments for incompatible values"""

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

    def _get_compute_constants(self):
        """pre-compute constant values for efficiency"""
        n_dof = self._hm.force_constants.shape[-1]
        layers_LCB = [self._hm.layers[i] for i in self._lass[Section.LCB]]
        layers_RCB = [self._hm.layers[i] for i in self._lass[Section.RCB]]
        n_LCBs = n_dof * len(layers_LCB[-1])
        n_RCBs = n_dof * len(layers_RCB[0])
        H_D = self.get_matrix(Section.D)
        n_D = H_D.shape[0] * n_dof
        layers_LC = [self._hm.layers[i] for i in self._lass[Section.LC]]
        layers_RC = [self._hm.layers[i] for i in self._lass[Section.RC]]
        n_LCs = n_dof * len(layers_LC[-1])
        n_RCs = n_dof * len(layers_RC[0])

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

        return AGF._ComputeConstants(
            H_D=unfold_matrix(self.get_matrix(Section.D)),
            H_LCB=unfold_matrix(self.get_matrix(Section.LCB)),
            H_RCB=unfold_matrix(self.get_matrix(Section.RCB)),
            H_LC=unfold_matrix(self.get_matrix(Section.LC)),
            H_RC=unfold_matrix(self.get_matrix(Section.RC)),
            t_LC_LCBs=unfold_matrix(t_LC_LCBs),
            t_LCBs_LC=unfold_matrix(t_LCBs_LC),
            t_RC_RCBs=unfold_matrix(t_RC_RCBs),
            t_RCBs_RC=unfold_matrix(t_RCBs_RC),
            t_LCs_D=unfold_matrix(t_LCs_D),
            t_D_LCs=unfold_matrix(t_D_LCs),
            t_RCs_D=unfold_matrix(t_RCs_D),
            t_D_RCs=unfold_matrix(t_D_RCs),
            I_D=np.eye(n_D, dtype=np.complex128),
            I_LCBs=np.eye(n_LCBs, dtype=np.complex128),
            I_RCBs=np.eye(n_RCBs, dtype=np.complex128),
            n_LCBs=n_LCBs,
            n_RCBs=n_RCBs,
            n_LCs=n_LCs,
            n_RCs=n_RCs
        )

    def get_matrix(self, *key):
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
        w2I_LCBs = (omega**2 + 1.j * delta) * self._const.I_LCBs
        H_LC = self._const.H_LC
        t_LC_LCBs = self._const.t_LC_LCBs
        t_LCBs_LC = self._const.t_LCBs_LC
        gLs = np.linalg.inv(w2I_LCBs - H_LC - t_LCBs_LC @ g_LCBs @ t_LC_LCBs)

        g_RCBs = np.linalg.inv(dec_R.Ws)
        w2I_RCBs = (omega**2 + 1.j * delta) * self._const.I_RCBs
        H_RC = self._const.H_RC
        t_RC_RCBs = self._const.t_RC_RCBs
        t_RCBs_RC = self._const.t_RCBs_RC
        gRs = np.linalg.inv(w2I_RCBs - H_RC - t_RCBs_RC @ g_RCBs @ t_RC_RCBs)

        # calculate self energy matrices
        I_D = self._const.I_D

        n_LCs = self._const.n_LCs
        seL = I_D.copy()
        t_LCs_D = self._const.t_LCs_D
        t_D_LCs = self._const.t_D_LCs
        seL[-n_LCs:, -n_LCs:] = t_LCs_D @ gLs @ t_D_LCs

        n_RCs = self._const.n_RCs
        seR = I_D.copy()
        t_RCs_D = self._const.t_RCs_D
        t_D_RCs = self._const.t_D_RCs
        seR[:n_RCs, :n_RCs] = t_RCs_D @ gRs @ t_D_RCs

        # compute the 'device' Green's function
        w2I_D = omega**2 * I_D
        H_D = self._const.H_D
        G = np.linalg.inv(w2I_D - H_D - seL - seR)

        M1 = 1.j * (seL - seL.conj().T)
        M2 = 1.j * (seR - seR.conj().T)

        return GreensFunctionMatrix(omega, delta, G, M1, M2)


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


def get_solver(atom_positions_file: str,
               layer_map_file: str,
               force_constants_file: str,
               layer_assignments: Dict[Union[int, Section], Sequence[int]],
               sort_atoms: bool = True,
               atoms_sort_coordinate: Union[int, int] = 2,
               sort_force_constants: bool = True,
               atom_dof: int = 3) -> AGF:
    """
    Create an Atomistic Green's Function solver from data files and layer assignments.

    :param atom_positions_file: path to file containing atom positions
    :param layer_map_file: path to layer map file
    :param force_constants_file: path to force constants for every atom by ID
    :param layer_assignments: a each section to layer indices
    :param sort_atoms: sort the atoms by the atom sort coordinate within every layer
    :param atoms_sort_coordinate: first, second, or third cartesian coordinate
    :param sort_force_constants: sort force constants into same ID order as layers
    :param atom_dof: number of degrees of freedom per atom
    :return: an AGF instance, ready to compute Green's functions.
    """
    from agf.structure import read_atoms
    from agf.hm import (sort_atoms_in_layers,
                        read_force_constants,
                        sort_force_constants_by_layer)

    layers = read_atoms(atom_positions_file, layer_map_file)
    if sort_atoms:
        layers = sort_atoms_in_layers(layers, atoms_sort_coordinate)

    fcs = read_force_constants(force_constants_file, atom_dof)
    if sort_force_constants:
        fcs = sort_force_constants_by_layer(fcs, layers)

    matrix = HarmonicMatrix(fcs, layers)
    return AGF(matrix, layer_assignments)


def test():
    atom_positions_file = "~/Desktop/agf_tests/RELAXED_simplest_device.data"
    layer_map_file = "~/Desktop/agf_tests/zstack_test.map"
    force_constants_file = "~/Desktop/agf_tests/FC_simplest_device"
    layer_assignments = {
        Section.LCB: [0, 1],
        Section.LC: 2,
        Section.D: 3,
        Section.RC: 4,
        Section.RCB: [5, 6]
    }

    agf = get_solver(atom_positions_file,
                     layer_map_file,
                     force_constants_file,
                     layer_assignments)

    omegas = np.linspace(10, 10e13)
    deltas = get_zhang_delta(omegas)
    trans = []
    i = 1
    n_omega = len(omegas)
    for omega, delta in zip(omegas, deltas):
        print(f"frequency {i}/{n_omega}")
        result = agf.compute(omega, delta)
        trans.append(result.transmission)
        i += 1

    import matplotlib.pyplot as plt
    plt.plot(omegas, trans)
    plt.show()


if __name__ == "__main__":
    test()
