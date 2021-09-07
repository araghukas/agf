from dataclasses import dataclass
from typing import List, Union, Sequence, Dict
from enum import IntEnum
from os.path import expanduser
import numpy as np
import warnings

from agf.utility import extract_matrix, unfold_matrix
from agf.sancho_rubio import decimate
from agf import get_zhang_delta


class Section(IntEnum):
    """an enumeration of simulation sections as per Zhang-Mingo"""
    LCB = 0  # left contact bulk
    LC = 1  # left contact surface
    D = 2  # device
    RC = 3  # right contact surface
    RCB = 4  # right contact bulk


class Coordinate(IntEnum):
    """an enumeration of Cartesian coordinates"""
    x = 0
    y = 1
    z = 2


@dataclass(frozen=True)
class Atom:
    """an atom of some type at some position in 3D space"""
    ID: int
    typ: int
    position: np.ndarray


@dataclass(frozen=True)
class Layer:
    """a collection of atoms"""
    number: int
    atoms: List[Atom]

    @property
    def IDs(self) -> List[int]:
        """ordered IDs of all constituent atoms"""
        return [atom.ID for atom in self.atoms]

    def __len__(self):
        """number of atoms"""
        return len(self.atoms)


def only_numerical_entries(line: str) -> bool:
    """return True if line contains only numbers and spaces"""
    if len(line) == 0:
        return False

    for c in line:
        if c.isspace() or c == '.':
            continue
        if not c.isdigit():
            return False
    return True


def read_atoms(atom_positions_file: str,
               layer_map_file: str) -> List[Layer]:
    """
    Construct a list of layers containing atoms with ID, type, and position.

    The `atom_positions` file should contain a line for each atoms of the form:

        ID TYPE x y z ...

    only the first 5 space-delimited entries are used. This function finds those lines
    by skipping ahead to lines with ONLY numerical entries and of length at least 5.

    The `layer_map_file` should be of the form:

        # comment line
        1 N1
            id11 type11
            id12 type12
            id13 type13
            ....
            id1(N1) type1(N1)
        2 N2
            id21 type21
            id22 type22
            ...
            id2(N2) type2(N2)
        ...
    """
    atom_positions_file = expanduser(atom_positions_file)
    positions_dict = {}  # {ID: [x, y, z]}
    with open(atom_positions_file) as f_atom:
        lines = [line.strip() for line in f_atom.readlines()]

        for line in lines:
            if not only_numerical_entries(line):
                # skip to atoms section
                continue
            entries = line.split()
            ID = int(entries[0])
            position = np.array([float(x) for x in entries[2:5]])
            positions_dict[ID] = position

    layers = []
    layer_map_file = expanduser(layer_map_file)
    with open(layer_map_file) as f:
        # skip comment line
        f.readline()

        # prime loop with first layer
        line = f.readline()
        nums = [int(x) for x in line.split()[1:]]
        layer_number = nums[0]
        atoms = []
        line = f.readline()

        atom_idx = 0
        while line:
            if line.startswith("layer"):
                layers.append(Layer(layer_number, atoms))
                nums = [int(x) for x in line.split()[1:]]
                layer_number = nums[0]
                atoms = []
            else:
                ID, typ = [int(x) for x in line.split()]
                atoms.append(Atom(ID, typ, positions_dict[ID]))
                atom_idx += 1
            line = f.readline()

        layers.append(Layer(layer_number, atoms))

    return layers


def read_force_constants(force_constants_filename: str, d: int) -> np.ndarray:
    """
    Read force constants from text file into a numpy array.
    The returned array has shape (n,n,d,d), i.e. each matrix element is a dxd matrix.
    Moreover, the returned array is indexed in ascending order of atom ID.
    """
    force_constants_filename = expanduser(force_constants_filename)
    with open(force_constants_filename) as f:
        line = f.readline()

        # outermost matrix size
        n, m = [int(x) for x in line.split()]

        # actual matrix with explicit size
        force_constants = np.zeros((n, m, d, d), dtype=np.double)

        # extract interactions
        line = f.readline()
        atom_i, atom_j = [int(x) - 1 for x in line.split()]
        for i in range(d):
            line = f.readline()
            force_constants[atom_i][atom_j][i] = [float(x) for x in line.split()]
        line = f.readline()

        while line:
            atom_i, atom_j = [int(x) - 1 for x in line.split()]
            for i in range(d):
                line = f.readline()
                force_constants[atom_i][atom_j][i] = [float(x) for x in line.split()]
            line = f.readline()

    return force_constants


def sort_atoms_in_layers(layers: List[Layer],
                         coordinate: Union[int, Coordinate]) -> List[Layer]:
    """sort atoms in layers by a position coordinate of constituent atoms"""
    new_layers = []
    for layer in layers:
        atoms = layer.atoms
        atoms.sort(key=lambda atom: atom.position[Coordinate(coordinate)])
        new_layers.append(Layer(layer.number, atoms))

    return new_layers


def sort_force_constants_by_layer(force_constants: np.ndarray,
                                  layers: List[Layer]) -> np.ndarray:
    """
    Sort the force constants into the same atomic order as the layers,
    assuming force_constants are sorted by atom ID ranging 1 to N, i.e.

    :param force_constants: folded force constants, shape (N,N,d,d)
    :param layers: a list of layers for sorting force constants
    :return: sorted folded force constants
    """
    structure_ID_sequence = []
    for layer in layers:
        structure_ID_sequence += layer.IDs

    # where row/col indices *should* be, according to layers
    new_indices = np.asarray([ID - 1 for ID in structure_ID_sequence], dtype=int)

    if len(new_indices) != len(force_constants):
        raise ValueError("length mismatch between layers and force constants.")

    # rearrange rows
    fcs = force_constants[new_indices]

    # rearrange cols
    for j in range(fcs.shape[0]):
        fcs[j, :] = fcs[j, :][new_indices]

    return fcs


class HarmonicMatrix:
    """a layer-wise representation of an (N,N,d,d) force constants array"""

    @property
    def force_constants(self) -> np.ndarray:
        """(N,N,d,d) array of interactions between all pairs of N atoms"""
        return self._force_constants

    @force_constants.setter
    def force_constants(self, arr: np.ndarray):
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("interactions array must be square.")
        if arr.ndim != 4 or arr.shape[2] != arr.shape[3]:
            raise ValueError(f"interaction array shape {arr.shape} is not (N,N,d,d)")
        self._force_constants = arr

    @property
    def layers(self) -> List[Layer]:
        """list of layers comprising the simulation geometry"""
        return self._layers

    @layers.setter
    def layers(self, _layers: List[Layer]):
        N_atoms = sum(len(layer) for layer in _layers)
        if N_atoms != len(self._force_constants):
            raise ValueError("layer lengths incompatible with force constants array.")

        self._layers = _layers
        self._construct_layer_index()

    def _construct_layer_index(self):
        """make a list of layer start and end indices within the force constants matrix"""
        self._layer_locations = {}
        offset = 0
        for layer in self._layers:
            n_atoms = len(layer)
            self._layer_locations[layer.number] = (offset, n_atoms + offset)
            offset += n_atoms

    def __init__(self,
                 force_constants: np.ndarray,
                 layers: List[Layer]):
        """
        Initialize a harmonic matrix from an array of force constants
        and a list of layers.

        It is assumed that force constants are in the same atomic order as the layers.

        :param force_constants: (N,N,d,d) array of force constants between atoms
        :param layers: a list of Layers containing N atoms
        """
        self.force_constants = force_constants.astype(np.complex128)
        self.layers = layers

    def index_layers(self, layer_indices: Union[int, Sequence[int]]) -> np.ndarray:
        """get force_constants indices of atoms in the layer"""
        if type(layer_indices) is int:
            a, b = self._layer_locations[layer_indices]
            index = np.arange(a, b, 1)
        else:
            lengths = [len(self._layers[i]) for i in layer_indices]
            index = np.empty(sum(lengths), dtype=int)
            n = 0
            for i, length in zip(layer_indices, lengths):
                a, b = self._layer_locations[i]
                index[n:n + length] = np.arange(a, b, 1)
                n += length

        return index

    def get_interaction(self,
                        layer_indices_1: Union[int, Sequence[int]],
                        layer_indices_2: Union[int, Sequence[int]]) -> np.ndarray:
        """
        Get the matrix of force constants between all atoms the first
        list of layers and all atoms in the second list of layers.
        """
        row_index = self.index_layers(layer_indices_1)
        col_index = self.index_layers(layer_indices_2)
        return extract_matrix(row_index, col_index, self._force_constants)


class AssignmentError(Exception):
    """For invalid layer assignments on AGF instances."""


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


class AGF:
    """
    An Atomistic Greens Function method calculator.

    Determines the Green's function of the 'device' portion, at a given frequency
    via the `compute(omega, delta)` method.
    """

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

        return _ComputeConstants(
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
               atoms_sort_coordinate: Union[int, Coordinate] = 2,
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
    layers = read_atoms(atom_positions_file, layer_map_file)
    if sort_atoms:
        layers = sort_atoms_in_layers(layers, atoms_sort_coordinate)

    fcs = read_force_constants(force_constants_file, atom_dof)
    if sort_force_constants:
        fcs = sort_force_constants_by_layer(fcs, layers)

    hm = HarmonicMatrix(fcs, layers)
    return AGF(hm, layer_assignments)


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
