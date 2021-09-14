"""Creation and manipulation of the total harmonic matrix."""
from typing import List, Union, Sequence
from os.path import expanduser
import numpy as np

from agf.structure import Layer
from agf.utility import extract_matrix


def read_harmonic_constants(harmonic_constants_filename: str, d: int) -> np.ndarray:
    """
    Read harmonic constants from text file into a numpy array.
    The returned array has shape (n,n,d,d), i.e. each matrix element is a dxd matrix.
    Moreover, the returned array is indexed in ascending order of atom ID.

    :param harmonic_constants_filename: path to file containing harmonic constants
    :param d: number of atomic degrees of freedom
    """
    harmonic_constants_filename = expanduser(harmonic_constants_filename)
    with open(harmonic_constants_filename) as f:
        line = f.readline()

        # outermost matrix size
        n, m = [int(x) for x in line.split()]

        # actual matrix with explicit size
        harmonic_constants = np.zeros((n, m, d, d), dtype=np.double)

        # extract interactions
        line = f.readline()
        atom_i, atom_j = [int(x) - 1 for x in line.split()]
        for i in range(d):
            line = f.readline()
            harmonic_constants[atom_i][atom_j][i] = [float(x) for x in line.split()]
        line = f.readline()

        while line:
            atom_i, atom_j = [int(x) - 1 for x in line.split()]
            for i in range(d):
                line = f.readline()
                harmonic_constants[atom_i][atom_j][i] = [float(x) for x in line.split()]
            line = f.readline()

    return harmonic_constants


def sort_atoms_in_layers(layers: List[Layer],
                         coordinate: int) -> List[Layer]:
    """sort atoms in layers by a position coordinate of constituent atoms"""
    new_layers = []
    for layer in layers:
        atoms = layer.atoms
        atoms.sort(key=lambda atom: atom.position[coordinate])
        new_layers.append(Layer(layer.number, atoms))

    return new_layers


def sort_harmonic_constants_by_layer(harmonic_constants: np.ndarray,
                                     layers: List[Layer]) -> np.ndarray:
    """
    Sort the harmonic constants into the same atomic order as the layers,
    assuming harmonic_constants are sorted by atom ID ranging 1 to N, i.e.

    :param harmonic_constants: folded harmonic constants, shape (N,N,d,d)
    :param layers: a list of layers for sorting harmonic constants
    :return: sorted folded harmonic constants
    """
    structure_ID_sequence = []
    for layer in layers:
        structure_ID_sequence += layer.IDs

    # where row/col indices *should* be, according to layers
    new_indices = np.asarray([ID - 1 for ID in structure_ID_sequence], dtype=int)

    if len(new_indices) != len(harmonic_constants):
        raise ValueError("length mismatch between layers and harmonic constants.")

    # rearrange rows
    fcs = harmonic_constants[new_indices]

    # rearrange cols
    for j in range(fcs.shape[0]):
        fcs[j, :] = fcs[j, :][new_indices]

    return fcs


class HarmonicMatrix:
    """a layer-wise representation of an (N,N,d,d) harmonic constants array"""

    @property
    def harmonic_constants(self) -> np.ndarray:
        """(N,N,d,d) array of interactions between all pairs of N atoms"""
        return self._harmonic_constants

    @harmonic_constants.setter
    def harmonic_constants(self, arr: np.ndarray):
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("interactions array must be square.")
        if arr.ndim != 4 or arr.shape[2] != arr.shape[3]:
            raise ValueError(f"interaction array shape {arr.shape} is not (N,N,d,d)")
        self._harmonic_constants = arr

    @property
    def layers(self) -> List[Layer]:
        """list of layers comprising the simulation geometry"""
        return self._layers

    @layers.setter
    def layers(self, _layers: List[Layer]):
        N_atoms = sum(len(layer) for layer in _layers)
        if N_atoms != len(self._harmonic_constants):
            raise ValueError("layer lengths incompatible with harmonic constants array.")

        self._layers = _layers
        self._construct_layer_index()

    def _construct_layer_index(self):
        """make a list of layer start and end indices within the harmonic constants matrix"""
        self._layer_locations = {}
        offset = 0
        for layer in self._layers:
            n_atoms = len(layer)
            self._layer_locations[layer.number] = (offset, n_atoms + offset)
            offset += n_atoms

    def __init__(self,
                 harmonic_constants: np.ndarray,
                 layers: List[Layer]):
        """
        Initialize a harmonic matrix from an array of harmonic constants
        and a list of layers.

        It is assumed that harmonic constants are in the same atomic order as the layers.

        :param harmonic_constants: (N,N,d,d) array of harmonic constants between atoms
        :param layers: a list of Layers containing N atoms
        """
        self.harmonic_constants = harmonic_constants.astype(np.complex128)
        self.layers = layers

    def index_layers(self, layer_indices: Union[int, Sequence[int]]) -> np.ndarray:
        """get harmonic_constants indices of atoms in the layer"""
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
        Get the matrix of harmonic constants between all atoms the first
        list of layers and all atoms in the second list of layers.
        """
        row_index = self.index_layers(layer_indices_1)
        col_index = self.index_layers(layer_indices_2)
        return extract_matrix(row_index, col_index, self._harmonic_constants)
