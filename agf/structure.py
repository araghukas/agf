import os
import numpy as np

from typing import List, Dict
from dataclasses import dataclass


@dataclass(frozen=True)
class Atom:
    """an atom of some type at some position in 3D space"""
    index: int
    typ: int
    position: np.ndarray

    def __post_init__(self):
        if self.position.ndim != 1 or len(self.position) != 3:
            raise ValueError("position {} is not a 3-vector"
                             .format(self.position))


@dataclass(frozen=True)
class Layer:
    """
    An ordered group of atoms organized by ID and type.
    These are stacked to produce structure systems for AGF.
    """
    number: int
    ids: List[int]
    types: List[int]

    @property
    def N(self) -> int:
        """number of atoms in the layer"""
        return len(self.ids)

    def __post_init__(self):
        if len(self.ids) != len(self.types):
            raise ValueError("lengths of ids and types lists are not equal")


class StructureSystem:
    """
    Generic contact-device-contact structure for AGF calculations.
    Each of the three sections is a list of Layers.
    """

    @property
    def layers(self) -> List[Layer]:
        """list of all layers"""
        return self._layers

    @property
    def contact1(self) -> List[Layer]:
        """list of layers belonging to first contact"""
        return [self._layers[layer_number] for layer_number in self._c1]

    @property
    def device(self) -> List[Layer]:
        """list of layers belonging to device portion"""
        return [self._layers[layer_number] for layer_number in self._dv]

    @property
    def contact2(self) -> List[Layer]:
        """list of layers belonging to second contact"""
        return [self._layers[layer_number] for layer_number in self._c2]

    @property
    def section_ids(self) -> List[List[List[int]]]:
        """list of ids in contact1, device, and contact2 regions"""
        return [
            [layer.ids for layer in self.contact1],
            [layer.ids for layer in self.device],
            [layer.ids for layer in self.contact2]
        ]

    def __init__(self, map_file_path: str, data_file_path: str,
                 b1: int, b2: int):
        """
        Initialize a CDC structure using a layer map file and two boundaries
        to divide the layers into a contact-device-contact structure.

        :param map_file_path: path to layer map file
        :param b1: largest index of plane in first contact (inclusive)
        :param b2: largest index of plane in device (inclusive)
        NOTE: these indices start with 0.
        """

        self._layers = read_layer_map(map_file_path)

        if not (0 < b1 < b2 < len(self.layers)):
            raise ValueError("invalid boundary layer indices b1 = %d, b2 = %d"
                             % (b1, b2))

        self._assign_layers(b1, b2)
        self._data_map = read_data_file(data_file_path)

    def _assign_layers(self, b1: int, b2: int) -> None:
        """
        Based on the boundaries, create lists of layers representing
        each of three regions.
        """
        # first contact: layers [0,b1]
        self._c1 = [i for i in range(b1 + 1)]

        # device: layers (b1,b2]
        self._dv = [i for i in range(b1 + 1, b2 + 1)]

        # second contact (b2,N]
        self._c2 = [i for i in range(b2 + 1, len(self.layers))]

    def locate_atom(self, atom_id: int) -> Atom:
        """returns Atom object summarizing corresponding line in atoms data file"""
        return self._data_map[atom_id]


def read_layer_map(file_path: str) -> List[Layer]:
    """
    Read a layer map file generated by the 'nwlattice' package
    """
    layers = []
    file_path = os.path.expanduser(file_path)
    with open(file_path) as f:
        # skip comment line
        f.readline()

        # prime loop with first layer
        line = f.readline()
        nums = [int(x) for x in line.split()[1:]]
        layer_number = nums[0]
        layer_types = []
        layer_ids = []

        line = f.readline()
        while line:
            if line.startswith("layer"):
                layers.append(Layer(layer_number, layer_ids, layer_types))
                nums = [int(x) for x in line.split()[1:]]
                layer_number = nums[0]
                layer_types = []
                layer_ids = []
            else:
                nums = [int(x) for x in line.split()]
                layer_types.append(nums[0])
                layer_ids.append(nums[1])
            line = f.readline()
        layers.append(Layer(layer_number, layer_ids, layer_types))

    return layers


def read_data_file(file_path: str) -> Dict[int, Atom]:
    """
    Read an xyz data file to get position for every atom by id.
    Returns a dictionary {atom_id: Atom(typ, position)}
    """
    ids_map = {}
    file_path = os.path.expanduser(file_path)
    with open(file_path) as f:
        # skip to atoms section
        line = f.readline()
        while line:
            line = line.strip()
            words = line.split()
            if len(words) < 5:
                line = f.readline()
                continue

            chars = ''.join(c for c in line if not c.isdigit()).strip('. ')
            if not chars:
                break
            else:
                line = f.readline()

        # proceed to read atoms <ID, typ, x, y, z>
        index = 0
        while line:
            str_nums = [s for s in line.split()]
            atom_id = int(str_nums[0])
            typ = int(str_nums[1])
            position = np.array([float(x) for x in str_nums[2:5]])

            ids_map[atom_id] = Atom(index, typ, position)
            line = f.readline()
            index += 1

    return ids_map
