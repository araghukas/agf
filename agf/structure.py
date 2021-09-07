from enum import IntEnum
from dataclasses import dataclass
from typing import List
from os.path import expanduser
import numpy as np


class Section(IntEnum):
    """an enumeration of simulation sections as per Zhang-Mingo"""
    LCB = 0  # left contact bulk
    LC = 1  # left contact surface
    D = 2  # device
    RC = 3  # right contact surface
    RCB = 4  # right contact bulk


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
