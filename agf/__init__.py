from typing import Dict, Union, Sequence, Callable
from numpy import ndarray
import warnings

from agf.structure import Section
from agf.agf import AGF
from agf.hm import HarmonicMatrix


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


def compute_transmission(omegas: Sequence[float],
                         atom_positions_file: str,
                         layer_map_file: str,
                         force_constants_file: str,
                         layer_assignments: Dict[Union[int, Section], Sequence[int]],
                         sort_atoms: bool = True,
                         atoms_sort_coordinate: Union[int, int] = 2,
                         sort_force_constants: bool = True,
                         atom_dof: int = 3,
                         delta_func: Callable[[float], float] = None,
                         delta_func_kwargs: dict = None,
                         save_results: bool = True,
                         results_savename: str = None) -> ndarray:
    """
    Compute the transmission over several angular frequencies and return the result.

    :param omegas: frequencies at which to compute
    :param atom_positions_file: path to file containing atom positions
    :param layer_map_file: path to layer map file
    :param force_constants_file: path to force constants for every atom by ID
    :param layer_assignments: a each section to layer indices
    :param sort_atoms: sort the atoms by the atom sort coordinate within every layer
    :param atoms_sort_coordinate: first, second, or third cartesian coordinate
    :param sort_force_constants: sort force constants into same ID order as layers
    :param atom_dof: number of degrees of freedom per atom
    :param delta_func: function that returns broadening at each frequency
    :param delta_func_kwargs: dictionary of keyword arguments passed to `delta_func`
    :param save_results: save the results to a CSV file
    :param results_savename: name of the results file

    :return: array of transmission values
     """
    from numpy import zeros, asarray

    model = get_solver(atom_positions_file,
                       layer_map_file,
                       force_constants_file,
                       layer_assignments,
                       sort_atoms,
                       atoms_sort_coordinate,
                       sort_force_constants,
                       atom_dof)

    if delta_func is None:
        from agf.utility import get_zhang_delta
        delta_func = get_zhang_delta
        delta_func_kwargs = {}
    elif delta_func_kwargs is None:
        delta_func_kwargs = {}

    if not save_results:
        if results_savename:
            warnings.warn("results won't be saved, pass 'save_results=True' to save.")

    if not results_savename:
        if save_results:
            results_savename = "transmission_.csv"
            warnings.warn(
                f"results will be saved under default name '{results_savename}'."
            )

    n_omegas = len(omegas)

    omegas = asarray(omegas)
    deltas = delta_func(omegas, **delta_func_kwargs)
    trans = zeros(n_omegas)

    i_freq = 1
    for omega, delta in zip(omegas, deltas):
        print(f"frequency {i_freq}/{n_omegas}")
        result = model.compute(omega, delta)
        trans[i_freq - 1] = result.transmission
        i_freq += 1

    if save_results:
        from pandas import DataFrame
        df = DataFrame({
            "omega": omegas,
            "transmission": trans
        })
        df.to_csv(results_savename)

    return trans
