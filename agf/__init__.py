from typing import Dict, Union, Sequence, Callable
from numpy import ndarray
import warnings

from agf.structure import Section
from agf.agf import AGF
from agf.hm import HarmonicMatrix

__version__ = "7Sept2021"


def disable_log():
    AGF.PRINT_LOG = False


def enable_log():
    AGF.PRINT_LOG = True


def get_solver(atom_positions_file: str,
               layer_map_file: str,
               force_constants_file: str,
               layer_assignments: Dict[Union[int, Section], Sequence[int]],
               sort_atoms: bool = True,
               atoms_sort_coordinate: Union[int, int] = 2,
               sort_force_constants: bool = True,
               atom_dof: int = 3,
               log_progress: bool = True) -> AGF:
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
    :param log_progress: print AGF progress messages
    :return: an AGF instance, ready to compute Green's functions.
    """
    from agf.structure import read_atoms
    from agf.hm import (sort_atoms_in_layers,
                        read_force_constants,
                        sort_force_constants_by_layer)

    if not log_progress:
        disable_log()
    layers = read_atoms(atom_positions_file, layer_map_file)
    if sort_atoms:
        AGF.print("sorting atoms by coordinate")
        layers = sort_atoms_in_layers(layers, atoms_sort_coordinate)

    fcs = read_force_constants(force_constants_file, atom_dof)
    if sort_force_constants:
        AGF.print("sorting force constants by layer")
        fcs = sort_force_constants_by_layer(fcs, layers)

    matrix = HarmonicMatrix(fcs, layers)
    model = AGF(matrix, layer_assignments)
    return model


def compute_transmission(omegas: Sequence[float],
                         atom_positions_file: str,
                         layer_map_file: str,
                         force_constants_file: str,
                         layer_assignments: Dict[Union[int, Section], Sequence[int]],
                         sort_atoms: bool = True,
                         atoms_sort_coordinate: Union[int, int] = 2,
                         sort_force_constants: bool = True,
                         atom_dof: int = 3,
                         log_progress: bool = True,
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
    :param log_progress: print AGF progress messages
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
                       atom_dof,
                       log_progress)

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
    AGF.print("\nindex, omega, transmission")
    for omega, delta in zip(omegas, deltas):
        result = model.compute(omega, delta)
        t = result.transmission
        AGF.print("{:<4,d} {:<12,.6e} {:<12,.6e}".format(i_freq-1, omega, t))
        trans[i_freq - 1] = t
        i_freq += 1

    if save_results:
        try:
            from pandas import DataFrame
            df = DataFrame({
                "omega": omegas,
                "transmission": trans
            })
            df.to_csv(results_savename)
        except ModuleNotFoundError:
            with open(results_savename, 'w') as f:
                f.write(",omega,transmission\n")
                i = 0
                for omega, t in zip(omegas, trans):
                    f.write("{:d},{:f},{:f}\n".format(i, omega, t))
                    i += 1
                f.write("\n")

    return trans
