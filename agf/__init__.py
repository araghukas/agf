from typing import Dict, Union, Sequence, Callable
from numpy import ndarray
from os.path import expanduser
import warnings

from agf.structure import Section
from agf.agf import AGF
from agf.hm import HarmonicMatrix

__version__ = "2021.10.1"


def disable_log():
    AGF.PRINT_LOG = False


def enable_log():
    AGF.PRINT_LOG = True


def get_solver(atom_positions_file: str,
               layer_map_file: str,
               harmonic_constants_file: str,
               layer_assignments: Dict[Union[int, Section], Sequence[int]],
               sort_atoms: bool = True,
               atoms_sort_coordinate: int = 2,
               sort_harmonic_constants: bool = True,
               atom_dof: int = 3,
               log_progress: bool = True) -> AGF:
    """
    Create an Atomistic Green's Function solver from data files and layer assignments.

    :param atom_positions_file: path to file containing atom positions
    :param layer_map_file: path to layer map file
    :param harmonic_constants_file: path to harmonic constants for every atom by ID
    :param layer_assignments: a each section to layer indices
    :param sort_atoms: sort the atoms by the atom sort coordinate within every layer
    :param atoms_sort_coordinate: first, second, or third cartesian coordinate
    :param sort_harmonic_constants: sort harmonic constants into same ID order as layers
    :param atom_dof: number of degrees of freedom per atom
    :param log_progress: print AGF progress messages
    :return: an AGF instance, ready to compute Green's functions.
    """
    from agf.structure import read_atoms
    from agf.hm import (sort_atoms_in_layers,
                        read_harmonic_constants,
                        sort_harmonic_constants_by_layer)

    if not log_progress:
        disable_log()
    AGF.print("reading atoms and layer map")
    layers = read_atoms(atom_positions_file, layer_map_file)
    if sort_atoms:
        AGF.print("sorting atoms by coordinate")
        layers = sort_atoms_in_layers(layers, atoms_sort_coordinate)

    AGF.print("reading harmonic constants")
    hcs = read_harmonic_constants(harmonic_constants_file, atom_dof)
    if sort_harmonic_constants:
        AGF.print("sorting harmonic constants by layer")
        hcs = sort_harmonic_constants_by_layer(hcs, layers)

    AGF.print("creating harmonic matrix")
    matrix = HarmonicMatrix(hcs, layers)
    AGF.print("initializing model")
    model = AGF(matrix, layer_assignments)
    return model


def compute_transmission(omegas: Sequence[float],
                         atom_positions_file: str,
                         layer_map_file: str,
                         harmonic_constants_file: str,
                         layer_assignments: Dict[Union[int, Section], Sequence[int]],
                         sort_atoms: bool = True,
                         atoms_sort_coordinate: Union[int, int] = 2,
                         sort_harmonic_constants: bool = True,
                         atom_dof: int = 3,
                         log_progress: bool = True,
                         delta_func: Callable[[float], float] = None,
                         delta_func_kwargs: dict = None,
                         omega_start_index: int = 0,
                         results_savename: str = None) -> ndarray:
    """
    Compute the transmission over several angular frequencies and return the result.

    :param omegas: frequencies at which to compute
    :param atom_positions_file: path to file containing atom positions
    :param layer_map_file: path to layer map file
    :param harmonic_constants_file: path to harmonic constants for every atom by ID
    :param layer_assignments: a each section to layer indices
    :param sort_atoms: sort the atoms by the atom sort coordinate within every layer
    :param atoms_sort_coordinate: first, second, or third cartesian coordinate
    :param sort_harmonic_constants: sort harmonic constants into same ID order as layers
    :param atom_dof: number of degrees of freedom per atom
    :param log_progress: print AGF progress messages
    :param delta_func: function that returns broadening at each frequency
    :param delta_func_kwargs: dictionary of keyword arguments passed to `delta_func`
    :param omega_start_index: starting index of the frequencies in the output
    :param results_savename: name of the results file

    :return: array of transmission values
     """
    from numpy import zeros, asarray

    AGF.print("getting solver")
    model = get_solver(atom_positions_file,
                       layer_map_file,
                       harmonic_constants_file,
                       layer_assignments,
                       sort_atoms,
                       atoms_sort_coordinate,
                       sort_harmonic_constants,
                       atom_dof,
                       log_progress)

    if delta_func is None:
        from agf.utility import get_zhang_delta
        delta_func = get_zhang_delta
        delta_func_kwargs = {}
    elif delta_func_kwargs is None:
        delta_func_kwargs = {}

    if not results_savename:
        results_savename = "transmission_.csv"
        warnings.warn(
            f"results will be saved under default name '{results_savename}'."
        )

    n_omegas = len(omegas)
    omegas = asarray(omegas)
    deltas = delta_func(omegas, **delta_func_kwargs)
    trans_arr = zeros(n_omegas)

    output_filename = expanduser(results_savename)
    i_print = omega_start_index
    i_arr = 0
    AGF.print("\nindex, omega, delta, condition, transmission")
    with open(output_filename, 'w') as output_file:
        output_file.write(",omega,delta,condition,transmission\n")

    for omega, delta in zip(omegas, deltas):
        result = model.compute(omega, delta)
        trans = result.transmission
        cond = result.condition
        AGF.print("{:<4,d} {:<12,.6e} {:<12,.6e} {:<12,.6e} {:<12,.6e}"
                  .format(i_print, omega, delta, cond, trans))
        with open(output_filename, 'a') as output_file:
            output_file.write("{:d},{:f},{:f},{:f},{:f}\n"
                              .format(i_print, omega, delta, cond, trans))
        trans_arr[i_arr] = trans
        i_print += 1
        i_arr += 1

    return trans_arr
