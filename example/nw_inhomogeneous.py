"""
A simple example using a small nanowire structure with a
structurally distinct 'device' region.

More specifically, the contacts are ZB-GaAs and the device is WZ-GaAs.

NOTE: this example requires the additional packages
    - nwlattice (https://github.com/araghukas/nwlattice)
    - my_phonolammps (https://github.com/araghukas/my-phonolammps)

"""
import numpy as np
import matplotlib.pyplot as plt

from nwlattice import nw
from nwlattice.indices import Manual
from my_phonolammps.phonon_runner import PhononRunner
import agf
from agf.units import LAMMPS_metal_to_Hz
from agf.structure import Section

DATA_FILE = "nw_inhomogeneous.data"
MAP_FILE = "nw_inhomogeneous.map"


def write_nanowire(width: float = None, n_xy: int = None) -> None:
    """
    write the nanowire data and layer map files.

    :param width: nanowire width in Angstroms
    :param n_xy: approximate radius (in atoms) of the nanowire (alternate to width)
    """
    w1 = nw.ZBTwinA(scale=5.65315, width=width, n_xy=n_xy, nz=22,
                    indexer=Manual([9, 10, 11, 12]))

    w1.write_points(DATA_FILE)

    # the nanowire is a stack of planes; partition the planes into sequential layers
    layer_planes_map = [
        [0, 1, 2],  # an A-B-C plane of ZB-GaAs (same as all other groups of 3)
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11, 12],  # D-F-D-F planes of WZ-GaAs segments in the center
        [13, 14, 15],
        [16, 17, 18],
        [19, 20, 21]
    ]

    w1.write_layer_map(MAP_FILE, layer_planes_map=layer_planes_map)


def relax_and_compute_hc(relax: bool = True, dummy: bool = False) -> PhononRunner:
    """
    Compute the harmonic constants.

    :param relax: run energy minimization in LAMMPS before computing
    :param dummy: return just the PhononRunner instance without running anything
    """
    phr = PhononRunner(wire_datafile=DATA_FILE,
                       potential_file="./lmp_generics/GaAs.tersoff",
                       output_dir='./nw_inhomogeneous/',
                       generic_relax="./lmp_generics/generic_relax.lammps",
                       generic_in="./lmp_generics/generic_in.lammps",
                       marker="nw_ih")
    if dummy:
        return phr

    if relax:
        phr.relax_structure()
    phr.compute_harmonic_constants()

    return phr


def main():
    write_nanowire(n_xy=2)
    phr = relax_and_compute_hc(dummy=True)

    omegas = np.linspace(1e-2, 7e-1, 500)
    layer_assignments = {
        Section.LCB: [0, 1],
        Section.LC: 2,
        Section.D: 3,
        Section.RC: 4,
        Section.RCB: [5, 6]
    }

    agf.compute_transmission(omegas=omegas,
                             atom_positions_file=phr.outputs.relaxed_unitcell_filename,
                             layer_map_file=MAP_FILE,
                             harmonic_constants_file=phr.outputs.harmonic_constants_filename,
                             layer_assignments=layer_assignments,
                             results_savename="nw_inhomogeneous/trans.csv")


def plot_computed_transmission(omegas_scale: float):
    """read the computed CSV file and plot the transmission function"""
    # or do this with pandas.read_csv
    omegas = []
    trans = []
    with open("nw_inhomogeneous/trans.csv", 'r') as t:
        t.readline()
        line = t.readline()
        while line:
            nums = line.strip().split(',')
            omegas.append(omegas_scale * float(nums[1]))
            trans.append(float(nums[2]))
            line = t.readline()

    fig, ax = plt.subplots(tight_layout=True)
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel(r"Transmission Function, $\Xi$")

    ax.plot(omegas, trans)

    plt.show()


if __name__ == "__main__":
    # main()
    plot_computed_transmission(omegas_scale=LAMMPS_metal_to_Hz)
