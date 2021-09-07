import numpy as np
from agf.structure import Section
from agf import compute_transmission


def main():
    from my_phonolammps.phonon_runner import PhononRunner

    # wire_datafile = "~/Desktop/agf_tests/zstack_test.data"
    # potential_file = "~/Desktop/agf_tests/GaAs.tersoff"
    # ph = PhononRunner(wire_datafile,
    #                   potential_file,
    #                   output_dir="~/Desktop/agf_tests",
    #                   generic_in="~/Desktop/agf_tests/generic_in.lammps",
    #                   generic_relax="~/Desktop/agf_tests/generic_relax.lammps",
    #                   marker="simple_nanowire")
    # ph.relax_structure()
    # ph.compute_forces()

    atom_relaxed_positions_file = "~/Desktop/agf_tests/RELAXED_simple_nanowire.data"
    layer_map_file = "~/Desktop/agf_tests/zstack_test.map"
    force_constants_file = "~/Desktop/agf_tests/FC_simple_nanowire"
    layer_assignments = {
        Section.LCB: [0, 1],
        Section.LC: 2,
        Section.D: 3,
        Section.RC: 4,
        Section.RCB: [5, 6]
    }

    omegas = np.linspace(1e-3, 6, 100)
    trans = compute_transmission(omegas,
                                 atom_relaxed_positions_file,
                                 layer_map_file,
                                 force_constants_file,
                                 layer_assignments)

    import matplotlib.pyplot as plt
    plt.plot(omegas, trans)
    plt.show()


if __name__ == "__main__":
    main()
