import numpy as np
from agf import Section, get_solver, get_zhang_delta


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
