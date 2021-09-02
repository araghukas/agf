"""
Reproduce Zhang et al. 2007 Figures 4 and 5, with ALL curves in Fig 5.
"""
from math import sqrt
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt

import agf

# constants
K0 = 32.  # N / m
m1 = 4.6e-26  # kg
m2 = 9.2e-26  # kg
m3 = 2.3e-26  # kg
l0 = 5.5e-10  # m


def get_Zhang_fcs(mc: float, md: float) -> np.ndarray:
    """construct the total harmonic matrix: homogeneous"""
    fc = fcd = fd = K0

    f00 = 2 * fc / mc
    f10 = -fc / mc
    f11 = (fc + fcd) / mc
    f12 = -fcd / sqrt(mc * md)
    f22 = (fd + fcd) / md
    f23 = -fd / md
    f33 = 2. * fd / md

    return np.array([
        # -5,  -4,  -3,  -2,  -1,  +0,  +1
        [f00, f10, 0.0, 0.0, 0.0, 0.0, 0.0],  # -5
        [f10, f11, f12, 0.0, 0.0, 0.0, 0.0],  # -4
        [0.0, f12, f22, f23, 0.0, 0.0, 0.0],  # -3
        [0.0, 0.0, f23, f33, f23, 0.0, 0.0],  # -2
        [0.0, 0.0, 0.0, f23, f22, f12, 0.0],  # -1
        [0.0, 0.0, 0.0, 0.0, f12, f11, f10],  # +0
        [0.0, 0.0, 0.0, 0.0, 0.0, f10, f00],  # +1
    ])


def get_Zhang_structure() -> agf.structure.StructureSystem:
    """construct the linear chain structure"""
    layers = [agf.structure.Layer(i, [i + 6]) for i in range(-5, 2)]
    return agf.structure.StructureSystem(layers, 1, 4)


def plot_Fig_5(omegas: Iterable[float],
               trans1: Iterable[float],
               trans2: Iterable[float],
               trans3: Iterable[float]):
    fig, ax = plt.subplots()

    ax.set_xlim(0, 6e13)
    ax.set_ylim(0, 1)
    ax.set_xticks([i * 1e13 for i in [0, 2, 4, 6]])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(r"Angular frequency, $\omega$ (rad/s)", fontsize=18)
    ax.set_ylabel(r"Transmission, $\Xi$", fontsize=18)

    ax.plot(omegas, trans1, color='k', linestyle='-', label='Homogeneous case')
    ax.plot(omegas, trans2, color='k', linestyle='--', label='Heterogeneous case 1')
    ax.plot(omegas, trans3, color='k', linestyle=':', label='Heterogeneous case 2')

    plt.show()


def plot_Fig_6(omegas: Iterable[float],
               dos_homo: Iterable[float],
               dos_LD: Iterable[float],
               dos_D: Iterable[float]):
    fig, ax = plt.subplots(tight_layout=True)

    ax.set_xlim(0, 3e27)
    ax.set_ylim(0, 3e-4)
    ax.set_xticks([0, 1e27, 2e27, 3e27])
    ax.set_yticks([i * 1e-4 for i in range(0, 3)])
    ax.set_xlabel(r"$\omega^2$ (rad$^2$ sec$^{-2}$)", fontsize=18)
    ax.set_ylabel(r"Density of states (s m$^{-1}$)", fontsize=18)

    x = np.asarray(omegas)**2
    ax.plot(x / 2, dos_homo, linestyle='-', color='k', label='Homogeneous chain')
    ax.plot(x, dos_LD, linestyle='--', color='k', label='Atom LD')
    ax.plot(x, dos_D, linestyle=':', color='k', label='Atom D')
    plt.show()


def main(n_omegas: int = 300):
    struct = get_Zhang_structure()
    omegas = np.linspace(1e10, 6e13, n_omegas)
    deltas = agf.get_zhang_delta(omegas)

    # homogeneous case
    model1 = agf.AGF(struct, get_Zhang_fcs(m1, m1),
                     sort_force_constants=False, n_dof=1)
    trans1 = []
    dos_homo = []
    for omega, delta in zip(omegas, deltas):
        r = model1.compute(omega, delta)
        trans1.append(r.transmission)
        dos_homo.append(r.dos[1][1] / l0)

    # heterogeneous case 1
    model2 = agf.AGF(struct, get_Zhang_fcs(m1, m2),
                     sort_force_constants=False, n_dof=1)
    trans2 = []
    dos_LD = []
    # dos_LC = []  # can't do this one
    dos_D = []
    for omega, delta in zip(omegas, deltas):
        r = model2.compute(omega, delta)
        trans2.append(r.transmission)
        dos = r.dos
        dos_LD.append(dos[0][0] / l0)
        dos_D.append(dos[1][1] / l0)

    # heterogeneous case 2
    model3 = agf.AGF(struct, get_Zhang_fcs(m1, m3),
                     sort_force_constants=False, n_dof=1)
    trans3 = []
    for omega, delta in zip(omegas, deltas):
        r = model3.compute(omega, delta)
        trans3.append(r.transmission)

    plot_Fig_5(omegas, trans1, trans2, trans3)
    plot_Fig_6(omegas, dos_homo, dos_LD, dos_D)  # some scaling issues with this one


if __name__ == "__main__":
    main()
