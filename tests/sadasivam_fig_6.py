"""
Reproduce Sadasivam Figure 6.
"""
from scipy.constants import m_u
from math import sqrt
from typing import Iterable

import numpy as np

# constants
fc = 25.  # N/m
fd = 50.  # N/m
fcd = (fc + fd) / 2
mc = md = 20. * m_u  # kg
l0 = 5.5e-10  # m


def get_sadasivam_fcs() -> np.ndarray:
    """construct the total harmonic matrix"""
    f00 = 2 * fc / mc
    f10 = -fc / mc
    f11 = (fc + fcd) / mc
    f12 = -fcd / sqrt(mc * md)
    f22 = (fd + fcd) / md
    f23 = -fd / md
    f33 = 2. * fd / md

    return np.array([
        # -6,  -5,  -4,  -3,  -2,  -1,  +0,  +1,  +2
        [f00, f10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # -6
        [f10, f00, f10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # -5
        [0.0, f10, f11, f12, 0.0, 0.0, 0.0, 0.0, 0.0],  # -4
        [0.0, 0.0, f12, f22, f23, 0.0, 0.0, 0.0, 0.0],  # -3
        [0.0, 0.0, 0.0, f23, f33, f23, 0.0, 0.0, 0.0],  # -2
        [0.0, 0.0, 0.0, 0.0, f23, f22, f12, 0.0, 0.0],  # -1
        [0.0, 0.0, 0.0, 0.0, 0.0, f12, f11, f10, 0.0],  # +0
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, f10, f00, f10],  # +1
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, f10, f00]  # +2
    ])


def plot_Fig_6(omegas: Iterable[float],
               trans: Iterable[float],
               dos: Iterable[float]) -> None:
    """plot FIG. 6"""
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.set_xlim(0., 9e13)
    ax1.set_ylim(0., 1.)
    ax1.set_xticks([i * 1e13 for i in range(10)])
    ax1.set_yticks([0.0, 0.5, 1.0])
    ax1.set_xlabel(r"$\omega$(s$^{-1}$)")
    ax1.set_ylabel("Transmission function")
    ax1.yaxis.label.set_color('b')

    ax2.set_ylim(0, 4e-5)
    ax2.set_yticks([0., 2e-5, 4e-5])
    ax2.set_ylabel(r"LDOS at center of device (s m$^{-1}$)")
    ax2.yaxis.label.set_color('g')

    ax1.plot(omegas, trans, color='b')
    ax2.plot(omegas, dos, color='g')

    plt.show()


def main(n_omegas: int = 300):
    from agf import HarmonicMatrix, AGF, Section
    from agf.structure import Atom, Layer
    from agf.utility import fold_matrix

    fcs = get_sadasivam_fcs()
    fcs = fold_matrix(fcs, 1, 1)

    layers = [
        Layer(-6, [Atom(1, 1, np.array([0., 0., -6.]))]),
        Layer(-5, [Atom(1, 1, np.array([0., 0., -5.]))]),
        Layer(-4, [Atom(1, 1, np.array([0., 0., -4.]))]),
        Layer(-3, [Atom(1, 2, np.array([0., 0., -3.]))]),
        Layer(-2, [Atom(1, 2, np.array([0., 0., -2.]))]),
        Layer(-1, [Atom(1, 2, np.array([0., 0., -1.]))]),
        Layer(0, [Atom(1, 1, np.array([0., 0., 0.]))]),
        Layer(1, [Atom(1, 1, np.array([0., 0., 1.]))]),
        Layer(2, [Atom(1, 1, np.array([0., 0., 2.]))])
    ]

    hm = HarmonicMatrix(fcs, layers)
    layer_assignments = {
        Section.LCB: [-6, -5],
        Section.LC: -4,
        Section.D: [-3, -2, -1],
        Section.RC: 0,
        Section.RCB: [1, 2]
    }
    model = AGF(hm, layer_assignments)

    trans = []
    dos = []

    omegas = np.linspace(1e10, 9e13, n_omegas)
    deltas = 1e-7 * omegas**2
    for omega, delta in zip(omegas, deltas):
        r = model.compute(omega, delta)
        trans.append(r.transmission)
        dos.append((r.dos[1][1] / l0).real)  # dos at centre 'layer' of device

    plot_Fig_6(omegas, trans, dos)


if __name__ == "__main__":
    main()
