"""numerical constants for unit conversion"""
from math import sqrt, pi
import scipy.constants as si

# multiply by x in original units to produce x in S.I. units.
LAMMPS_metal_to_Hz = sqrt(si.e / si.m_u) / si.angstrom / (2 * pi)  # -> [Hz]
