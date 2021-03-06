from setuptools import setup, find_packages
from agf import __version__

setup(
    name="agf",
    version=__version__,
    description="A package for computing Atomistic Green's Functions based on the Zhang-Mingo method.",
    url="https://github.com/araghukas/agf.git",
    author_email="ghukasa@mcmaster.ca",
    author="Ara Ghukasyan",
    license="MIT",
    install_requires=["numpy", "numba"],
    packages=find_packages()
)
