#!/home/liugy/miniconda3/envs/my_pymatgen/bin/python
# coding: UTF-8

import sys
import argparse
import numpy as np

from pymatgen.core import Structure
from pymatgen.util.coord import pbc_diff


def rmsd_pbc(file_path_1, file_path_2):
    """
    Calculate absolute root-mean-square diffence between two structures. No rotation nor
    recentering will be considered. Periodic boundary condition will be considered.
    """
    try:
        a = Structure.from_file(filename=file_path_1)
        b = Structure.from_file(filename=file_path_2)
    except Exception:
        print("File import failed.")

    # check if two structures are valid for compare
    natoms = check_validity(a, b)

    # get fractional coords of each structure
    a_frac = [a[i].frac_coords for i in range(natoms)]
    b_frac = [b[i].frac_coords for i in range(natoms)]

    # get frac_diff considering pbc (abs(diff) <= 0.5)
    frac_diff = pbc_diff(a_frac, b_frac)

    # convert to cartesian coords difference
    cart_diff = a.lattice.get_cartesian_coords(frac_diff)

    # original definition of RMSD
    # return np.sqrt(np.sum(cart_diff ** 2) / natoms)

    # revised definition. The top 5 deviation is considered
    # aiming to better reflect the difference in structures
    return np.sum(np.sort(np.abs(cart_diff))[:5]) / 5


def check_validity(structure_a, structure_b):
    if len(structure_a) == len(structure_b):
        return len(structure_a)
    else:
        sys.exit("Invalid structures to compare.")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(
     description="Calculate root-mean-square-difference between two structures.")
    argparse.add_argument("structure_1", nargs=1)
    argparse.add_argument("structure_2", nargs=1)
    args = argparse.parse_args()

    file_a = args.structure_1[0]
    file_b = args.structure_2[0]
    print(rmsd_pbc(file_a, file_b))
