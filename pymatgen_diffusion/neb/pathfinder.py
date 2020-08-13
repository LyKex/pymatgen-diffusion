# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

import warnings
import os
from typing import List
import itertools

import numpy as np
from tqdm import tqdm

from pymatgen.core import Structure, PeriodicSite
from pymatgen.core.periodic_table import get_el_sp, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# from pymatgen.core.lattice import get_points_in_sphere
from pymatgen.util.coord import pbc_diff
from pymatgen.io.lammps.data import LammpsBox

__author__ = "Iek-Heng Chu"
__version__ = "1.0"
__date__ = "March 14, 2017"

"""
Algorithms for NEB migration path analysis.
"""


# TODO: (1) ipython notebook example files, unittests


class IDPPSolver:
    """
    A solver using image dependent pair potential (IDPP) algo to get an
    improved initial NEB path. For more details about this algo, please
    refer to Smidstrup et al., J. Chem. Phys. 140, 214106 (2014).

    """

    def __init__(self, structures):
        """
        Initialization.

        Args:
            structures (list of pmg_structure) : Initial guess of the NEB path
                (including initial and final end-point structures).
        """

        latt = structures[0].lattice
        natoms = structures[0].num_sites
        nimages = len(structures) - 2
        target_dists = []

        # Initial guess of the path (in Cartesian coordinates) used in the IDPP
        # algo.
        init_coords = []

        # Construct the set of target distance matrices via linear interpola-
        # tion between those of end-point structures.
        for i in range(1, nimages + 1):
            # Interpolated distance matrices
            dist = structures[0].distance_matrix + i / (nimages + 1) * (
                structures[-1].distance_matrix - structures[0].distance_matrix
            )
            # linear interpolated distance
            target_dists.append(dist)  # with shape in [ni,na,na]
        target_dists = np.array(target_dists)

        # A set of weight functions. It is set as 1/d^4 for each image. Here,
        # we take d as the average of the target distance matrix and the actual
        # distance matrix.
        weights = np.zeros_like(target_dists, dtype=np.float64)
        for ni in range(nimages):
            avg_dist = (target_dists[ni] + structures[ni + 1].distance_matrix) / 2.0
            weights[ni] = 1.0 / (
                avg_dist ** 4 + np.eye(natoms, dtype=np.float64) * 1e-8
            )

        # Set of translational vector matrices (anti-symmetric) for the images.
        translations = np.zeros((nimages, natoms, natoms, 3), dtype=np.float64)
        for ni, i in itertools.product(range(nimages + 2), range(natoms)):
            frac_coords = structures[ni][i].frac_coords
            init_coords.append(latt.get_cartesian_coords(frac_coords))
            # ?consider periodic boundary condition?
            if ni not in [0, nimages + 1]:
                for j in range(i + 1, natoms):
                    img = latt.get_distance_and_image(
                        frac_coords, structures[ni][j].frac_coords
                    )[1]
                    translations[ni - 1, i, j] = latt.get_cartesian_coords(img)
                    translations[ni - 1, j, i] = -latt.get_cartesian_coords(img)

        # build element string list
        elements = []
        for i in structures[0]:
            elements.append(
                str(i.species.elements[0]) if i.species.is_element else None
            )

        self.init_coords = np.array(init_coords).reshape(nimages + 2, natoms, 3)
        self.translations = translations
        self.weights = weights
        self.structures = structures
        self.target_dists = target_dists
        self.nimages = nimages
        self.natoms = natoms
        self.elements = elements
        self.lammps_dump_box = None
        self.paramerters = {}

    def _screen(self, target_dists: np.array, coords, r=5):
        """
        Screen target_dsit assuming atoms behave like rigid balls.
        Nearest neighbors of each atoms will be recognized and will
        push atoms away if clash occurs; pull them over if there is
        too big a gap.

        Args:
            target_dists: linear interpolated distance for images
            between initial and final structures.
            coords: cartesian coordinates of all images including
            initial and final structures.
            r: neighbor range in angstrom

        Return:
            [target_dists] adjusted target_dists.
        """
        # DEBUG
        # print(">>> _screen")

        natoms = self.structures[0].num_sites
        # TODO dynamically setting r, max_bond_length and radius
        max_bond_length = 4.09
        radius = 1.44
        images = []
        # generate the image structures from coords
        for ni in range(1, self.nimages + 1):
            new_sites = []
            for site, cart_coords in zip(self.structures[ni], coords[ni]):
                new_site = PeriodicSite(
                    site.species,
                    coords=cart_coords,
                    lattice=site.lattice,
                    coords_are_cartesian=True,
                    properties=site.properties,
                )
                new_sites.append(new_site)
            images.append(Structure.from_sites(new_sites))
        # find nearest neighbors within r angstroms
        neighbors = []
        for ni in range(self.nimages):
            neighbors.append(images[ni].get_all_neighbors(r))
        neighbor_indices = []
        for ni in range(self.nimages):
            index_temp = []
            for na in range(natoms):
                temp = []
                for nn in range(len(neighbors[ni][na])):
                    temp.append(neighbors[ni][na][nn].index)
                index_temp.append(temp)
            neighbor_indices.append(index_temp)
        # DEBUG
        neighbor_26 = []
        for ni in range(self.nimages):
            neighbor_26.append(neighbor_indices[ni][25])

        # get distance matrix of each images
        dists = []
        for i in range(self.nimages):
            dists.append(images[i].distance_matrix)
        # # unit vector of all neighbors towrads center atom
        # # unit_vec[ni][na][3]
        # unit_vec = []
        # for ni in range(self.nimages):
        #     temp_vec = np.zeros([natoms, 3])
        #     for na in range(natoms):
        #         for nn in range(len(neighbors[ni][na])):
        #             temp_vec[na] += (images[ni].cart_coords[na]
        #                              - neighbors[ni][na][nn].site.coords)
        #         temp_vec[na] = temp_vec[na] / LA.norm(temp_vec[na])
        #     unit_vec.append(temp_vec)

        # adjust anomalies in neighbors
        # neighbors[ni][na][nn][PerodicSite] --> target_dist[ni][na][na]

        # for ni in range(self.nimages):
        #     anomalies_temp = []
        #     isTooFar = False
        #     isTooClose = False
        #     for na, nn in itertools.product(range(natoms),
        #                                     range(len(neighbors[ni][na]))):
        #         d_temp = neighbors[ni][na][ni].distance
        #         diff_bond = d_temp - max_bond_length
        #         diff_radius = 2 * radius - d_temp
        #         if (diff_radius > 0):
        #             isTooClose = True
        #         if (diff_bond > 0):
        #             isTooFar = True
        #         if (isTooClose):
        #             indices_temp.append(neighbors[ni][na][nn].index)
        #             vector_tmep.append()
        #         if (isTooFar):
        #             indices_temp.append(neighbors[ni][na][nn].index)
        #             vector_tmep.append()
        # images[ni].translate_sites(indices = indices_temp,
        #                            vector = vector_temp)
        # obtain new target_distance
        for ni in range(self.nimages):
            # for i, j in itertools.combinations(range(natoms), 2):
            for i in range(natoms):
                for j in range(i + 1, natoms):
                    # d_ij = target_dists[ni][i][j]
                    # if (d_ij < 2 * radius and d_ij > 0):
                    #     # if too close
                    #     target_dists[ni][i][j] = 2 * radius
                    #     target_dists[ni][j][i] = 2 * radius
                    if dists[ni][i][j] < 2 * radius:
                        # if too close
                        target_dists[ni][i][j] += 1
                        target_dists[ni][j][i] += 1
                    if (
                        j in neighbor_indices[ni][i]
                        and dists[ni][i][j] > max_bond_length
                    ):
                        # if too far
                        # this may push atoms even further
                        target_dists[ni][i][j] = max_bond_length
                        target_dists[ni][j][i] = max_bond_length
        return (target_dists, neighbor_26)

    def rerun(self, **run_kwargs):
        # generate target distance from path
        target_dists = []
        for ni in range(self.nimages):
            target_dists.append(self.structures[ni + 1].distance_matrix)
        self.target_dists = target_dists

        return self.run(**run_kwargs)

    def run(
        self,
        maxiter=1000,
        tol=1e-5,
        gtol=1e-3,
        step_size=0.05,
        max_disp=0.05,
        spring_const=5.0,
        species=None,
    ):
        """
        Perform iterative minimization of the set of objective functions in an
        NEB-like manner. In each iteration, the total force matrix for each
        image is constructed, which comprises both the spring forces and true
        forces. For more details about the NEB approach, please see the
        references, e.g. Henkelman et al., J. Chem. Phys. 113, 9901 (2000).

        Args:
            maxiter (int): Maximum number of iterations in the minimization
                process.
            tol (float): Tolerance of the change of objective functions between
                consecutive steps.
            gtol (float): Tolerance of maximum force component (absolute
                value).
            step_size (float): Step size associated with the displacement of
                the atoms during the minimization process.
            max_disp (float): Maximum allowed atomic displacement in each
                iteration.
            spring_const (float): A virtual spring constant used in the NEB-
                like relaxation process that yields so-called IDPP path.
            species (list of string): If provided, only those given species are
                allowed to move. The atomic positions of other species are
                obtained via regular linear interpolation approach.

        Returns:
            [Structure] Complete IDPP path (including end-point structures)
        """

        coords = self.init_coords.copy()
        old_funcs = np.zeros((self.nimages,), dtype=np.float64)
        idpp_structures = [self.structures[0]]

        if species is None:
            indices = list(range(len(self.structures[0])))
        else:
            species = [get_el_sp(sp) for sp in species]
            indices = [
                i for i, site in enumerate(self.structures[0]) if site.specie in species
            ]

            if len(indices) == 0:
                raise ValueError("The given species are not in the system!")

        for n in tqdm(range(maxiter)):
            # Get the sets of objective functions, true and total force
            # matrices.
            funcs, true_forces = self._get_funcs_and_forces(coords)
            tot_forces = self._get_total_forces(
                coords, true_forces, spring_const=spring_const
            )
            # Each atom is allowed to move up to max_disp
            disp_mat = step_size * tot_forces[:, indices, :]
            disp_mat = np.where(
                np.abs(disp_mat) > max_disp, np.sign(disp_mat) * max_disp, disp_mat
            )
            coords[1 : (self.nimages + 1), indices] += disp_mat

            # DEBUG
            # print(">>>step: {} >>>\n total: {}\ndisp: {}".format(n, tot_forces, disp_mat))

            max_force = np.abs(tot_forces[:, indices, :]).max()
            tot_res = np.sum(np.abs(old_funcs - funcs))

            # DEBUG
            # res.append(tot_res)
            # force.append(max_force)
            # print(">>> delta energy", tot_res)
            # print(">>> max force", max_force)
            if tot_res < tol and max_force < gtol:
                # print(">>> delta energy", res)
                # print(">>> max force", force)
                break

            old_funcs = funcs

        else:
            warnings.warn(
                "Maximum iteration number is reached without convergence!", UserWarning
            )

        for ni in range(self.nimages):
            # generate the improved image structure
            new_sites = []

            for site, cart_coords in zip(self.structures[ni + 1], coords[ni + 1]):
                new_site = PeriodicSite(
                    site.species,
                    coords=cart_coords,
                    lattice=site.lattice,
                    coords_are_cartesian=True,
                    properties=site.properties,
                )
                new_sites.append(new_site)

            idpp_structures.append(Structure.from_sites(new_sites))

        # Also include end-point structure.
        idpp_structures.append(self.structures[-1])
        return idpp_structures
        # return (idpp_structures, d_3_4, d_3_9, d_3_10, d_26, neighbor_26)

    @classmethod
    def from_endpoints(cls, endpoints, nimages=5, sort_tol=1.0):
        """
        A class method that starts with end-point structures instead. The
        initial guess for the IDPP algo is then constructed using linear
        interpolation.

        Args:
            endpoints (list of Structure objects): The two end-point
                structures.
            nimages (int): Number of images between the two end-points.
            sort_tol (float): Distance tolerance (in Angstrom) used to match
                the atomic indices between start and end structures. Need to
                increase the value in some cases.
            pi_bond (float): pi_bond thickness in angstrom. When used will
                add this value to all the elements other than carbon.
        """
        try:
            images = endpoints[0].interpolate(
                endpoints[1], nimages=nimages + 1, autosort_tol=sort_tol
            )
        except Exception as e:
            if "Unable to reliably match structures " in str(e):
                warnings.warn(
                    "Auto sorting is turned off because it is unable"
                    " to match the end-point structures!",
                    UserWarning,
                )
                images = endpoints[0].interpolate(
                    endpoints[1], nimages=nimages + 1, autosort_tol=0
                )
            else:
                raise e

        return IDPPSolver(images)

    def _get_funcs_and_forces(self, x):
        """
        Calculate the set of objective functions as well as their gradients,
        i.e. "effective true forces"

        x: cartesian coordinates of each images

        return: funcs, true_forces [ni][nn]
        """
        funcs = []
        funcs_prime = []
        trans = self.translations
        natoms = trans.shape[1]
        weights = self.weights
        target_dists = self.target_dists

        for ni in range(len(x) - 2):
            vec = [x[ni + 1, i] - x[ni + 1] - trans[ni, i] for i in range(natoms)]
            trial_dist = np.linalg.norm(vec, axis=2)
            aux = (
                (trial_dist - target_dists[ni])
                * weights[ni]
                / (trial_dist + np.eye(natoms, dtype=np.float64))
            )

            # Objective function
            func = np.sum((trial_dist - target_dists[ni]) ** 2 * weights[ni])

            # "True force" derived from the objective function.
            grad = np.sum(aux[:, :, None] * vec, axis=1)

            funcs.append(func)
            funcs_prime.append(grad)

        return 0.5 * np.array(funcs), -2 * np.array(funcs_prime)

    @staticmethod
    def get_unit_vector(vec):
        return vec / np.sqrt(np.sum(vec ** 2))

    def _get_total_forces(self, x, true_forces, spring_const):
        """
        Calculate the total force on each image structure, which is equal to
        the spring force along the tangent + true force perpendicular to the
        tangent. Note that the spring force is the modified version in the
        literature (e.g. Henkelman et al., J. Chem. Phys. 113, 9901 (2000)).

        x [niamges, natoms, 3]: cartesian coords of the whole path.
        """

        total_forces = []
        natoms = np.shape(true_forces)[1]

        for ni in range(1, len(x) - 1):
            # TODO add tolerance
            vec1 = (x[ni + 1] - x[ni]).flatten()
            vec2 = (x[ni] - x[ni - 1]).flatten()

            # Local tangent
            tangent = self.get_unit_vector(vec1) + self.get_unit_vector(vec2)
            tangent = self.get_unit_vector(tangent)

            # Spring force
            spring_force = (
                spring_const * (np.linalg.norm(vec1) - np.linalg.norm(vec2)) * tangent
            )

            # Total force
            flat_ft = true_forces[ni - 1].copy().flatten()
            total_force = true_forces[ni - 1] + (
                spring_force - np.dot(flat_ft, tangent) * tangent
            ).reshape(natoms, 3)
            total_forces.append(total_force)

        return np.array(total_forces)

    def clash_removal(
        self,
        path,
        moving_atoms=None,
        step_size=0.05,
        max_disp=0.1,
        max_iter=200,
        gtol=1e-3,
        step_update_method="decay",
        base_step=0.01,
        max_step=0.05,
        pi_bond=0.0,
        is_ionic=False,
        **kwargs,
    ):
        """
        Conduct steric clash removal based on given path.

        path (list of Structures): initial path for clash removal. The first and last
            structures coorepond to the initial and final states.
        k_steric (float): spring constant for steric hinderance.
        steric_threshold (float): atoms with internuclear distance smaller
            than this threshold will be subject to spring force.
        moving_atoms (list): indices of atoms allowed to move.
        **kwargs: keyword arguments for _get_clash_forces_and_energy

        """

        # initialize radii list
        self.radii_list = self._build_radii_list(pi_bond, is_ionic)
        latt = self.structures[0].lattice
        # generate initial structures for each images
        images = path[1:-1]
        # from images generate cart_coords
        # minimization is updated on image_coords not images
        image_coords = []
        for ni, i in itertools.product(range(self.nimages), range(self.natoms)):
            # no getter for cart_coords attribute, so has to do this
            frac_coord_temp = images[ni][i].frac_coords
            image_coords.append(latt.get_cartesian_coords(frac_coord_temp))
        image_coords = np.array(image_coords).reshape(self.nimages, self.natoms, 3)
        # calculate distance matrix for each images
        # original_distances = []
        # for ni in range(self.nimages):
        #     original_distances.append(images[ni].distance_matrix)
        # only moving_atoms are allowed to move
        if moving_atoms is None:
            moving_atoms = list(range(len(self.structures[0])))

        # prepare for step_size update
        max_force = float("inf")
        initial_step_size = step_size

        # get forces and disp_mat of first step

        forces = self._get_clash_forces_and_energy(image_coords=image_coords, **kwargs)
        disp_mat = step_size * forces[:, moving_atoms, :]

        # monitoring
        # attr_force_log = []
        # attr_index_log = []
        # rpl_force_log = []
        # rpl_index_log = []
        # disp_log = []
        progress = 0
        for n in range(max_iter):
            # get forces and energies
            # TODO calculate energy and use it as part of iteration criteria
            forces = self._get_clash_forces_and_energy(
                image_coords=image_coords, **kwargs
            )
            # # monitoring
            # attr_force_log.append(attr_f)
            # attr_index_log.append(attr_index)
            # rpl_force_log.append(rpl_f)
            # rpl_index_log.append(rpl_index)

            # dynamically change step size
            prev_max_force = max_force
            max_force = np.abs(forces).max()

            if step_update_method == "decay":
                decay = 0.01
                if max_force < prev_max_force:
                    step_size = initial_step_size * 1 / (1 + decay * n)
            if step_update_method == "expo":
                if max_force > prev_max_force:
                    step_size = step_size * 0.5
                    # reject last step
                    image_coords -= disp_mat
                    continue
                if max_force < prev_max_force:
                    step_size = step_size * 1.2
            if step_update_method == "triangular":
                if max_force < prev_max_force:
                    step_size = self._get_triangular_step(
                        iteration=n, base_step=base_step, max_step=max_step
                    )

            # update coords
            disp_mat = step_size * forces[:, moving_atoms, :]
            disp_mat = np.where(
                np.abs(disp_mat) > max_disp, np.sign(disp_mat) * max_disp, disp_mat
            )
            image_coords[:, moving_atoms] += disp_mat

            # monitoring
            # get max displacement of each iamges
            # scalar_disp = np.linalg.norm(disp_mat, axis=2)
            # max_scalar_disp = np.amax(scalar_disp, axis=1)
            # disp_log.append(max_scalar_disp)
            # monitoring progress
            curProgress = np.floor(n / max_iter * 10)
            if not curProgress == progress:
                progress = curProgress
                print("{:.0f}0%".format(curProgress))

            # check if meets tolerance requirements
            if max_force < gtol:
                print("max_force < gtol")
                break
        else:
            warnings.warn(
                "Maximum iteration number is reached without convergence!", UserWarning
            )

        # generate the improved image structure
        clash_removed_path = [self.structures[0]]
        for ni in range(self.nimages):
            new_sites = []
            for site, cart_coords in zip(self.structures[ni + 1], image_coords[ni]):
                new_site = PeriodicSite(
                    site.species,
                    coords=cart_coords,
                    lattice=site.lattice,
                    coords_are_cartesian=True,
                    properties=site.properties,
                )
                new_sites.append(new_site)

            clash_removed_path.append(Structure.from_sites(new_sites))

        # Also include end-point structure.
        clash_removed_path.append(self.structures[-1])
        return (
            clash_removed_path,
            # attr_force_log,
            # attr_index_log,
            # rpl_force_log,
            # rpl_index_log,
            # disp_log,
        )

    def _NEB_atoms(self, threshold):
        """
        Compare initial and final states of the given path and determine if an atom has
        large enough displacement.

        Args
        initial: fractional coords of initial state
        final: fractional coords of final state
        threshold: numerical threshold in cartesian coords
        """
        NEB_atoms = []
        initial = self.structures[0].frac_coords
        final = self.structures[-1].frac_coords
        frac_diff = pbc_diff(initial, final)
        disp = np.linalg.norm(
            self.structures[0].lattice.get_cartesian_coords(frac_diff), axis=1
        )
        for i in range(self.natoms):
            if disp[i] > threshold:
                NEB_atoms.append(i)
        # debug
        # print("\nNEB atoms list\nIndex | Element | Displacement")
        print("NEB atoms list:")
        for n in NEB_atoms:
            print("{}\t{}\t{}".format(n, self.elements[n], disp[n]))
        return NEB_atoms

    def clash_removal_NEB(
        self,
        path,
        maxiter,
        use_idpp,
        dump_dir,
        dump_cr=True,
        dump_idpp=True,
        con_co=1.0,
        dump_total=True,
        moving_atoms=None,
        step_size=0.05,
        max_disp=0.05,
        max_iter=200,
        gtol=1e-3,
        step_update_method=None,
        base_step=0.01,
        max_step=0.05,
        spring_const=5.0,
        k_bonded=5.0,
        k_steric=0.15,
        NEB_atoms=[],
        NEB_threshold=2.0,
        pi_bond=0.0,
        is_ionic=False,
        **kwargs,
    ):
        """
        Conduct clash removal process on given path with NEB.

        Args:
        path [ni+2, na, 3]: initial fractional coords path for clash removal
        maxiter (int): maximum iteration path (list of Structures): initial path for
        clash removal. The first and last structures coorepond to the initial and final
        states.
        moving_atoms (list of int): index of atoms that are allowed to move
        during NEB. If None, then all atoms are allowed to move.
        """

        self.radii_list = self._build_radii_list(pi_bond, is_ionic)
        # Construct cartesian coords for path
        # minimization is updated on path_coords not images

        # from given path generate cart_coords
        latt = self.structures[0].lattice
        initial_images = path[1:-1]
        image_coords = []
        for ni, i in itertools.product(range(self.nimages), range(self.natoms)):
            # no getter for cart_coords attribute, so has to do this
            frac_coord_temp = initial_images[ni][i].frac_coords
            image_coords.append(latt.get_cartesian_coords(frac_coord_temp))
        image_coords = np.array(image_coords).reshape(self.nimages, self.natoms, 3)
        # add initial and final state to path_coords
        path_coords = [self.init_coords[0], *image_coords, self.init_coords[-1]]
        path_coords = np.array(path_coords)

        # if moving_atoms is [], then all atoms are allowed to move
        if moving_atoms is None:
            moving_atoms = list(range(len(self.structures[0])))

        # find NEB atoms
        NEB_atoms += self._NEB_atoms(NEB_threshold)
        if not NEB_atoms:
            NEB_atoms = range(self.natoms)
            warnings.warn("No NEB atoms detected. All atoms are considered NEB atoms.")

        # initialize for IDPP force calculation
        # old_funcs = np.zeros((self.nimages,), dtype=np.float64)
        # idpp_structures = [self.structures[0]]

        # initialize for main loop
        max_forces = [float("inf")]
        initial_step = step_size
        for n in tqdm(range(maxiter)):

            # true force = IDPP force + CR force
            # total force = true force (perpendicular to tangent)
            #             + spring force (along tangent)
            if k_bonded == 0 and k_steric == 0:
                dump_cr, dump_idpp, dump_total = False, False, False
                true_forces = np.zeros([self.nimages, self.natoms, 3])
            else:
                # for each iteration clash force is evaluated on latest image coords
                clash_forces = self._get_clash_forces_and_energy(
                    image_coords=path_coords[1:-1],
                    NEB_atoms=NEB_atoms,
                    k_bonded=k_bonded,
                    k_steric=k_steric,
                    **kwargs,
                )
                if use_idpp:
                    _, idpp_forces = self._get_funcs_and_forces(path_coords)
                    # normalize CR_force
                    conjugated_cr = np.zeros(np.shape(clash_forces))
                    idpp_forces_norm = np.linalg.norm(idpp_forces, axis=2)
                    cr_forces_norm = np.linalg.norm(clash_forces, axis=2)
                    for ni in range(self.nimages):
                        for na in range(self.natoms):
                            if not cr_forces_norm[ni][na] == 0:
                                conjugated_cr[ni][na] = (
                                    clash_forces[ni][na] / cr_forces_norm[ni][na]
                                ) * idpp_forces_norm[ni][na]
                    true_forces = idpp_forces + con_co * conjugated_cr
                else:
                    true_forces = clash_forces

            # path_coords includes initial and final states
            total_forces = self._get_total_forces(
                path_coords, true_forces, spring_const
            )

            # output dump file for each imaegs
            if dump_cr:
                self.dump_writer("dump_cr", dump_dir, path_coords, clash_forces, n)
            if dump_idpp:
                self.dump_writer("dump_idpp", dump_dir, path_coords, idpp_forces, n)
            if dump_total:
                self.dump_writer("dump_total", dump_dir, path_coords, total_forces, n)

            # calculate displacement. disp_mat[ni][nn][3]
            disp_mat = step_size * total_forces[:, moving_atoms, :]
            disp_mat = np.where(
                np.abs(disp_mat) > max_disp, np.sign(disp_mat) * max_disp, disp_mat
            )
            # update images_coords
            path_coords[1:-1, moving_atoms] += disp_mat

            # DEBUG
            # print(">>>step: {} >>>\n total: {}\ndisp: {}".format(n, total_forces, disp_mat))

            # calculate max force and store
            max_forces.append(np.abs(total_forces[:, moving_atoms, :]).max())
            # stop criteria
            if max_forces[-1] < gtol:
                print("\n>>> Total force converges! >>>")
                break

            # change step size for better optimization
            if step_update_method == "decay" and max_forces[-1] < max_forces[-2]:
                step_size = initial_step * (1 / (1 + 0.01 * n))

        else:
            print("\n>>> max force at end of optimization: {} >>>".format(max_forces[-1]))
            warnings.warn(
                "CR-NEB: Maximum iteration number is reached without convergence!",
                UserWarning,
            )

        # generate the improved image structure
        clash_removed_path = [self.structures[0]]
        for ni in range(self.nimages):
            new_sites = []
            for site, cart_coords in zip(self.structures[ni + 1], path_coords[ni + 1]):
                new_site = PeriodicSite(
                    site.species,
                    coords=cart_coords,
                    lattice=site.lattice,
                    coords_are_cartesian=True,
                    properties=site.properties,
                )
                new_sites.append(new_site)

            clash_removed_path.append(Structure.from_sites(new_sites))
        # Also include end-point structure.
        clash_removed_path.append(self.structures[-1])

        return clash_removed_path

    def lammps_dump_str(self, coords, force_matrix, step):
        """
        Output cartesian coordinates and corresponding total forces in
        lammps dump format.

        Args:
        coords [nn, 3]: Cartesian coords of each atoms during optimization
        force_matrix [nn, 3]: force to output
        step: time step starts with 0
        Return:
        str: dump string representation for each time step.
        """
        # concatenation by join a list is significantly faster than += string
        dump = []
        # append header
        dump.append("ITEM: TIMESTEP\n")
        dump.append("{}\n".format(step))
        dump.append("ITEM: NUMBER OF ATOMS\n")
        dump.append("{}\n".format(self.natoms))

        # append box bounds in lammps dump format
        if self.lammps_dump_box is None:
            lmpbox = self.lattice_2_dump_lmpbox(self.structures[0].lattice)
            self.lammps_dump_box = self.get_box_str(lmpbox)
            dump.append(self.lammps_dump_box)
        else:
            dump.append(self.lammps_dump_box)

        # append coords and forces
        dump.append("ITEM: ATOMS id type x y z fx fy fz\n")
        for i in range(self.natoms):
            dump.append("{} {} ".format(i, self.elements[i]))
            for j in range(3):
                dump.append("{:.6f} ".format(coords[i][j]))
            for j in range(3):
                dump.append("{:.6f} ".format(force_matrix[i][j]))
            dump.append("\n")
        return "".join(dump)

    def lattice_2_dump_lmpbox(self, lattice, origin=(0, 0, 0)):
        """
        Converts a lattice object to LammpsBox,Adapted from
        pytmatgen.core.io.lammps.data.lattice_2_lmpbox. The original lmpbox is by lammps
        data format which is different from dump format in bounds definition. Note that
        this method will result in wrong lattice matrix so cannnot converted back into
        pymatgen lattice.

        Args:
            lattice (Lattice): Input lattice.
            origin: A (3,) array/list of floats setting lower bounds of
                simulation box. Default to (0, 0, 0).

        Returns:
            LammpsBox

        """
        a, b, c = lattice.abc
        xlo, ylo, zlo = origin
        xhi = a + xlo
        m = lattice.matrix
        xy = np.dot(m[1], m[0] / a)
        yhi = np.sqrt(b ** 2 - xy ** 2) + ylo
        xz = np.dot(m[2], m[0] / a)
        yz = (np.dot(m[1], m[2]) - xy * xz) / (yhi - ylo)
        zhi = np.sqrt(c ** 2 - xz ** 2 - yz ** 2) + zlo
        tilt = None if lattice.is_orthogonal else [xy, xz, yz]
        xlo_bound = xlo + min(0.0, xy, xz, xy + xz)
        xhi_bound = xhi + max(0.0, xy, xz, xy + xz)
        ylo_bound = ylo + min(0.0, yz)
        yhi_bound = yhi + max(0.0, yz)
        bounds = [[xlo_bound, xhi_bound], [ylo_bound, yhi_bound], [zlo, zhi]]
        return LammpsBox(bounds, tilt)

    def get_box_str(self, lmpbox: LammpsBox):
        is_orthogonal = lmpbox.tilt is None
        m = lmpbox.bounds.copy()
        out_str = []
        out_str.append(
            "ITEM: BOX BOUNDS pp pp pp\n"
            if is_orthogonal
            else "ITEM: BOX BOUNDS xy xz yz pp pp pp\n"
        )
        for i in range(len(m)):
            for j in m[i]:
                out_str.append("{:.6f} ".format(j))
            out_str.append("\n" if is_orthogonal else "{:.6f}\n".format(lmpbox.tilt[i]))
        return "".join(out_str)

    def dump_writer(self, filename, dump_dir, coords, forces, n):
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        for i in range(self.nimages):
            with open(
                os.path.join(dump_dir, "{}_{:02d}".format(filename, i + 1)), "a"
            ) as f:
                f.write(self.lammps_dump_str(coords[i + 1], forces[i], n))
        return True

    @staticmethod
    def _get_triangular_step(
        iteration: int, half_period=5, base_step=0.05, max_step=0.1
    ):
        """
        Given the inputs, calculates the step_size that should be applicable
        for this iteration using CLR technique from Anand Saha
        (http://teleported.in/posts/cyclic-learning-rate/).
        """
        cycle = np.floor(1 + iteration / (2 * half_period))
        x = np.abs(iteration / half_period - 2 * cycle + 1)
        step = base_step + (max_step - base_step) * np.maximum(0, (1 - x))
        return step

    @staticmethod
    def _decay(iteration: int, initial_step_size, decay):
        step_size = initial_step_size * 1 / (1 + decay * iteration)
        return step_size

    def _get_clash_forces_and_energy(
        self,
        image_coords,
        NEB_atoms,
        max_bond_tol=0.2,
        # original_distances=None,
        k_steric=5.0,
        k_bonded=0.05,
        repul_tol=0.0,
        steric_tol=1e-8,
        elastic_limit=0.2,
        **kwargs,
    ):
        """
        calculate forces and energies

        Args:
        image_coords ([ni,na,3]): current cart coords of each images
        k_steric (float): spring constant for steric hinderance
        steric_shreshold (float): atoms with internulcear distance smaller than
            this value will be subject to repulsive force.
        steric_tol (float): atoms too close together will be regarded as same atoms

        Returns:
            Clash_forces[ni, nn, 3]
        """
        # debug
        # print(k_steric, k_bonded, repul_tol, max_bond_tol)

        # get lattice
        lattice = self.structures[0].lattice
        # get frac_coords of current image_coords
        frac_image_coords = []
        for ni in range(self.nimages):
            frac_image_coords.append(lattice.get_fractional_coords(image_coords[ni]))

        # find steric hindered atoms
        steric_hindered = []
        for ni in range(self.nimages):
            #     # get frac_coord diff considering pbc
            #     diff = pbc_diff(frac_image_coords[ni][i], frac_image_coords[ni][j])
            #     # convert back to cart coords
            #     dist = np.linalg.norm(lattice.get_cartesian_coords(diff))
            #     if (
            #         dist < self._get_steric_threshold(i, j, repul_tol)
            #         and dist > steric_tol
            #     ):
            #         steric_hindered.append([ni, i, j, dist])
            # calculate cartesian dist of each atom pairs
            diff = lattice.get_all_distances(
                frac_image_coords[ni], frac_image_coords[ni]
            )
            for i, j in itertools.combinations(range(self.natoms), 2):
                if (
                    diff[i][j] < self._get_steric_threshold(i, j, repul_tol)
                    and diff[i][j] > steric_tol
                ):
                    steric_hindered.append([ni, i, j, diff[i][j]])

        # calculate repulsive forces
        repulsive_forces = np.zeros((self.nimages, self.natoms, 3), dtype=np.float64)
        for case in steric_hindered:
            ni, i, j, r = case
            coord1 = image_coords[ni][i]
            coord2 = image_coords[ni][j]
            # direction pointing towards atom i
            direction = self.get_direction_pbc(coord1, coord2)
            delta_d = abs(r - self._get_steric_threshold(i, j, repul_tol))
            f = k_steric * delta_d ** 2 * direction
            # force and counter force
            repulsive_forces[ni][i] += f
            repulsive_forces[ni][j] += -f
        # monitor repulsive forces
        # scalar_rpl_force = np.linalg.norm(repulsive_forces, axis=2)
        # max_rpl_force_index = np.argmax(scalar_rpl_force, axis=1)
        # max_rpl_force = np.amax(scalar_rpl_force, axis=1)

        # calculate attractive forces
        bonded_neighbors = self._find_bonded_neighbors(
            frac_image_coords, image_coords, NEB_atoms, **kwargs
        )
        attractive_forces = np.zeros((self.nimages, self.natoms, 3), dtype=np.float64)
        for nei in bonded_neighbors:
            # each element in bonded_neighbors is a BondedNeighbor object
            ni = nei.image
            i = nei.index
            n_i = nei.n_index
            d = nei.nn_distance
            coord_bonded = image_coords[ni][i]
            coord_pulling = image_coords[ni][n_i]

            # get direction (towards pulling atoms) considering PBC
            direction = self.get_direction_pbc(coord_pulling, coord_bonded)

            # get displacement and calculate force
            delta_d = d - self._get_max_bond_length(i, n_i, max_bond_tol=max_bond_tol)
            f = (k_bonded * delta_d ** 2) * direction

            # apply force on atoms
            # attractive_forces have no counter forces unless they are both in atom set
            attractive_forces[ni][i] += f
            # DEBUG monitor attractive forces
            # print(
            #     "image: {} on atom: {} direction: {} from {} at {} distance {}".format(
            #         ni, i, direction, nei.n_index, image_coords[ni][nei.n_index], nei.nn_distance
            #     )
            # )
        clash_forces = repulsive_forces + attractive_forces
        return clash_forces

    def _find_bonded_neighbors(
        self,
        frac_coords,
        image_coords,
        NEB_atoms: List,
        r_threshold: float = 3.0,
        numerical_tol: float = 1e-8,
        max_bond_tol=0.2,
    ):
        """
        Alls atoms are assumed to be connected. For any atoms, if its closest neighbor
        is further than the corresponding max_bond_length, then all its neighbors will
        pull the atom. The neighbor atom pairs are searched within a radius using
        pymatgen.core.lattice.get_points_in_sphere() which will try to use cthyon code.

        Args:
            frac_coords ([ni,na,3]): fractional coords of current optimizing
                structure.
            moving_sites: list of indices of manually selected NEB atoms
            r (float): radius of the search range.

        Return:
            cases (2D list): [image number, atom index, BondedNeighbor 1,
             BondedNeighbor 2...]
        """

        lattice = self.structures[0].lattice

        # add moving sites to NEB_atoms to allow manual selection
        NEB_atoms = set(NEB_atoms)

        cases = []
        for ni in range(self.nimages):
            clash_atoms = set()
            # find clash atoms around NEB atoms
            for n in NEB_atoms:
                # get_points_in_sphere return
                # [fcoord, dist, indices, supercell_image]
                _, _, temp_atoms, _ = lattice.get_points_in_sphere(
                    frac_coords[ni],
                    image_coords[ni][n],
                    r_threshold,
                    zip_results=False,
                )
                clash_atoms = clash_atoms | set(temp_atoms)
            # for NEB atoms and clash atoms, check if their neighbors need bonding
            atoms_set = NEB_atoms | clash_atoms
            # print("final atoms set for image {}: {}".format(ni+1, atoms_set))
            for i in atoms_set:
                atom_neighbors = lattice.get_points_in_sphere(
                    frac_coords[ni], image_coords[ni][i], r_threshold, zip_results=True,
                )
                for _, d, index, _ in atom_neighbors:
                    # check if the neighbor is far enough
                    # print("atom: {} neighbor: {} distance: {}".format(i, index, d))
                    if d > self._get_max_bond_length(i, index, max_bond_tol):
                        # atom i is attracted to atom index
                        cases.append(BondedNeighbor(i, index, d, ni))
        # debug
        # print(cases,"\n")
        return cases

    def get_direction_pbc(self, coords1, coords2):
        """
        return a unit vector pointing towards coord1
        coords1 and coords2: cartesian coordinates of a single atom
        """
        latt = self.structures[0].lattice
        frac_coords1 = latt.get_fractional_coords(coords1)
        frac_coords2 = latt.get_fractional_coords(coords2)
        coord_diff = pbc_diff(frac_coords1, frac_coords2)
        return self.get_unit_vector(latt.get_cartesian_coords(coord_diff))

    def _get_steric_threshold(self, atom_1, atom_2, tol):
        d = self.radii_list[atom_1] + self.radii_list[atom_2] - tol
        return d

    def _get_max_bond_length(self, atom1, atom2, max_bond_tol):
        d = self.radii_list[atom1] + self.radii_list[atom2] + max_bond_tol
        return d

    def _build_radii_list(self, pi_bond=0.0, is_ionic=False):
        """
        Build radii list of each atoms in the structure. If steric_threshold or
        max_bond_length is not manually set, radii_list should be used to
        automaticallty generate those parameters according to atomic radius.
        """
        # radii_table will overide radii of pymatgen element atomic_radius.
        # The value for Alkali metals (Li, Na, K) is measured on models of ions
        # adsorped on graphene, therefore it has accounted for the pi_bond radius.
        # Other metals are measured on corresponding unit cells of Material Studio.
        atomic_radii_table = {
            "C": 0.7,
            "N": 0.8,
            "H": 1,  # for testing purpose only
            "Al": 1.43,
            "Ni": 1.25,
            "Cu": 1.278,
            "Ag": 1.445,
        }
        ionic_radii_table = {
            # I A
            "H": 0.365,
            "Li": 0.9,
            "Na": 1.16,
            "K": 1.52,
            "Rb": 1.66,
            # II A
            "Be": 0.59,
            "Mg": 0.86,
            "Ca": 1.14,
            "Sr": 1.32,
            # III A
            "B": 0.41,
            "Al": 0.68,
            "Ga": 1.25,
            "In": 0.94,
            "Ti": 0.5,
            # V A
            "Bi": 1.5,
            "As": 1.2,
            # VI A
            "O": 1.44,
            # "O": 0.73, # conflict
            "S": 1.8,
            "Se": 1.16,
            "Te": 1.35,
            # VII A
            "F": 0.71,
            "Cl": 0.99,
            "Br": 1.14,
            "I": 1.33,
            "W": 0.6,
        }
        if is_ionic:
            radii_table = ionic_radii_table
        else:
            radii_table = atomic_radii_table

        radii = []
        output_list = {}
        for e in self.elements:
            if e in radii_table:
                radii.append(radii_table[e] + pi_bond)
            else:
                radii.append(Element(e).atomic_radius + pi_bond)
            if e not in output_list:
                output_list[e] = radii[-1]
        # show which radius is being used
        for k in output_list:
            print("element:{} radius:{}".format(k, output_list[k]))
        return radii

    # def _get_radius(self, atom_index, pi_bond):
    #     structure = self.structures[0]

    #     if structure[atom_index].species.is_element:
    #         elmnt = structure[atom_index].species.elements[0]
    #         # TODO ionic radii list should be added
    #         if elmnt == Element("Li"):
    #             r = 0.90 + pi_bond
    #             # r = 1.34 + pi_bond
    #         elif set() == Element("Na"):
    #             r = 1.16 + pi_bond
    #         elif elmnt == Element("K"):
    #             r = 1.52 + pi_bond
    #         elif elmnt == Element("C"):
    #             r = elmnt.atomic_radius
    #         else:
    #             r = elmnt.atomic_radius + pi_bond
    #     else:
    #         raise ValueError("sites in structures should be elements not compositions")
    #     print("element:{} radius:{}".format(elmnt, r))
    #     return r


class BondedNeighbor:
    """
    container for bonded neighbors used during calculation of bonded force
    """

    def __init__(self, index, n_index, nn_distance, image):
        """
        index: index of the atom in question
        n_index: the index of neighbor atom to the atom in question
        nn_distance: the distance between two atoms
        image: the number of image
        """
        self.index = index
        self.n_index = n_index
        self.nn_distance = nn_distance
        self.image = image

    def __repr__(self):
        return (
            "atom index: {} neighbor index:{} " "neighbor distance: {} in image {}\n"
        ).format(self.index, self.n_index, self.nn_distance, self.image)


class MigrationPath:
    """
    A convenience container representing a migration path.
    """

    def __init__(self, isite, esite, symm_structure):
        """
        Args:
            isite: Initial site
            esite: End site
            symm_structure: SymmetrizedStructure
        """
        self.isite = isite
        self.esite = esite
        self.symm_structure = symm_structure
        self.msite = PeriodicSite(
            esite.specie, (isite.frac_coords + esite.frac_coords) / 2, esite.lattice
        )
        sg = self.symm_structure.spacegroup
        for i, sites in enumerate(self.symm_structure.equivalent_sites):
            if sg.are_symmetrically_equivalent([isite], [sites[0]]):
                self.iindex = i
            if sg.are_symmetrically_equivalent([esite], [sites[0]]):
                self.eindex = i

    def __repr__(self):
        return (
            "Path of %.4f A from %s [%.3f, %.3f, %.3f] "
            "(ind: %d, Wyckoff: %s) to %s [%.3f, %.3f, %.3f] "
            "(ind: %d, Wyckoff: %s)"
            % (
                self.length,
                self.isite.specie,
                self.isite.frac_coords[0],
                self.isite.frac_coords[1],
                self.isite.frac_coords[2],
                self.iindex,
                self.symm_structure.wyckoff_symbols[self.iindex],
                self.esite.specie,
                self.esite.frac_coords[0],
                self.esite.frac_coords[1],
                self.esite.frac_coords[2],
                self.eindex,
                self.symm_structure.wyckoff_symbols[self.eindex],
            )
        )

    @property
    def length(self):
        """
        Returns:
            (float) Length of migration path.
        """
        return np.linalg.norm(self.isite.coords - self.esite.coords)

    def __hash__(self):
        return self.iindex + self.eindex

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if self.symm_structure != other.symm_structure:
            return False

        return self.symm_structure.spacegroup.are_symmetrically_equivalent(
            (self.isite, self.msite, self.esite),
            (other.isite, other.msite, other.esite),
        )

    def get_structures(self, nimages=5, vac_mode=True, idpp=False, **idpp_kwargs):
        """
        Generate structures for NEB calculation.

        Args:
            nimages (int): Defaults to 5. Number of NEB images. Total number of
                structures returned in nimages+2.
            vac_mode (bool): Defaults to True. In vac_mode, a vacancy diffusion
                mechanism is assumed. The initial and end sites of the path
                are assumed to be the initial and ending positions of the
                vacancies. If vac_mode is False, an interstitial mechanism is
                assumed. The initial and ending positions are assumed to be
                the initial and ending positions of the interstitial, and all
                other sites of the same specie are removed. E.g., if NEBPaths
                were obtained using a Li4Fe4P4O16 structure, vac_mode=True
                would generate structures with formula Li3Fe4P4O16, while
                vac_mode=False would generate structures with formula
                LiFe4P4O16.
            idpp (bool): Defaults to False. If True, the generated structures
                will be run through the IDPPSolver to generate a better guess
                for the minimum energy path.
            **idpp_kwargs: Passthrough kwargs for the IDPPSolver.run.

        Returns:
            [Structure] Note that the first site of each structure is always
            the migrating ion. This makes it easier to perform subsequent
            analysis.
        """
        migrating_specie_sites = []
        other_sites = []
        isite = self.isite
        esite = self.esite

        for site in self.symm_structure.sites:
            if site.specie != isite.specie:
                other_sites.append(site)
            else:
                if vac_mode and (
                    isite.distance(site) > 1e-8 and esite.distance(site) > 1e-8
                ):
                    migrating_specie_sites.append(site)

        start_structure = Structure.from_sites(
            [self.isite] + migrating_specie_sites + other_sites
        )
        end_structure = Structure.from_sites(
            [self.esite] + migrating_specie_sites + other_sites
        )

        structures = start_structure.interpolate(
            end_structure, nimages=nimages + 1, pbc=False
        )

        if idpp:
            solver = IDPPSolver(structures)
            return solver.run(**idpp_kwargs)

        return structures

    def write_path(self, fname, **kwargs):
        """
        Write the path to a file for easy viewing.

        Args:
            fname (str): File name.
            **kwargs: Kwargs supported by NEBPath.get_structures.
        """
        sites = []
        for st in self.get_structures(**kwargs):
            sites.extend(st)
        st = Structure.from_sites(sites)
        st.to(filename=fname)


class DistinctPathFinder:
    """
    Determines symmetrically distinct paths between existing sites.
    The path info can then be used to set up either vacancy or interstitial
    diffusion (assuming site positions are known). Note that this works mainly
    for atomic mechanism, and does not work for correlated migration.
    """

    def __init__(
        self,
        structure,
        migrating_specie,
        max_path_length=None,
        symprec=0.1,
        perc_mode=">1d",
    ):
        """
        Args:
            structure: Input structure that contains all sites.
            migrating_specie (Specie-like): The specie that migrates. E.g.,
                "Li".
            max_path_length (float): Maximum length of NEB path in the unit
                of Angstrom. Defaults to None, which means you are setting the
                value to the min cutoff until finding 1D or >1D percolating
                paths.
            symprec (float): Symmetry precision to determine equivalence.
            perc_mode(str): The percolating type. Default to ">1d", because
                usually it is used to find possible NEB paths to form
                percolating networks. If you just want to check the min 1D
                percolation, set it to "1d".
        """
        self.structure = structure
        self.migrating_specie = get_el_sp(migrating_specie)
        self.symprec = symprec
        a = SpacegroupAnalyzer(self.structure, symprec=self.symprec)
        self.symm_structure = a.get_symmetrized_structure()

        junc = 0
        distance_list = []
        max_r = max_path_length or min(structure.lattice.abc)
        junc_cutoff = max_r
        for sites in self.symm_structure.equivalent_sites:
            if sites[0].specie == self.migrating_specie:
                site0 = sites[0]
                dists = []
                neighbors = self.symm_structure.get_neighbors(site0, r=max_r)
                for nn in sorted(neighbors, key=lambda nn: nn.distance):
                    if nn.site.specie == self.migrating_specie:
                        dists.append(nn.distance)
                if len(dists) > 2:
                    junc += 1
                distance_list.append(dists)
        # Avoid isolated atoms (# of neighbors < 2)
        if len(sorted(distance_list, key=len)[0]) < 2:
            path_cutoff = max_r
        # We define junction as atoms have at least three paths including
        # equivalent ones.
        elif junc == 0:
            path_cutoff = sorted(distance_list, key=lambda d: d[1])[-1][1]
        else:
            # distance_list are sorted as [[a0,a1,a2],[b0,b1,b2],
            # [c0,c1,c2],...]
            # in which a0<a1<a2,b0<b1<b2,...
            # path_cutoff = max(a1,b1,c1,...), junc_cutoff=min(a2,b2,c2)
            path_cutoff = sorted(distance_list, key=lambda d: d[1])[-1][1]
            junc_distance_list = [d for d in distance_list if len(d) >= 3]
            junc_cutoff = sorted(junc_distance_list, key=lambda d: d[2])[0][2]

        if max_path_length is None:
            if perc_mode.lower() == "1d":
                self.max_path_length = path_cutoff
            else:
                self.max_path_length = max(junc_cutoff, path_cutoff)
        else:
            self.max_path_length = max_path_length

    def get_paths(self):
        """
        Returns:
            [MigrationPath] All distinct migration paths.
        """
        paths = set()
        for sites in self.symm_structure.equivalent_sites:
            if sites[0].specie == self.migrating_specie:
                site0 = sites[0]
                for nn in self.symm_structure.get_neighbors(
                    site0, r=round(self.max_path_length, 3) + 0.01
                ):
                    if nn.site.specie == self.migrating_specie:
                        path = MigrationPath(site0, nn.site, self.symm_structure)
                        paths.add(path)

        return sorted(paths, key=lambda p: p.length)

    def write_all_paths(self, fname, nimages=5, **kwargs):
        """
        Write a file containing all paths, using hydrogen as a placeholder for
        the images. H is chosen as it is the smallest atom. This is extremely
        useful for path visualization in a standard software like VESTA.

        Args:
            fname (str): Filename
            nimages (int): Number of images per path.
            kwargs: Passthrough kwargs to path.get_structures.
        """
        sites = []
        for p in self.get_paths():
            structures = p.get_structures(
                nimages=nimages, species=[self.migrating_specie], **kwargs
            )
            sites.append(structures[0][0])
            sites.append(structures[-1][0])
            for s in structures[1:-1]:
                sites.append(PeriodicSite("H", s[0].frac_coords, s.lattice))
        sites.extend(structures[0].sites[1:])
        Structure.from_sites(sites).to(filename=fname)
