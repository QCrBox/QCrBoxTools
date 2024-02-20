# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

"""
This module, part of the `qcrboxtools` package, offers tools for analyzing and comparing
crystallographic data stored in CIF (Crystallographic Information File) format. The
functions within can be used to computate of differences in atomic positions and anisotropic
displacement parameters (ADPs) between different CIF datasets. Additionally, it includes
a function to check the convergence of these parameters against specified criteria.

Functions:
- cell_dict2atom_sites_dict: Converts cell dictionary to atom sites dictionary
  with a transformation matrix.
- add_cart_pos: Converts atomic positions from fractional to Cartesian coordinates.
- position_difference: Calculates the positional differences between two CIF datasets.
- anisotropic_adp_difference: Computes differences in anisotropic ADPs between two CIF datasets.
- check_converged: Determines if the differences in atomic positions and ADPs meet
  specified convergence criteria.
"""

from pathlib import Path
from typing import Union, Dict, List, Any, Tuple
import numpy as np
from ..cif.read import read_cif_safe, cifdata_str_or_index, split_sus, split_su_single

def cell_dict2atom_sites_dict(
    cell_dict: Dict[str, Union[float, np.ndarray]]
) -> Dict[str, Union[str, np.ndarray]]:
    """
    Converts a cell dictionary into an atom sites dictionary that includes transformation matrix.

    The transformation matrix is generated based on the cell parameters. This matrix contains
    the three lattice vectors as rows. It is used to convert from fractional to Cartesian
    coordinates.

    Args:
        cell_dict (Dict[str, Union[float, np.ndarray]]): A dictionary representing a unit cell.
        It should contain the following keys: '_cell.length_a', '_cell.length_b', '_cell.length_c',
        '_cell_angle.alpha', '_cell.angle_beta', '_cell.angle_gamma'. The corresponding values
        should be floats representing the cell lengths (a, b, c) in angstroms and the cell angles
        (alpha, beta, gamma) in degrees.

    Returns:
        Dict[str, Union[str, np.ndarray]]: A dictionary containing the transformation matrix
        and its description. The keys are '_atom_sites_cartn_transform.axes' (with value being
        a string description of the transformation axes) and '_atom_sites_Cartn_tran_matrix'
        (with value being a 3x3 numpy array representing the transformation matrix).
    """
    a = split_su_single(cell_dict['_cell.length_a'])[0]
    b = split_su_single(cell_dict['_cell.length_b'])[0]
    c = split_su_single(cell_dict['_cell.length_c'])[0]
    alpha = split_su_single(cell_dict['_cell.angle_alpha'])[0] / 180.0 * np.pi
    beta = split_su_single(cell_dict['_cell.angle_beta'])[0] / 180.0 * np.pi
    gamma = split_su_single(cell_dict['_cell.angle_gamma'])[0] / 180.0 * np.pi
    matrix = np.array(
        [
            [
                a,
                b * np.cos(gamma),
                c * np.cos(beta)
            ],
            [
                0,
                b * np.sin(gamma),
                c * (np.cos(alpha) - np.cos(gamma) * np.cos(beta)) / np.sin(gamma)
            ],
            [
                0,
                0,
                c / np.sin(gamma) * np.sqrt(1.0 - np.cos(alpha)**2 - np.cos(beta)**2
                                            - np.cos(gamma)**2
                                            + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
            ]
        ]
    )
    atom_sites_dict = {
        '_atom_sites_cartn_transform.axes': 'a parallel to x; b in the plane of y and z',
        '_atom_sites_cartn_transform.matrix': matrix
    }
    return atom_sites_dict

def add_cart_pos(
    atom_site_dict: Dict[str, List[float]],
    cell_dict: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convert fractional atomic positions to Cartesian coordinates based on the unit cell parameters.

    Parameters
    ----------
    atom_site_dict : Dict[str, List[float]]
        Dictionary containing fractional atomic positions as lists of floats
        in cif format. Needs to include: '_atom_site_fract_x',
        '_atom_site_fract_y', and '_atom_site_fract_z'.

    cell_dict : Dict[str, Any]
        Dictionary containing cell parameters in cif notation

    Returns
    -------
    atom_site_out : Dict[str, Any]
        The input dictionary with added Cartesian coordinates. These are added
        as lists of floats with keys '_atom_site_Cartn_x', '_atom_site_Cartn_y',
        and '_atom_site_Cartn_z'.

    atom_sites_dict : Dict[str, Any]
        Dictionary containing the transformation matrix for conversion from
        fractional to Cartesian coordinates and the cartesian convention in
        as cif keys.
    """
    atom_sites_dict = cell_dict2atom_sites_dict(cell_dict)
    xyz_fract = np.array(
        [atom_site_dict[f'_atom_site.fract_{val}'] for val in ('x', 'y', 'z')]
    ).T
    xyz_cartn = np.einsum(
        'xy, zy -> zx',
        atom_sites_dict['_atom_sites_cartn_transform.matrix'],
        xyz_fract
    )
    atom_site_out = atom_site_dict.copy()
    atom_site_out['_atom_site.cartn_x'] = list(xyz_cartn[:,0])
    atom_site_out['_atom_site.cartn_y'] = list(xyz_cartn[:,1])
    atom_site_out['_atom_site.cartn_z'] = list(xyz_cartn[:,2])
    return atom_site_out, atom_sites_dict

def position_difference(
    cif1_path: Path,
    cif1_dataset: Union[int, str],
    cif2_path: Path,
    cif2_dataset: Union[int, str]
):
    """
    Computes positional differences between datasets in two CIF files.

    Parameters
    ----------
    cif1_path : Path
        Path to the first CIF file.
    cif1_dataset : Union[int, str]
        Dataset index or name in the first CIF file.
    cif2_path : Path
        Path to the second CIF file.
    cif2_dataset : Union[int, str]
        Dataset index or name in the second CIF file.

    Returns
    -------
    Dict[str, float]
        A dictionary containing metrics such as 'max abs position', 'mean abs position',
        'max position/su', and 'mean position/su', reflecting the differences in atomic
        positions between the two datasets.
    """
    block1, _ = cifdata_str_or_index(read_cif_safe(cif1_path), cif1_dataset)
    block2, _ = cifdata_str_or_index(read_cif_safe(cif2_path), cif2_dataset)

    positions_sus1 = [split_sus(block1[f'_atom_site.fract_{xyz}']) for xyz in ('x', 'y', 'z')]
    atom_site_frac1 = {
        f'_atom_site.fract_{xyz}': vals[0] for xyz, vals in zip(('x', 'y', 'z'), positions_sus1)
    }
    frac1 = np.stack([val[0] for val in positions_sus1], axis=1)
    frac1_su = np.stack([val[1] for val in positions_sus1], axis=1)
    positions_sus2 = [split_sus(block2[f'_atom_site.fract_{xyz}']) for xyz in ('x', 'y', 'z')]
    atom_site_frac2 = {
        f'_atom_site.fract_{xyz}': vals[0] for xyz, vals in zip(('x', 'y', 'z'), positions_sus2)
    }
    frac2 = np.stack([val[0] for val in positions_sus2], axis=1)
    frac2_su = np.stack([val[1] for val in positions_sus2], axis=1)

    atom_site1, _ = add_cart_pos(atom_site_frac1, block1)
    atom_site2, _ = add_cart_pos(atom_site_frac2, block2)

    cart1 = np.stack((
        atom_site1['_atom_site.cartn_x'],
        atom_site1['_atom_site.cartn_y'],
        atom_site1['_atom_site.cartn_z'],
    ), axis=1)

    cart2 = np.stack((
        atom_site2['_atom_site.cartn_x'],
        atom_site2['_atom_site.cartn_y'],
        atom_site2['_atom_site.cartn_z'],
    ), axis=1)

    distances = np.linalg.norm(cart1 - cart2, axis=-1)

    abs_diff = np.abs(frac1 - frac2)
    sus_diff = (frac1_su**2 + frac2_su**2)**0.5

    return_dict = {
        'max abs position': np.max(distances),
        'mean abs position': np.mean(distances),
        'max position/su': np.max(abs_diff / sus_diff),
        'mean position/su': np.mean(abs_diff / sus_diff)
    }

    return return_dict

def anisotropic_adp_difference(
    cif1_path: Path,
    cif1_dataset: Union[int, str],
    cif2_path: Path,
    cif2_dataset: Union[int, str]
):
    """
    Calculates differences in anisotropic atomic displacement parameters (ADPs) between
    two CIF datasets.

    Parameters
    ----------
    cif1_path : Path
        Path to the first CIF file.
    cif1_dataset : Union[int, str]
        Dataset index or name in the first CIF file.
    cif2_path : Path
        Path to the second CIF file.
    cif2_dataset : Union[int, str]
        Dataset index or name in the second CIF file.

    Returns
    -------
    Dict[str, float]
        A dictionary with metrics like 'max abs uij', 'mean abs uij', 'max uij/su', and
        'mean uij/su', indicating the differences in ADPs between the datasets.
    """
    block1, _ = cifdata_str_or_index(read_cif_safe(cif1_path), cif1_dataset)
    block2, _ = cifdata_str_or_index(read_cif_safe(cif2_path), cif2_dataset)

    uij_sus1 = [split_sus(block1[f'_atom_site_aniso.u_{ij}']) for ij in (11, 22, 33, 12, 13, 23)]
    uij_sus2 = [split_sus(block2[f'_atom_site_aniso.u_{ij}']) for ij in (11, 22, 33, 12, 13, 23)]

    uij1 = np.stack([val[0] for val in uij_sus1], axis=1)
    uij1_su = np.stack([val[1] for val in uij_sus1], axis=1)

    uij2 = np.stack([val[0] for val in uij_sus2], axis=1)
    uij2_su = np.stack([val[1] for val in uij_sus2], axis=1)

    abs_diff = np.abs(uij1 - uij2)
    sus_diff = (uij1_su**2 + uij2_su**2)**0.5

    return_dict = {
        'max abs uij': np.max(abs_diff),
        'mean abs uij': np.mean(abs_diff),
        'max uij/su': np.max(abs_diff / sus_diff),
        'mean uij/su': np.mean(abs_diff / sus_diff)
    }

    return return_dict

def check_converged(
    cif1_path: Path,
    cif1_dataset: Union[int, str],
    cif2_path: Path,
    cif2_dataset: Union[int, str],
    criteria_dict: Dict[str, float]
):
    """
    Evaluates if the differences between two CIF datasets meet specified convergence criteria.

    This function computes various metrics to compare atomic positions and anisotropic
    atomic displacement parameters (ADPs) between two CIF datasets. It then checks
    these metrics against user-defined convergence criteria.

    Parameters
    ----------
    cif1_path : Path
        Path to the first CIF file.
    cif1_dataset : Union[int, str]
        Dataset index or name in the first CIF file.
    cif2_path : Path
        Path to the second CIF file.
    cif2_dataset : Union[int, str]
        Dataset index or name in the second CIF file.
    criteria_dict : Dict[str, float]
        A dictionary specifying the convergence criteria for each metric.
        Possible entries include:
        - 'max abs position': Maximum absolute position difference allowed in Angstrom.
        - 'mean abs position': Mean absolute position difference allowed in Angstrom.
        - 'max position/su': Maximum ratio of difference in position parameters to estimated
           standard uncertainty (su allowed.
        - 'mean position/su': Mean ratio of difference in position parameters to su allowed.
        - 'max abs uij': Maximum absolute difference in anisotropic ADPs allowed.
        - 'mean abs uij': Mean absolute difference in anisotropic ADPs allowed.
        - 'max uij/su': Maximum ratio of ADP difference to su allowed.
        - 'mean uij/su': Mean ratio of ADP difference to su allowed.

    Returns
    -------
    bool
        Returns True if all the evaluated metrics meet the convergence criteria, otherwise False.
    """
    diff_dict = position_difference(
        cif1_path,
        cif1_dataset,
        cif2_path,
        cif2_dataset
    )
    diff_dict.update(anisotropic_adp_difference(
        cif1_path,
        cif1_dataset,
        cif2_path,
        cif2_dataset
    ))
    messages = []
    converged = True
    for name, criterion in criteria_dict.items():
        comparison = diff_dict[name]
        if comparison <= criterion:
            messages.append(f'{name} is converged ({comparison}, <= {criterion})')
        else:
            messages.append(f'{name} is not converged ({comparison}, > {criterion})')
            converged = False
    return converged
