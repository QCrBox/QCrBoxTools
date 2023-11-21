from pathlib import Path
from typing import Union, Dict, List, Any, Tuple
import numpy as np
from .cif import read_cif_safe, cifdata_str_or_index, split_esds, split_esd_single

def cell_dict2atom_sites_dict(
    cell_dict: Dict[str, Union[float, np.ndarray]]
) -> Dict[str, Union[str, np.ndarray]]:
    """
    Converts a cell dictionary into an atom sites dictionary that includes transformation matrix.

    The transformation matrix is generated based on the cell parameters. This matrix contains the three lattice vectors
    as rows. It is used to convert from fractional to Cartesian coordinates.

    Args:
        cell_dict (Dict[str, Union[float, np.ndarray]]): A dictionary representing a unit cell. It should contain the
        following keys: '_cell_length_a', '_cell_length_b', '_cell_length_c', '_cell_angle_alpha', '_cell_angle_beta',
        '_cell_angle_gamma'. The corresponding values should be floats representing the cell lengths (a, b, c) in angstroms
        and the cell angles (alpha, beta, gamma) in degrees.

    Returns:
        Dict[str, Union[str, np.ndarray]]: A dictionary containing the transformation matrix and its description. The keys
        are '_atom_sites_Cartn_transform_axes' (with value being a string description of the transformation axes) and
        '_atom_sites_Cartn_tran_matrix' (with value being a 3x3 numpy array representing the transformation matrix).
    """
    a = split_esd_single(cell_dict['_cell_length_a'])[0]
    b = split_esd_single(cell_dict['_cell_length_b'])[0]
    c = split_esd_single(cell_dict['_cell_length_c'])[0]
    alpha = split_esd_single(cell_dict['_cell_angle_alpha'])[0] / 180.0 * np.pi
    beta = split_esd_single(cell_dict['_cell_angle_beta'])[0] / 180.0 * np.pi
    gamma = split_esd_single(cell_dict['_cell_angle_gamma'])[0] / 180.0 * np.pi
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
        '_atom_sites_Cartn_transform_axes': 'a parallel to x; b in the plane of y and z',
        '_atom_sites_Cartn_tran_matrix': matrix
    }
    return atom_sites_dict

def add_cart_pos(atom_site_dict: Dict[str, List[float]], cell_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
    xyz_fract = np.array([atom_site_dict[f'_atom_site_fract_{val}'] for val in ('x', 'y', 'z')]).T
    xyz_cartn = np.einsum('xy, zy -> zx', atom_sites_dict['_atom_sites_Cartn_tran_matrix'], xyz_fract)
    atom_site_out = atom_site_dict.copy()
    atom_site_out['_atom_site_Cartn_x'] = list(xyz_cartn[:,0])
    atom_site_out['_atom_site_Cartn_y'] = list(xyz_cartn[:,1])
    atom_site_out['_atom_site_Cartn_z'] = list(xyz_cartn[:,2])
    return atom_site_out, atom_sites_dict

def position_difference(
    cif1_path: Path,
    cif1_dataset: Union[int, str],
    cif2_path: Path,
    cif2_dataset: Union[int, str]
):
    block1, _ = cifdata_str_or_index(read_cif_safe(cif1_path), cif1_dataset)
    block2, _ = cifdata_str_or_index(read_cif_safe(cif2_path), cif2_dataset)

    positions_esds1 = [split_esds(block1[f'_atom_site_fract_{xyz}']) for xyz in ('x', 'y', 'z')]
    atom_site_frac1 = {f'_atom_site_fract_{xyz}': vals[0] for xyz, vals in zip(('x', 'y', 'z'), positions_esds1)}
    frac1 = np.stack([val[0] for val in positions_esds1], axis=1)
    frac1_esd = np.stack([val[1] for val in positions_esds1], axis=1)
    positions_esds2 = [split_esds(block2[f'_atom_site_fract_{xyz}']) for xyz in ('x', 'y', 'z')]
    atom_site_frac2 = {f'_atom_site_fract_{xyz}': vals[0] for xyz, vals in zip(('x', 'y', 'z'), positions_esds2)}
    frac2 = np.stack([val[0] for val in positions_esds2], axis=1)
    frac2_esd = np.stack([val[1] for val in positions_esds2], axis=1)

    atom_site1, _ = add_cart_pos(atom_site_frac1, block1)
    atom_site2, _ = add_cart_pos(atom_site_frac2, block2)

    cart1 = np.stack((
        atom_site1['_atom_site_Cartn_x'],
        atom_site1['_atom_site_Cartn_y'],
        atom_site1['_atom_site_Cartn_z'],
    ), axis=1)

    cart2 = np.stack((
        atom_site2['_atom_site_Cartn_x'],
        atom_site2['_atom_site_Cartn_y'],
        atom_site2['_atom_site_Cartn_z'],
    ), axis=1)

    distances = np.linalg.norm(cart1 - cart2, axis=-1)

    abs_diff = np.abs(frac1 - frac2)
    esds_diff = (frac1_esd**2 + frac2_esd**2)**0.5

    return_dict = {
        'max abs position': np.max(distances),
        'mean abs position': np.mean(distances),
        'max position/esd': np.max(abs_diff / esds_diff),
        'mean position/esd': np.mean(abs_diff / esds_diff)
    }

    return return_dict

