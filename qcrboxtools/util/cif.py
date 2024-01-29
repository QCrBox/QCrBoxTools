# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

"""
This module provides utility functions for handling CIF (Crystallographic Information File) files.
It includes functionalities such as:
- Safely reading CIF files.
- Retrieving CIF datasets based on an index or identifier.
- Extracting numerical values and estimated standard uncertainties (sus) from formatted CIF strings.
- Replacing the structure block of a CIF file with another.
"""

from pathlib import Path
import re
from typing import Union, Tuple, Iterable, Optional, List

from iotbx import cif
import numpy as np

class NoCentringFoundError(Exception):
    """
    Exception raised when no centring information is found in a CIF file.
    """

class InConsistentCentringError(Exception):
    """
    Exception raised for inconsistent centring information in CIF files.
    """


def read_cif_safe(cif_path: Union[str, Path]) -> cif.model.cif:
    """
    Read a CIF file and return its content as a CIF model.
    Also works with Pathlib paths

    Parameters
    ----------
    cif_path : Union[str, Path]
        The path to the CIF file.

    Returns
    -------
    cif.model.cif
        The CIF model parsed from the file.
    """
    with open(cif_path, 'r', encoding='UTF-8') as fobj:
        cif_content = fobj.read()

    return cif.reader(input_string=cif_content).model()

def cifdata_str_or_index(model: cif.model.cif, dataset: [int, str]) -> cif.model.block:
    """
    Retrieve a CIF dataset block from the model using an index or identifier.

    Parameters
    ----------
    model : iotbx.cif.model.cif
        The CIF model containing datasets.
    dataset : Union[int, str]
        Index or string identifier for the dataset.

    Returns
    -------
    Tuple[cif.model.block, str]
        The CIF dataset block and its identifier.
    """
    if isinstance(dataset, int):
        keys = list(model.keys())
        dataset = keys[dataset]
    return model[dataset], dataset

def split_su_single(input_string: str) -> Tuple[float, float]:
    """
    Extract the value and standard uncertainty from a CIF formatted string.

    Parameters
    ----------
    input_string : str
        String containing a numeric value and possibly an SU.

    Returns
    -------
    Tuple[float, float]
        The numeric value and its standard uncertainty.
    """
    input_string = str(input_string)

    if not is_num_su(input_string):
        raise ValueError(f'{input_string} is not a valid string to split into value(su)')
    su_pattern = r'([^\.]+)\.?(.*?)\(([\d\.]+)\)'
    match = re.match(su_pattern, input_string)
    if match is None:
        return float(input_string), np.nan
    if len(match.group(2)) == 0:
        return float(match.group(1)), float(match.group(3))
    magnitude = 10.0**(-len(match.group(2)))
    if match.group(1).startswith('-'):
        sign = -1
    else:
        sign = 1
    # append the strings to reduce floating point errors (do not use magnitude)
    value = float(match.group(1)) + sign * float('0.' + match.group(2))
    su = magnitude * float(match.group(3).replace('.', ''))
    return value, su

def split_sus(input_strings: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract values and standard uncertainties from a list of formatted strings.

    Parameters
    ----------
    input_strings : Iterable[str]
        A list of input strings, each containing a numeric value and possibly an SU.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays of numeric values and their associated standard uncertainties.
    """
    values, sus = zip(*map(split_su_single, input_strings))
    return np.array(list(values)), np.array(list(sus))

def del_atom_site_condition(key:str) -> bool:
    """Check if to be deleted because an atom_site entry (these are replaced)"""
    return key.startswith('_atom_site')

def del_geom_condition(key:str) -> bool:
    """Check if to be deleted because an geom entry (these are based on outdate info)"""
    return key.startswith('_geom')

def del_refine_condition(key:str) -> bool:
    """Check if to be deleted because an refine entry (these combined cif is not converged)"""
    exceptions = ('_refine_ls_weighting','_refine_ls_extinction')
    return key.startswith('_refine') and not key.startswith(exceptions)

def del_refln_condition(key:str) -> bool:
    """Check if to be deleted because a refln entry with calc (these are based on outdated info)"""
    return key.startswith('_refln') and '_calc' in key

def del_refine_file(key:str) -> bool:
    """Check if to be deleted because if containes the res/ins file (is replaced)"""
    test_keys = (
        '_shelx_res_file', # shelxl
        '_iucr_refine_instructions_details' # olex2
    )
    return any(key == test_key for test_key in test_keys)

def is_num_su(string: str) -> bool:
    """
    Check if a string is compatible with numerical values and standard uncertainties.

    Parameters
    ----------
    string : str
        The string to be checked.

    Returns
    -------
    bool
        True if the string only contains characters valid for numerical values
        and standard uncertainties, False otherwise.
    """
    return re.search(r'[^\d\.\-\+\(\)]', string) is None

def replace_structure_from_cif(
    cif_path: Union[str, Path],
    cif_dataset: str,
    structure_cif_path: Union[str, Path],
    structure_cif_dataset: str,
    output_cif_path: Union[str, Path]
) -> None:
    """
    Replace the structural information in a CIF file with that from another CIF file.

    Parameters
    ----------
    cif_path : Union[str, Path]
        Path to the original CIF file.
    cif_dataset : str
        Dataset name in the original CIF to be replaced.
    structure_cif_path : Union[str, Path]
        Path to the CIF file with the replacement structure.
    structure_cif_dataset : str
        Dataset name in the replacement CIF file.
    output_cif_path : Union[str, Path]
        Path to save the modified CIF file.

    Returns
    -------
    None
    """
    cif_obj = read_cif_safe(cif_path=cif_path)
    structure_cif_obj = read_cif_safe(cif_path=structure_cif_path)

    new_cif_obj = cif_obj.copy()

    new_block, cif_dataset = cifdata_str_or_index(
        cif_obj,
        cif_dataset
    )

    structure_cif_block, structure_cif_dataset = cifdata_str_or_index(
        structure_cif_obj,
        structure_cif_dataset
    )

    conditions = (
        del_atom_site_condition,
        del_geom_condition,
        del_refine_condition,
        del_refln_condition,
        del_refine_file
    )

    #delete loops
    keys = tuple(new_block.loops.keys())
    for key in keys:
        if any(condition(key) for condition in conditions):
            del new_block[key]

    #delete items
    for key in new_block.keys():
        if any(condition(key) for condition in conditions):
            del new_block[key]

    copy_loops = ('_atom_site', '_atom_site_aniso')

    for loop_key in copy_loops:
        loop = structure_cif_block[loop_key]

        for key, vals in loop.items():
            num_su_column = all(map(is_num_su, vals))
            brackets_column = any('(' in val  for val in vals)
            if num_su_column and brackets_column:
                values, _ = split_sus(vals)
                for i, new_val in enumerate(f'{val}' for val in values):
                    loop[key][i] = new_val

        new_block.add_loop(loop)

    keys = [key for key in structure_cif_block if key.startswith('_atom_sites')]

    add_if_present = (
        '_shelx_res_file',
        '_iucr_refine_instructions_details'
    )
    for key in add_if_present:
        if key in structure_cif_block:
            keys.append(key)

    for key in keys:
        new_block.add_data_item(key, structure_cif_block[key])

    new_cif_obj[cif_dataset] = new_block

    with open(output_cif_path, 'w', encoding='UTF-8') as fobj:
        fobj.write(str(new_cif_obj))

def check_centring_equal(
    block1: cif.model.block,
    block2: cif.model.block
) -> bool:
    """
    Check if the lattice centring is the same in two CIF blocks.

    Parameters
    ----------
    block1 : cif.model.block
        The first CIF block.
    block2 : cif.model.block
        The second CIF block.

    Returns
    -------
    bool
        True if lattice centring is the same, False otherwise.

    Raises
    ------
    NoCentringFoundError
        If no centring information is found in either block.
    InConsistentCentringError
        If centring information is inconsistent within a block.
    """
    block1_centrings = []
    block2_centrings = []

    space_group_entries = (
        '_space_group_name_H-M_alt',
        '_space_group_name_Hall',
        '_symmetry_space_group_name_H-M',
        '_symmetry_space_group_name_Hall'
    )

    for entry in space_group_entries:
        if entry in block1:
            block1_centrings.append(block1[entry].replace('-', '')[0].upper())
        if entry in block2:
            block2_centrings.append(block2[entry].replace('-', '')[0].upper())

    if len(block1_centrings) == 0:
        raise NoCentringFoundError('No _space_group_name_H-M_alt/Hall found in cif1')
    if len(block2_centrings) == 0:
        raise NoCentringFoundError('No _space_group_name_H-M_alt/Hall found in cif2')
    if any(check_centring != block1_centrings[0] for check_centring in block1_centrings[1:]):
        raise InConsistentCentringError('Centrings from entries do not agree for cif1')
    if any(check_centring != block2_centrings[0] for check_centring in block2_centrings[1:]):
        raise InConsistentCentringError('Centrings from entries do not agree for cif2')
    return block1_centrings[0] == block2_centrings[0]

def check_crystal_system(block1: cif.model.block, block2: cif.model.block) -> bool:
    """
    Check if two CIF blocks belong to the same crystal system.

    Parameters
    ----------
    block1 : cif.model.block
        The first CIF block.
    block2 : cif.model.block
        The second CIF block.

    Returns
    -------
    bool
        True if both blocks belong to the same crystal system, False otherwise.
    """
    return block1['_space_group_crystal_system'] == block2['_space_group_crystal_system']

def cif_iso2aniso(
    input_cif_path: Union[str, Path],
    cif_dataset: Union[str, int],
    output_cif_path: Union[str, Path],
    select_names: Optional[List[str]] = None,
    select_elements: Optional[List[str]] = None,
    select_regexes: Optional[List[re.Pattern]] = None,
    overwrite: bool = False
) -> None:
    """
    Convert isotropic displacement parameters to anisotropic in a CIF file. Atoms can be
    selected by a list of names, elements or regexes by using the three keyword arguments.
    Already anisotropic atoms are not replaced. This behaviour can be changed by overwrite.

    Parameters
    ----------
    input_cif_path : Union[str, Path]
        Path to the input CIF file.
    cif_dataset : Union[str, int]
        Dataset name in the CIF file if string or index of dataset in file if int
    output_cif_path : Union[str, Path]
        Path to save the modified CIF file.
    select_names : Optional[List[str]], optional
        Specific atom names to convert, by default None.
    select_elements : Optional[List[str]], optional
        Specific elements to convert, by default None.
    select_regexes : Optional[List[re.Pattern]], optional
        Python re regex patterns to match atom names for conversion, by default None.
    overwrite : bool, optional
        Overwrite existing anisotropic parameters if True, by default False.

    Returns
    -------
    None
    """
    cif_content = read_cif_safe(input_cif_path)
    block, block_name = cifdata_str_or_index(cif_content, cif_dataset)
    atom_site_labels = list(block['_atom_site_label'])

    # Get selected atoms
    if select_names is None:
        select_names = []

    if select_elements is not None:
        select_names += [
            name for name, element in zip(atom_site_labels, block['_atom_site_type_symbol'])
            if element in select_elements
        ]

    if select_regexes is not None:
        for regex in select_regexes:
            select_names += [
                name for name in atom_site_labels if re.match(regex, name) is not None
            ]

    select_names = list(set(select_names))

    # if overwrite False remove preexistring
    existing = list(block['_atom_site_aniso_label'])
    if not overwrite:
        select_names = [name for name in select_names if name not in existing]

    # calculate values and set adp type
    new_values = {}
    for name in select_names:
        uiso_index = atom_site_labels.index(name)
        uiso = split_su_single(block['_atom_site_U_iso_or_equiv'][uiso_index])[0]
        new_values[name] = single_value_iso2aniso(
            uiso,
            split_su_single(block['_cell_angle_alpha'])[0],
            split_su_single(block['_cell_angle_beta'])[0],
            split_su_single(block['_cell_angle_gamma'])[0]
        )
        block['_atom_site_adp_type'][uiso_index] = 'Uani'

    # build up new atom_site_aniso arrays
    loop = block['_atom_site_aniso']
    new_aniso_labels = list(sorted(existing + select_names, key=atom_site_labels.index))

    for _ in range(len(new_aniso_labels) - loop.n_rows()):
        loop.add_row(['?'] * loop.n_columns())
    loop.update_column('_atom_site_aniso_label', new_aniso_labels)
    for ij_index, ij in enumerate((11, 22, 33, 12, 13, 23)):
        aniso_key = f'_atom_site_aniso_U_{ij}'
        loop.update_column(f'_atom_site_aniso_U_{ij}', [
            f'{new_values[name][ij_index]:8.8f}'
            if name in select_names else block[aniso_key][existing.index(name)]
            for name in new_aniso_labels
        ])


    cif_content[block_name] = block

    with open(output_cif_path, 'w', encoding='UTF-8') as fobj:
        fobj.write(str(cif_content))

def calc_rec_angle_cosines(
    alpha: float,
    beta: float,
    gamma: float
) -> Tuple[float, float, float]:
    """
    Calculate the reciprocal angles from given crystal angles.

    Parameters
    ----------
    alpha : float
        The alpha angle of the crystal (in degrees).
    beta : float
        The beta angle of the crystal (in degrees).
    gamma : float
        The gamma angle of the crystal (in degrees).

    Returns
    -------
    Tuple[float, float, float]
        The cosine of the reciprocal angles alpha*, beta*, and gamma*.
    """
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    cos_alpha_star = ((np.cos(beta_rad) * np.cos(gamma_rad) - np.cos(alpha_rad))
                      / (np.sin(beta_rad) * np.sin(gamma_rad)))
    cos_beta_star = ((np.cos(alpha_rad) * np.cos(gamma_rad) - np.cos(beta_rad))
                     / (np.sin(alpha_rad) * np.sin(gamma_rad)))
    cos_gamma_star = ((np.cos(alpha_rad) * np.cos(beta_rad) - np.cos(gamma_rad))
                      / (np.sin(alpha_rad) * np.sin(beta_rad)))

    return cos_alpha_star, cos_beta_star, cos_gamma_star

def single_value_iso2aniso(
    uiso: float,
    alpha: float,
    beta: float,
    gamma: float
) -> Tuple[float, float, float, float, float, float]:
    """
    Convert a single isotropic U value to anisotropic U values.

    Parameters
    ----------
    uiso : float
        The isotropic U value.
    alpha : float
        The alpha angle of the crystal (in degrees).
    beta : float
        The beta angle of the crystal (in degrees).
    gamma : float
        The gamma angle of the crystal (in degrees).

    Returns
    -------
    Tuple[float, float, float, float, float, float]
        The anisotropic U values (U11, U22, U33, U12, U13, U23).
    """
    cos_alpha_star, cos_beta_star, cos_gamma_star = calc_rec_angle_cosines(alpha, beta, gamma)

    u12 = uiso * cos_gamma_star
    u13 = uiso * cos_beta_star
    u23 = uiso * cos_alpha_star

    return uiso, uiso, uiso, u12, u13, u23
