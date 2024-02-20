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
from itertools import product
from pathlib import Path
import re
from collections import defaultdict
from typing import Union, Tuple, Iterable, Optional, List, Dict

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


def cifdata_str_or_index(model: cif.model.cif, dataset: Union[int, str]) -> cif.model.block:
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
    if dataset in model.keys():
        return model[dataset], dataset
    try:
        dataset_index = int(dataset)
        keys = list(model.keys())
        dataset = keys[dataset_index]
        return model[dataset], dataset
    except ValueError as exc:
        raise ValueError(
            'Dataset does not exist in cif and cannot be cast into int as index. '
            + f'Got: {dataset}'
        ) from exc
    except IndexError as exc:
        raise IndexError(
            'The given dataset does not exists and integer index is out of range'
            + f'. Got: {dataset}'
        ) from exc


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


def del_atom_site_condition(key: str) -> bool:
    """Check if to be deleted because an atom_site entry (these are replaced)"""
    return key.startswith('_atom_site')


def del_geom_condition(key: str) -> bool:
    """Check if to be deleted because an geom entry (these are based on outdate info)"""
    return key.startswith('_geom')


def del_refine_condition(key: str) -> bool:
    """Check if to be deleted because an refine entry (these combined cif is not converged)"""
    exceptions = ('_refine_ls_weighting', '_refine_ls_extinction')
    return key.startswith('_refine') and not key.startswith(exceptions)


def del_refln_condition(key: str) -> bool:
    """Check if to be deleted because a refln entry with calc (these are based on outdated info)"""
    return key.startswith('_refln') and 'calc' in key


def del_refine_file(key: str) -> bool:
    """Check if to be deleted because if containes the res/ins file (is replaced)"""
    test_keys = (
        '_shelx.res_file',  # shelxl
        '_iucr.refine_instructions_details'  # olex2
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

    # delete loops
    keys = tuple(new_block.loops.keys())
    for key in keys:
        if any(condition(key) for condition in conditions):
            del new_block[key]

    # delete items
    for key in new_block.keys():
        if any(condition(key) for condition in conditions):
            del new_block[key]

    copy_loops = ('_atom_site', '_atom_site_aniso')

    for loop_key in copy_loops:
        loop = structure_cif_block[loop_key]

        for key, vals in loop.items():
            num_su_column = all(map(is_num_su, vals))
            brackets_column = any('(' in val for val in vals)
            if num_su_column and brackets_column:
                values, _ = split_sus(vals)
                for i, new_val in enumerate(f'{val}' for val in values):
                    loop[key][i] = new_val

        new_block.add_loop(loop)

    keys = [key for key in structure_cif_block if key.startswith('_atom_sites')]

    add_if_present = (
        '_shelx.res_file',
        '_iucr.refine_instructions_details'
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
        '_space_group.name_h-m_alt',
        '_space_group.name_hall'
    )

    for entry in space_group_entries:
        if entry in block1:
            block1_centrings.append(block1[entry].replace('-', '')[0].upper())
        if entry in block2:
            block2_centrings.append(block2[entry].replace('-', '')[0].upper())

    if len(block1_centrings) == 0:
        raise NoCentringFoundError('No _space_group.name_h-m_alt/hall found in cif1')
    if len(block2_centrings) == 0:
        raise NoCentringFoundError('No _space_group.name_h-m_alt/hall found in cif2')
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
    return block1['_space_group.crystal_system'] == block2['_space_group.crystal_system']


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
    atom_site_labels = list(block['_atom_site.label'])

    # Get selected atoms
    if select_names is None:
        select_names = []

    if select_elements is not None:
        select_names += [
            name for name, element in zip(atom_site_labels, block['_atom_site.type_symbol'])
            if element in select_elements
        ]

    if select_regexes is not None:
        for regex in select_regexes:
            select_names += [
                name for name in atom_site_labels if re.match(regex, name) is not None
            ]

    select_names = list(set(select_names))

    # if overwrite False remove preexistring
    existing = list(block['_atom_site_aniso.label'])
    if not overwrite:
        select_names = [name for name in select_names if name not in existing]

    # calculate values and set adp type
    new_values = {}
    for name in select_names:
        uiso_index = atom_site_labels.index(name)
        uiso = split_su_single(block['_atom_site.u_iso_or_equiv'][uiso_index])[0]
        new_values[name] = single_value_iso2aniso(
            uiso,
            split_su_single(block['_cell.angle_alpha'])[0],
            split_su_single(block['_cell.angle_beta'])[0],
            split_su_single(block['_cell.angle_gamma'])[0]
        )
        block['_atom_site.adp_type'][uiso_index] = 'Uani'

    # build up new atom_site_aniso arrays
    loop = block['_atom_site_aniso']
    new_aniso_labels = list(sorted(existing + select_names, key=atom_site_labels.index))

    for _ in range(len(new_aniso_labels) - loop.n_rows()):
        loop.add_row(['?'] * loop.n_columns())
    loop.update_column('_atom_site_aniso.label', new_aniso_labels)
    for ij_index, ij in enumerate((11, 22, 33, 12, 13, 23)):
        aniso_key = f'_atom_site_aniso.u_{ij}'
        loop.update_column(f'_atom_site_aniso.u_{ij}', [
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

class NonMatchingMergeKeys(Exception):
    """
    Exception raised when the keys intended for merging in CIF loops do not match.
    """

class NonExistingMergeKey(Exception):
    """
    Exception raised when no matching keys are found for merging in CIF loops.
    """

def loop_to_row_dict(
    loop: cif.model.loop,
    merge_keys: Tuple[str]
) -> Dict[Tuple, Dict[str, str]]:
    """
    Convert a CIF loop to a dictionary with keys as tuples of merge key values and
    values as dictionaries of non-merge key-value pairs.

    Parameters
    ----------
    loop : cif.model.loop
        The CIF loop to be converted.
    merge_keys : Tuple[str]
        A tuple of strings representing the keys on which to merge.

    Returns
    -------
    Dict[Tuple, Dict[str, str]]
        A dictionary where each key is a tuple of values from the merge keys, and
        each value is a dictionary of non-merge key-value pairs, with default values
        as '?' for missing entries.

    Notes
    -----
    This function is a helper intended to facilitate merging CIF loops based on
    specified keys. It organizes loop data for easy comparison and merging.
    """
    keys = (tuple(vals) for vals in zip(*(loop[key] for key in merge_keys)))
    nonmerge_keys = [key for key in loop.keys() if key not in merge_keys]

    values = [
        defaultdict(
            lambda: '?',
            [(key, val) for key, val in zip(nonmerge_keys, row_vals)]
        )
        for row_vals in zip(*(loop[key] for key in nonmerge_keys))
    ]
    return dict(zip(keys, values))

def merge_cif_loops(
    loop1: cif.model.loop,
    loop2: cif.model.loop,
    merge_on: Union[str, List[str]]=r'*\.label'
) -> cif.model.loop:
    """
    Merge two CIF loops based on matching keys specified by `merge_on`.

    Parameters
    ----------
    loop1 : cif.model.loop
        The first CIF loop to merge.
    loop2 : cif.model.loop
        The second CIF loop to merge.
    merge_on : Union[str, List[str]], optional
        A string or list of strings representing the regex patterns of keys on
        which the loops should be merged. Defaults to r'*\\.label' to match any
        key ending with '.label'.

    Returns
    -------
    cif.model.loop
        A new CIF loop that is the result of merging `loop1` and `loop2` based
        on the specified `merge_on` keys. Missing values are filles with '?'.

    Raises
    ------
    NonExistingMergeKey
        If no keys match the `merge_on` pattern in either loop.
    NonMatchingMergeKeys
        If the keys found to merge on do not have a one-to-one correspondence
        between the two loops.
    """
    if isinstance(merge_on, str):
        merge_on = [merge_on]

    merge_keys = [
        key for key in loop1.keys()
        if any(re.match(pattern, key) is not None for pattern in merge_on)
    ]
    merge_keys_check = [
        key for key in loop2.keys()
        if any(re.match(pattern, key) is not None for pattern in merge_on)
    ]

    keys_identical = all(
        key1 == key2 for key1, key2 in zip(sorted(merge_keys), sorted(merge_keys_check))
    )

    if len(merge_keys) == 0 and len(merge_keys_check) == 0:
        raise NonExistingMergeKey('No matching keys found for merging.')

    equal_number_keys = len(merge_keys) == len(merge_keys_check)
    if not (keys_identical and equal_number_keys):
        raise NonMatchingMergeKeys(
            ('Found keys for matching loop are not identical. '
             + f'loop1:{sorted(merge_keys)} loop2:{sorted(merge_keys_check)}'
            )
        )

    start_dict = loop_to_row_dict(loop1, merge_keys)
    add_dict = loop_to_row_dict(loop2, merge_keys)

    for merge_key, inner_dict in add_dict.items():
        if merge_key not in start_dict:
            start_dict[merge_key] = inner_dict
        else:
            start_dict[merge_key].update(inner_dict)

    new_dict = {key: val for key, val in zip(merge_keys, zip(*start_dict.keys()))}
    nonmerge_keys = set(
        [key for key in loop1.keys() if key not in merge_keys]
        + [key for key in loop2.keys() if key not in merge_keys]
    )

    nonmerge_columns = zip(*[[row[key] for key in nonmerge_keys] for row in start_dict.values()])
    non_merge_dict = {key: val for key, val in zip(nonmerge_keys, nonmerge_columns)}
    new_dict.update(non_merge_dict)

    return cif.model.loop(data=new_dict)

class NonUniqueBlockMerging(Exception):
    """
    Exception raised when a CIF block's loops are attempted to be merged more
    than once with different loops.
    """

MERGE_CIF_DEFAULT = 1

def merge_cif_blocks(
    block1: cif.model.block,
    block2: cif.model.block,
    possible_markers: Union[int, List[str], str] = MERGE_CIF_DEFAULT
) -> cif.model.block:
    """
    Merge two CIF blocks by attempting to merge their loops based on specified key patterns.

    This function iterates over loops within each block, merging them based on regex patterns
    defined in `possible_markers`. It aims to create a unified block from two sources, handling
    key mismatches or overlaps as specified.

    Parameters
    ----------
    block1 : cif.model.block
        The first CIF block to merge.
    block2 : cif.model.block
        The second CIF block to merge. Values take precedence over block1.
    possible_markers : Union[int, List[str], str], optional
        A list of regex patterns, a single regex pattern for keys to merge loops on, or an integer
        flag to use a default set of markers. Defaults to MERGE_CIF_DEFAULT, applying a default
        set of patterns.

    Returns
    -------
    cif.model.block
        A new CIF block that results from merging `block1` and `block2`.

    Raises
    ------
    NonUniqueBlockMerging
        If a loop within either block is merged more than once with different loops, indicating
        a conflict in merging criteria or an attempt to merge non-unique loops.
    """

    if possible_markers == MERGE_CIF_DEFAULT:
        possible_markers = [r'.*\.id', r'.*.label_?\d*', r'.*_refln\.index.*', r'_atom_type\.symbol']
    if isinstance(possible_markers, str):
        possible_markers = [possible_markers]
    # merge blocks
    used_loops1 = []
    used_loops2 = []
    new_loops = {}
    entry2loop_name = {}
    iter_product = product(block1.loops.items(), block2.loops.items(), possible_markers)
    for (loop1_name, loop1), (loop2_name, loop2), marker in iter_product:
        try:
            merged_loop = merge_cif_loops(loop1, loop2, merge_on=marker)
            if loop1_name in used_loops1:
                raise NonUniqueBlockMerging(
                    (f'loop1: {loop1_name} merged at least twice. '
                     + f'Second merge with {loop2_name} of block2')
                )
            if loop2_name in used_loops2:
                raise NonUniqueBlockMerging(
                    (f'loop2: {loop2_name} merged at least twice. '
                     + f'Second merge with {loop1_name} of block1')
                )
            used_loops1.append(loop1_name)
            used_loops2.append(loop2_name)
            entry2loop_name.update({entry: merged_loop.name for entry in merged_loop.keys()})
            new_loops[merged_loop.name] = merged_loop
        except (NonExistingMergeKey, NonMatchingMergeKeys):
            pass
    for block, used_loops in zip((block1, block2), (used_loops1, used_loops2)):
        missing_loops = [name for name in block.loops.keys() if name not in used_loops]
        for name in missing_loops:
            new_loops[name] = block.loops[name]
            entry2loop_name.update({entry: name for entry in block.loops[name].keys()})

    merged_block = cif.model.block()

    for entry in block1.keys():
        if entry in entry2loop_name:
            new_loop = new_loops[entry2loop_name[entry]]
            if new_loop is not None:
                merged_block.add_loop(new_loop)
                new_loops[entry2loop_name[entry]] = None
        else:
            merged_block.add_data_item(entry, block1[entry])

    for entry in block2.keys():
        if entry in entry2loop_name:
            new_loop = new_loops[entry2loop_name[entry]]
            if new_loop is not None:
                merged_block.add_loop(new_loop)
                new_loops[entry2loop_name[entry]] = None
        else:
            merged_block.add_data_item(entry, block2[entry])

    return merged_block

def merge_cif_files(
    cif_path: Union[str, Path],
    block_name: str,
    cif_path2: Union[str, Path],
    block_name2: str,
    output_path: Union[str, Path],
    output_block_name: str,
    possible_markers: Union[int, List[str], str]=MERGE_CIF_DEFAULT
):
    """
    Merges two specified blocks from two Crystallographic Information File (CIF)
    paths into a single block, and writes the result to a new CIF file.

    This function reads two CIF files, extracts specified blocks from each,
    merges these blocks based on common indices or specified markers, and
    writes the merged block to a new CIF file with a specified block name.
    The identification and extraction of blocks are flexible, allowing by name
    or by index if the block name is not found. The merging process respects
    specified markers to align columns and rows from two different blocks
    of CIF data. The output is a new CIF file with the merged block.

    Parameters
    ----------
    cif_path : Union[str, Path]
        Path to the first CIF file.
    block_name : str
        Name or index of the block to extract from the first CIF file.
        If block_name is not present in the CIF file, the function will
        attempt to call it into an integer to select by index.
    cif_path2 : Union[str, Path]
        Path to the second CIF file.
    block_name2 : str
        Name or index of the block to extract from the second CIF file.
        If block_name2 is not present in the CIF file, the function will
        attempt to call it into an integer to select by index.
    output_path : Union[str, Path]
        Path for the output CIF file containing the merged block.
    output_block_name : str
        Name for the block in the output CIF file.
    possible_markers : Union[int, List[str], str], optional
        Regex marker(s) to use for merging the blocks. If not specified,
        uses a default set of markers or an index.
    """
    model1 = read_cif_safe(cif_path)
    if block_name in model1:
        block1 = model1[block_name]
    else:
        block1, _ = cifdata_str_or_index(model1, int(block_name))

    model2 = read_cif_safe(cif_path2)
    if block_name2 in model2:
        block2 = model2[block_name2]
    else:
        block2, _ = cifdata_str_or_index(model2, int(block_name2))

    output_block = merge_cif_blocks(block1, block2, possible_markers)

    output_cif = cif.model.cif()
    output_cif[output_block_name] = output_block

    Path(output_path).write_text(str(output_cif), encoding='UTF-8')
