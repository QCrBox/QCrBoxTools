from pathlib import Path
from iotbx import cif
from typing import Iterable
import re
import numpy as np

def read_cif_safe(cif_path):
    with open(cif_path, 'r', encoding='UTF-8') as fobj:
        cif_content = fobj.read()

    return cif.reader(input_string=cif_content).model()

def cifdata_str_or_index(model: dict, dataset: [int, str]) -> cif.model.block:
    """
    Retrieve CIF dataset block from the model based on an index or string identifier.

    Parameters:
    - model (dict): The CIF model containing datasets.
    - dataset (int or str): Index or string identifier for the desired dataset.

    Returns:
    - dict: The selected CIF dataset.
    """
    if isinstance(dataset, int):
        keys = list(model.keys())
        dataset = keys[dataset]
    return model[dataset]

def split_esd_single(input_string: str):
    PATTERN = r'([^\.]+)\.?(.*?)\(([\d\.]+)\)'
    match = re.match(PATTERN, input_string)
    if match is None:
        return float(input_string), np.nan
    if len(match.group(2)) == 0:
        return float(match.group(1)), float(match.group(3))
    magnitude = 10.0**(-len(match.group(2)))
    if match.group(1).startswith('-'):
        sign = -1
    else:
        sign = 1
    # append the strings to reduce floating point errors
    value = float(match.group(1)) + sign * float('0.' + match.group(2))
    esd = magnitude * float(match.group(3).replace('.', ''))
    return value, esd

def split_esds(input_strings: Iterable):
    values, esds = zip(*map(split_esd_single, input_strings))
    return np.array(list(values)), np.array(list(esds))

def del_atom_site_condition(key):
    return key.startswith('_atom_site')

def del_geom_condition(key):
    return key.startswith('_geom')

def del_refine_condition(key):
    exceptions = ('_refine_ls_weighting','_refine_ls_extinction')
    return key.startswith('_refine') and not key.startswith(exceptions)

def del_refln_condition(key):
    return key.startswith('_refln') and '_calc' in key

def del_refine_file(key):
    test_keys = (
        '_shelx_res_file', # shelxl
        '_iucr_refine_instructions_details' # olex2
    )
    return any(key == test_key for test_key in test_keys)

def is_num_esd(string):
    return re.search(r'[^\d\.\-\+\(\)]', string) is None

def replace_structure_from_cif(
    cif_path: Path,
    cif_dataset: str,
    structure_cif_path: Path,
    structure_cif_dataset: str,
    output_cif_path: Path):
    cif_obj = read_cif_safe(cif_path=cif_path)
    structure_cif_obj = read_cif_safe(cif_path=structure_cif_path)

    new_cif_obj = cif_obj.copy()

    new_block = cif_obj[cif_dataset].copy()

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
            del(new_block[key])

    #delete items
    for key in new_block.keys():
        if any(condition(key) for condition in conditions):
            del(new_block[key])

    structure_cif_block = structure_cif_obj[structure_cif_dataset]
    copy_loops = ('_atom_site', '_atom_site_aniso')

    for loop_key in copy_loops:
        loop = structure_cif_block[loop_key]

        for key, vals in loop.items():
            num_esd_column = all(map(is_num_esd, vals))
            brackets_column = any('(' in val  for val in vals)
            if num_esd_column and brackets_column:
                values, _ = split_esds(vals)
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


