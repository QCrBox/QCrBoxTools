"""
This module provides utilities for the conversion of CIF data to the SHELX HKL format
"""

import numpy as np
from iotbx import cif

from ...util.cif import cifdata_str_or_index

def format_floats(val: float) -> str:
    """
    Format a floating-point number to a specific string format.

    Parameters:
    - val (float): The floating-point number to be formatted.

    Returns:
    - str: The formatted string representation of the input float.
    """
    if val < 0:
        return f'{val: .8f}'[:8]
    else:
        return f' {val:.8f}'[:8]

def cif2hkl4(cif_path: str, cif_dataset: [int, str], hkl_path: str) -> None:
    """
    Convert CIF data to the HKL format and save to the specified file path.

    Parameters:
    - cif_path (str): Path to the input CIF file.
    - cif_dataset (int or str): Index or string identifier for the desired CIF dataset.
    - hkl_path (str): Path where the converted HKL data should be saved.

    Returns:
    - None
    """
    with open(cif_path, 'r', encoding='UTF-8') as fo:
        cif_content = fo.read()

    cif_data, _ = cifdata_str_or_index(
        cif.reader(input_string=cif_content).model(),
        cif_dataset
    )

    if '_shelx_hkl_file' in cif_data:
        hkl_content = cif_data['_shelx_hkl_file']
    else:
        if '_diffrn_refln_scale_group_code' in cif_data:
            use_entries = [
                np.array(cif_data['_diffrn_refln_index_h'], dtype=np.int64),
                np.array(cif_data['_diffrn_refln_index_k'], dtype=np.int64),
                np.array(cif_data['_diffrn_refln_index_l'], dtype=np.int64),
                [format_floats(float(val)) for val in cif_data['_diffrn_refln_intensity_net']],
                [format_floats(float(val)) for val in cif_data['_diffrn_refln_intensity_u']],
                np.array(cif_data['_diffrn_refln_scale_group_code'], dtype=np.int64),
            ]
            line_format = '{:4d}{:4d}{:4d}{}{}{:4d}'
        else:
            use_entries = [
                np.array(cif_data['_diffrn_refln_index_h'], dtype=np.int64),
                np.array(cif_data['_diffrn_refln_index_k'], dtype=np.int64),
                np.array(cif_data['_diffrn_refln_index_l'], dtype=np.int64),
                [format_floats(float(val)) for val in cif_data['_diffrn_refln_intensity_net']],
                [format_floats(float(val)) for val in cif_data['_diffrn_refln_intensity_u']]
            ]
            line_format = '{:4d}{:4d}{:4d}{}{}'
        hkl_content = '\n'.join(line_format.format(*entryset) for entryset in zip(*use_entries))
    with open(hkl_path, 'w', encoding='ASCII') as fo:
        fo.write(hkl_content)
