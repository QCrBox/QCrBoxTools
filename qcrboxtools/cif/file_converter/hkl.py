# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

"""
This module provides utilities for the conversion of CIF data to the SHELX HKL format
"""

from typing import Union

import numpy as np

from ..merge import cifdata_str_or_index
from ..read import read_cif_as_unified


def format_floats(val: float) -> str:
    """
    Format a floating-point number to a specific string format.

    Parameters:
    - val (float): The floating-point number to be formatted.

    Returns:
    - str: The formatted string representation of the input float.
    """
    if val < 0:
        return f"{val: .8f}"[:8]
    else:
        return f" {val:.8f}"[:8]


def cif2hkl4(cif_path: str, cif_dataset: Union[int, str], hkl_path: str) -> None:
    """
    Convert CIF data to the HKL format and save to the specified file path.

    Parameters:
    - cif_path (str): Path to the input CIF file.
    - cif_dataset (int or str): Index or string identifier for the desired CIF dataset.
    - hkl_path (str): Path where the converted HKL data should be saved.

    Returns:
    - None
    """
    cif_model = read_cif_as_unified(cif_path)

    cif_data, _ = cifdata_str_or_index(cif_model, cif_dataset)

    if "_shelx.hkl_file" in cif_data:
        hkl_content = cif_data["_shelx.hkl_file"]
    else:
        if "_diffrn_refln.scale_group_code" in cif_data:
            use_entries = [
                np.array(cif_data["_diffrn_refln.index_h"], dtype=np.int64),
                np.array(cif_data["_diffrn_refln.index_k"], dtype=np.int64),
                np.array(cif_data["_diffrn_refln.index_l"], dtype=np.int64),
                [format_floats(float(val)) for val in cif_data["_diffrn_refln.intensity_net"]],
                [format_floats(float(val)) for val in cif_data["_diffrn_refln.intensity_net_su"]],
                np.array(cif_data["_diffrn_refln.scale_group_code"], dtype=np.int64),
            ]
            line_format = "{:4d}{:4d}{:4d}{}{}{:4d}"
        else:
            use_entries = [
                np.array(cif_data["_diffrn_refln.index_h"], dtype=np.int64),
                np.array(cif_data["_diffrn_refln.index_k"], dtype=np.int64),
                np.array(cif_data["_diffrn_refln.index_l"], dtype=np.int64),
                [format_floats(float(val)) for val in cif_data["_diffrn_refln.intensity_net"]],
                [format_floats(float(val)) for val in cif_data["_diffrn_refln.intensity_net_su"]],
            ]
            line_format = "{:4d}{:4d}{:4d}{}{}"
        hkl_content = "\n".join(line_format.format(*entryset) for entryset in zip(*use_entries))
    with open(hkl_path, "w", encoding="ASCII") as fo:
        fo.write(hkl_content)
