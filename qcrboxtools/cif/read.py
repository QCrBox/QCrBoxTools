# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
import re
from typing import Iterable, Union, Tuple

from iotbx import cif
import numpy as np

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
            'The given dataset does not exists and integer index is out of range. '
            + f'Got: {dataset}'
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
