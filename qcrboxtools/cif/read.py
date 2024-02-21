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
    cif_path = Path(cif_path)

    return cif.reader(input_string=cif_path.read_text(encoding='UTF-8')).model()


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

