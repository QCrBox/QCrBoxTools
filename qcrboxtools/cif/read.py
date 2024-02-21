# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from typing import List, Union, Optional, Tuple

from iotbx import cif
from .uncertainties import split_su_block, split_su_cif
from .entries import block_to_unified_keywords, cif_to_unified_keywords

def read_cif_safe(cif_path: Union[str, Path]) -> cif.model.cif:
    """
    Reads a CIF file and returns its content as a CIF model, supporting both string
    paths and pathlib.Path objects.

    Parameters
    ----------
    cif_path : Union[str, Path]
        The path to the CIF file to be read.

    Returns
    -------
    cif.model.cif
        The CIF model parsed from the given file.
    """
    cif_path = Path(cif_path)

    return cif.reader(input_string=cif_path.read_text(encoding='UTF-8')).model()


def cifdata_str_or_index(
    model: cif.model.cif,
    dataset: Union[int, str]
) -> Tuple[cif.model.block, str]:
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

def read_cif_as_unified(
    cif_path: Union[str, Path],
    dataset: Optional[str] = None,
    convert_keywords: bool = True,
    custom_categories: Optional[List[str]] = None,
    split_sus: bool = True
) -> Union[cif.model.block, cif.model.cif]:
    """
    Read a CIF file, optionally process it for unified keywords and split standard
    uncertainties, and return the CIF model or a specific block.

    Parameters
    ----------
    cif_path : Union[str, Path]
        The path to the CIF file.
    dataset : Optional[str], optional
        The identifier for a specific dataset block within the CIF file. If not provided,
        the entire CIF model is processed and returned.
    convert_keywords : bool, default True
        Whether to convert CIF data item names to a unified naming scheme.
    custom_categories : Optional[List[str]], optional
        User defined categories (e.g. 'iucr', 'olex2' or 'shelx') that can be taken
        into account for entry name conversion, where an entry _category_example would
        look up _category.example in a block of the provided cif. Only used if
        `convert_keywords` is True.
    split_sus : bool, default True
        Whether to split values and their standard uncertainties (if present) into separate entries.

    Returns
    -------
    Union[cif.model.block, cif.model.cif]
        Depending on the input, either a single CIF block or the entire CIF model,
        processed according to the specified parameters.
    """
    cif_model = read_cif_safe(cif_path)
    if dataset is not None:
        block, _ = cifdata_str_or_index(cif_model, dataset)
        if convert_keywords:
            block = block_to_unified_keywords(block, custom_categories)
        if split_sus:
            block = split_su_block(block)
        return block
    if convert_keywords:
        cif_model = cif_to_unified_keywords(cif_model, custom_categories)
    if split_sus:
        cif_model = split_su_cif(cif_model)
    return cif_model

def cif_file_unify_split(
    input_cif_path: Union[str, Path],
    output_cif_path: Union[str, Path],
    convert_keywords: bool = True,
    custom_categories: Optional[List[str]] = None,
    split_sus: bool = True
) -> None:
    """
    Reads, processes, and writes a CIF file with optional modifications.

    Processes an input CIF file by optionally converting keywords and splitting
    standard uncertainties (SUs), then saves the modified content to a new file.

    Parameters
    ----------
    input_cif_path : Union[str, Path]
        The input CIF file path.
    output_cif_path : Union[str, Path]
        The output file path for the processed CIF.
    convert_keywords : bool, optional
        If True, converts keywords to a unified format.
    custom_categories : Optional[List[str]], optional
        Custom categories for keyword conversion, if applicable.
    split_sus : bool, optional
        If True, splits values from their SUs in the CIF content.

    Returns
    -------
    None
    """
    cif_model = read_cif_as_unified(
        input_cif_path,
        convert_keywords=convert_keywords,
        custom_categories=custom_categories,
        split_sus=split_sus
    )

    # Write the modified CIF model to the specified output file.
    Path(output_cif_path).write_text(str(cif_model), encoding='UTF-8')
