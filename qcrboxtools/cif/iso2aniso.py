# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from .read import cifdata_str_or_index, read_cif_safe
from .uncertainties import split_su_single


def cif_iso2aniso(
    input_cif_path: Union[str, Path],
    cif_dataset: Union[str, int],
    output_cif_path: Union[str, Path],
    select_names: Optional[List[str]] = None,
    select_elements: Optional[List[str]] = None,
    select_regexes: Optional[List[re.Pattern]] = None,
    overwrite: bool = False,
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
    atom_site_labels = list(block["_atom_site.label"])

    # Get selected atoms
    if select_names is None:
        select_names = []

    if select_elements is not None:
        select_names += [
            name
            for name, element in zip(atom_site_labels, block["_atom_site.type_symbol"])
            if element in select_elements
        ]

    if select_regexes is not None:
        for regex in select_regexes:
            select_names += [name for name in atom_site_labels if re.match(regex, name) is not None]

    select_names = list(set(select_names))

    # if overwrite False remove preexistring
    existing = list(block["_atom_site_aniso.label"])
    if not overwrite:
        select_names = [name for name in select_names if name not in existing]

    # calculate values and set adp type
    new_values = {}
    for name in select_names:
        uiso_index = atom_site_labels.index(name)
        uiso = split_su_single(block["_atom_site.u_iso_or_equiv"][uiso_index])[0]
        new_values[name] = single_value_iso2aniso(
            uiso,
            split_su_single(block["_cell.angle_alpha"])[0],
            split_su_single(block["_cell.angle_beta"])[0],
            split_su_single(block["_cell.angle_gamma"])[0],
        )
        block["_atom_site.adp_type"][uiso_index] = "Uani"

    # build up new atom_site_aniso arrays
    loop = block["_atom_site_aniso"]
    new_aniso_labels = list(sorted(existing + select_names, key=atom_site_labels.index))

    for _ in range(len(new_aniso_labels) - loop.n_rows()):
        loop.add_row(["?"] * loop.n_columns())
    loop.update_column("_atom_site_aniso.label", new_aniso_labels)
    for ij_index, ij in enumerate((11, 22, 33, 12, 13, 23)):
        aniso_key = f"_atom_site_aniso.u_{ij}"
        loop.update_column(
            f"_atom_site_aniso.u_{ij}",
            [
                f"{new_values[name][ij_index]:8.8f}" if name in select_names else block[aniso_key][existing.index(name)]
                for name in new_aniso_labels
            ],
        )

    cif_content[block_name] = block

    with open(output_cif_path, "w", encoding="UTF-8") as fobj:
        fobj.write(str(cif_content))


def calc_rec_angle_cosines(alpha: float, beta: float, gamma: float) -> Tuple[float, float, float]:
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

    cos_alpha_star = (np.cos(beta_rad) * np.cos(gamma_rad) - np.cos(alpha_rad)) / (np.sin(beta_rad) * np.sin(gamma_rad))
    cos_beta_star = (np.cos(alpha_rad) * np.cos(gamma_rad) - np.cos(beta_rad)) / (np.sin(alpha_rad) * np.sin(gamma_rad))
    cos_gamma_star = (np.cos(alpha_rad) * np.cos(beta_rad) - np.cos(gamma_rad)) / (np.sin(alpha_rad) * np.sin(beta_rad))

    return cos_alpha_star, cos_beta_star, cos_gamma_star


def single_value_iso2aniso(
    uiso: float, alpha: float, beta: float, gamma: float
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
