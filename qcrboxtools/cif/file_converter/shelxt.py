"""
Module for converting SHELXT INS files to CIF symmetry operation strings.

This module provides functions to read SHELXT INS files, extract symmetry
information, and convert it to CIF symmetry operation strings. It handles
various lattice types and symmetry instructions.

Functions
---------
symm_to_matrix_vector(instruction: str) -> Tuple[np.ndarray, np.ndarray]
    Converts a symmetry instruction into a symmetry matrix and a translation
    vector for that symmetry element.

ins2symm_cards_and_latt(ins_path: Path) -> Tuple[List[str], int]
    Reads an INS file and extracts symmetry cards and lattice information.

symm_cards_and_latt2symm_mat_vecs(symm_cards: List[str], latt: int) -> Tuple[np.ndarray, np.ndarray]
    Converts symmetry cards and lattice information to symmetry matrices and
    vectors.

symm_mat_vec2cifsymop(symm_mat: np.ndarray, symm_vec: np.ndarray) -> str
    Converts a symmetry matrix and vector to a CIF symmetry operation string.

ins2symop_loop(ins_path: Path) -> str
    Converts an INS file to a CIF symmetry operation loop string.

Constants
---------
latt2mat : dict
    Dictionary mapping lattice types to centering vectors.

"""

import fractions
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np

latt2mat = {
    1: np.array([[0, 0, 0]]),  # P
    2: np.array([[0, 0, 0], [0.5, 0.5, 0.5]]),  # I
    3: np.array([[0, 0, 0], [2 / 3, 1 / 3, 1 / 3], [1 / 3, 2 / 3, 2 / 3]]),  # O # TODO: Check this
    4: np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]),  # F
    5: np.array([[0, 0, 0], [0, 0.5, 0.5]]),  # A
    6: np.array([[0, 0, 0], [0.5, 0, 0.5]]),  # B
    7: np.array([[0, 0, 0], [0.5, 0.5, 0]]),  # C
}


def symm_to_matrix_vector(instruction: str) -> Tuple[np.ndarray, np.ndarray]:
    """Converts a symmetry instruction into a symmetry matrix and a translation
    vector for that symmetry element.

    Parameters
    ----------
    instruction : str
        Instruction string containing symmetry instruction for all three
        coordinates separated by comma signs (e.g -x, -y, 0.5+z)

    Returns
    -------
    symm_matrix: np.ndarray,
        size (3, 3) array containing the symmetry matrix for the symmetry element
    symm_vector: np.ndarray
        size (3) array containing the translation vector for the symmetry element
    """
    instruction_strings = [val.replace(" ", "").upper() for val in instruction.split(",")]
    matrix = np.zeros((3, 3), dtype=np.float64)
    vector = np.zeros(3, dtype=np.float64)
    for xyz, element in enumerate(instruction_strings):
        # search for fraction in a/b notation
        fraction1 = re.search(r"(-{0,1}\d{1,3})/(\d{1,3})(?![XYZ*])", element)
        # search for fraction in 0.0 notation
        fraction2 = re.search(r"(-{0,1}\d{0,1}\.\d{1,4})(?![XYZ*\d/])", element)
        # search for whole numbers
        fraction3 = re.search(r"(-{0,1}\d)(?![XYZ*/])", element)
        if fraction1:
            vector[xyz] = float(fraction1.group(1)) / float(fraction1.group(2))
        elif fraction2:
            vector[xyz] = float(fraction2.group(1))
        elif fraction3:
            vector[xyz] = float(fraction3.group(1))

        symm = re.findall(r"-{0,1}[\d\./]{0,8}\*?[XYZ]", element)
        for xyz_match in symm:
            xyz_match = xyz_match.replace("*", "")
            if len(xyz_match) == 1:
                sign = 1
            elif xyz_match[0] == "-" and len(xyz_match) == 2:
                sign = -1
            elif "/" in xyz_match:
                num_str, denom_str = xyz_match[:-1].split("/")
                sign = float(num_str) / float(denom_str)
            else:
                sign = float(xyz_match[:-1])
            index = "XYZ".index(xyz_match[-1])
            matrix[xyz, index] = sign

    return np.array(matrix), np.array(vector)


def ins2symm_cards_and_latt(ins_path: Path) -> Tuple[List[str], int]:
    """
    Reads an INS file and extracts symmetry cards and lattice information.

    Parameters
    ----------
    ins_path : Path
        Path to the SHELX .INS file.

    Returns
    -------
    symm_cards : List[str]
        List of symmetry cards extracted from the INS file.
    latt : int
        Lattice type integer extracted from the INS file.
    """
    ins_path = Path(ins_path)

    with ins_path.open(encoding="ASCII") as fobj:
        ins_lines = fobj.readlines()
    symm_cards = [line[4:].strip() for line in ins_lines if line.strip().startswith("SYMM ")]
    latt_line = next(line.strip() for line in ins_lines if line.strip().startswith("LATT"))

    latt = int(latt_line.split()[1])

    return symm_cards, latt


def symm_cards_and_latt2symm_mat_vecs(symm_cards: List[str], latt: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts symmetry cards and lattice information to symmetry matrices and
    vectors.

    Parameters
    ----------
    symm_cards : List[str]
        List of symmetry cards.
    latt : int
        Lattice type.

    Returns
    -------
    symm_mats : np.ndarray
        Array of symmetry matrices.
    symm_vecs : np.ndarray
        Array of symmetry vectors.
    """
    centring_vectors = latt2mat[abs(latt)]

    is_centred = latt > 0

    n_elements = (len(symm_cards) + 1) * len(centring_vectors) * (int(is_centred) + 1)

    symm_mats = np.zeros((n_elements, 3, 3))
    symm_vecs = np.zeros((n_elements, 3))

    symm_mats[0] = np.eye(3)

    symm_mats_vecs = [symm_to_matrix_vector(symm) for symm in symm_cards]
    if len(symm_mats_vecs) == 0:
        card_mats = np.zeros((0, 3, 3))
        card_vecs = np.zeros((0, 3))
    else:
        card_mats, card_vecs = (np.array(el) for el in zip(*symm_mats_vecs))
    if is_centred:
        symm_mats[1] = -np.eye(3)
        max_index = (len(symm_cards) + 1) * 2
        symm_mats[2:max_index:2] = card_mats
        symm_vecs[2:max_index:2] = card_vecs
        symm_mats[3 : max_index + 1 : 2] = -card_mats
        symm_vecs[3 : max_index + 1 : 2] = -card_vecs
    else:
        max_index = len(symm_cards)
        symm_mats[1 : max_index + 1] = card_mats
        symm_vecs[1 : max_index + 1] = card_vecs
    print(max_index, len(centring_vectors))
    for i, centring in enumerate(centring_vectors):
        print(max_index * i, max_index * (i + 1))
        symm_mats[max_index * i : max_index * (i + 1)] = symm_mats[:max_index]
        symm_vecs[max_index * i : max_index * (i + 1)] = symm_vecs[:max_index] + centring[None, :]

    symm_vecs = (symm_vecs + 0.5) % 1 - 0.5
    symm_vecs[symm_vecs == -0.5] = 0.5

    return symm_mats, symm_vecs


def symm_mat_vec2cifsymop(symm_mat: np.ndarray, symm_vec: np.ndarray) -> str:
    """
    Converts a symmetry matrix and vector to a CIF symmetry operation string.

    Parameters
    ----------
    symm_mat : np.ndarray
        Symmetry matrix.
    symm_vec : np.ndarray
        Symmetry vector.

    Returns
    -------
    symm_string : str
        CIF symmetry operation string.
    """
    symm_string = ""
    for symm_parts, add in zip(symm_mat, symm_vec):
        symm_string_add = str(fractions.Fraction(add).limit_denominator(50))
        if symm_string_add != "0":
            symm_string += symm_string_add
        for symm_part, symbol in zip(symm_parts, "xyz"):
            if abs(symm_part) < 1e-10:
                continue
            if abs(1 - abs(symm_part)) < 1e-10:
                if symm_part > 0:
                    symm_string += f"+{symbol}"
                else:
                    symm_string += f"-{symbol}"
            else:
                fraction = fractions.Fraction(symm_part).limit_denominator(50)
                if str(fraction).startswith("-"):
                    symm_string += f"{str(fraction)}*{symbol}"
                else:
                    symm_string += f"+{str(fraction)}*{symbol}"
        symm_string += ","

    return symm_string[:-1]


def ins2symop_loop(ins_path: Path) -> str:
    """
    Converts an INS file to a CIF symmetry operation loop string.

    Parameters
    ----------
    ins_path : Path
        Path to the INS file.

    Returns
    -------
    symop_string : str
        CIF symmetry operation loop string.
    """
    symm_cards, latt = ins2symm_cards_and_latt(ins_path)
    symm_mats, symm_vecs = symm_cards_and_latt2symm_mat_vecs(symm_cards, latt)
    symm_strings = [symm_mat_vec2cifsymop(symm_mat, symm_vec) for symm_mat, symm_vec in zip(symm_mats, symm_vecs)]

    symop_string = "loop_\n  _space_group_symop.id\n  _space_group_symop.operation_xyz"
    for i, symm_string in enumerate(symm_strings):
        symop_string += f"\n  {i + 1}  '{symm_string}'"
    return symop_string
