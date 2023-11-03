from pathlib import Path
from iotbx import cif
from typing import Iterable
import re
import numpy as np


def split_esd_single(input_string: str):
    PATTERN = r'([^\.]+)\.?(.*?)\(([\d\.]+)\)'
    match = re.match(PATTERN, input_string)
    if len(match.group(2)) == 0:
        return float(match.group(1)), float(match.group(3))
    magnitude = 10.0**(-len(match.group(2)))
    if match.group(1).startswith('-'):
        sign = -1
    else:
        sign = 1
    value = float(match.group(1)) + sign * magnitude * float(match.group(2))
    esd = magnitude * float(match.group(3).replace('.', ''))
    return value, esd

def split_esds(input_strings: Iterable):
    values, esds = zip(*map(split_esd_single, input_strings))
    return np.array(list(values)), np.array(list(esds))

def add_structure_from_cif(
    cif_path: Path,
    structure_cif_path: Path,
    output_cif_path: Path):
    pass
