# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0
import re
from pathlib import Path

import numpy as np
import pytest

from qcrboxtools.cif.file_converter.hkl import cif2hkl4

test_file_dir = Path("./tests/cif/convert/test_files")


def valid_hkl_line(line):
    return re.search(r"[^\d\s\.\-]", line) is None and len(line.strip()) > 0


def read_hkl_line(line):
    if len(line) > 29:
        return [
            int(line[0:4]),
            int(line[4:8]),
            int(line[8:12]),
            float(line[12:20]),
            float(line[20:28]),
            int(line[28:]),
        ]
    else:
        return [
            int(line[0:4]),
            int(line[4:8]),
            int(line[8:12]),
            float(line[12:20]),
            float(line[20:28]),
        ]


def read_hkl_as_np(hkl_path, sort=False):
    with open(hkl_path, encoding="ASCII") as fo:
        hkl_lines = [read_hkl_line(line) for line in fo.readlines() if valid_hkl_line(line)]
    pivot = list(zip(*hkl_lines))
    if len(pivot) == 5:
        miller_h, miller_k, miller_l, i, su_i = pivot
        number = None
    elif len(pivot) == 6:
        miller_h, miller_k, miller_l, i, su_i, number = pivot
        number = np.array([int(val) for val in number])
    else:
        raise ValueError(f"len(pivot) was {len(pivot)} {str(pivot)}")
    hkl = np.stack(
        (
            miller_h,
            miller_k,
            miller_l,
        ),
        axis=-1,
    )
    i = np.array(i)
    su_i = np.array(su_i)
    remove_zero_mask = np.logical_not(np.all(hkl == 0, axis=-1))
    hkl = hkl[remove_zero_mask, :].copy()
    i = i[remove_zero_mask].copy()
    su_i = su_i[remove_zero_mask].copy()
    if number is not None:
        number = number[remove_zero_mask].copy()

    if sort:
        sort_mask = np.argsort(hkl[:, 0] * 1e8 + hkl[:, 1] * 1e4 + hkl[:, 2])
        hkl = hkl[sort_mask, :].copy()
        i = i[sort_mask].copy()
        su_i = su_i[sort_mask].copy()
        if number is not None:
            number = number[sort_mask].copy()
    su_i *= 99999.0 / i.max()
    i *= 99999.0 / i.max()
    return hkl, i, su_i, number


@pytest.mark.parametrize(
    "cif_path", [test_file_dir / "olex.cif", test_file_dir / "shelx.cif", test_file_dir / "olex_noscale.cif"]
)
def test_cif_2_shelx_hkl(cif_path, tmp_path):
    # read shelxl hkl (created by olex)
    target_path = test_file_dir / "target.hkl"
    # convert into numpy arrays hkl, intensity, su
    # sort arrays by h, k, l
    hkl_ref, i_ref, su_ref, _ = read_hkl_as_np(target_path, True)
    # create converted hkl from cif into temporary file
    out_hkl_path = tmp_path / "test.hkl"
    # read file the same way
    cif2hkl4(cif_path, 0, out_hkl_path)
    hkl_test, i_test, su_test, _ = read_hkl_as_np(out_hkl_path, True)
    # compare whether identical
    assert hkl_ref.shape[0] == hkl_test.shape[0], "Not the same number of reflections"

    assert np.all(hkl_ref == hkl_test), "miller indicees not the same"
    assert np.all(np.abs(i_ref - i_test) < 0.01), "intensities not the same"
    assert np.all(np.abs(su_ref - su_test) < 0.01), "sus not the same"
