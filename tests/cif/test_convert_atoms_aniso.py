# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

import numpy as np
import pytest

from qcrboxtools.cif.iso2aniso import (
    cif_iso2aniso,
    cifdata_str_or_index,
    read_cif_safe,
    single_value_iso2aniso,
    split_su_single,
)

test_file_path = Path("./tests/cif/cif_files")


@pytest.mark.parametrize(
    "uiso, expected",
    [
        (0.040, (0.040, 0.040, 0.040, -0.000, 0.007, -0.000)),
        (0.053, (0.053, 0.053, 0.053, -0.000, 0.009, -0.000)),
    ],
)
def test_single_value_iso2aniso(uiso, expected):
    alpha, beta, gamma = 90, 100.37, 90
    result = single_value_iso2aniso(uiso, alpha, beta, gamma)
    assert np.isclose(result, expected, atol=1e-3).all()


def uiso_matches_uaniso(atom_name, block):
    aniso_labels = list(block["_atom_site_aniso.label"])
    assert atom_name in aniso_labels
    aniso_index = aniso_labels.index(atom_name)
    uiso_index = list(block["_atom_site.label"]).index(atom_name)
    uiso = split_su_single(block["_atom_site.u_iso_or_equiv"][uiso_index])[0]
    uaniso = np.array(
        [split_su_single(block[f"_atom_site_aniso.u_{ij}"][aniso_index])[0] for ij in (11, 22, 33, 12, 13, 23)]
    )
    u_aniso_comp = single_value_iso2aniso(
        uiso,
        split_su_single(block["_cell.angle_alpha"])[0],
        split_su_single(block["_cell.angle_beta"])[0],
        split_su_single(block["_cell.angle_gamma"])[0],
    )
    assert np.isclose(uaniso, u_aniso_comp, atol=1e-3).all()
    assert block["_atom_site.adp_type"][uiso_index] == "Uani"


def test_cif_iso2aniso_byname(tmp_path):
    input_cif_path = test_file_path / "iso2aniso.cif"
    output_path = tmp_path / "output.cif"
    cif_dataset = 0
    # only replace H1a
    cif_iso2aniso(input_cif_path, cif_dataset, output_path, select_names=["H2a"])
    block, _ = cifdata_str_or_index(read_cif_safe(output_path), cif_dataset)
    # test H2a has been added
    uiso_matches_uaniso("H2a", block)

    # test H2b has not been added
    aniso_labels = list(block["_atom_site_aniso.label"])
    assert all(label not in aniso_labels for label in ("H2b", "H3a", "H3b"))


def test_cif_iso2aniso_byelement(tmp_path):
    input_cif_path = test_file_path / "iso2aniso.cif"
    output_path = tmp_path / "output.cif"
    cif_dataset = 0
    # replace all H
    cif_iso2aniso(input_cif_path, cif_dataset, output_path, select_elements=["H"])
    block, _ = cifdata_str_or_index(read_cif_safe(output_path), cif_dataset)

    # test H1a H1b H2a H2b have been added
    for atom_name in ("H2a", "H2b", "H3a", "H3b"):
        uiso_matches_uaniso(atom_name, block)


def test_cif_iso2aniso_byregex(tmp_path):
    input_cif_path = test_file_path / "iso2aniso.cif"
    output_path = tmp_path / "output.cif"
    cif_dataset = 0
    # replace H*b
    cif_iso2aniso(input_cif_path, cif_dataset, output_path, select_regexes=[r"H\db"])
    block, _ = cifdata_str_or_index(read_cif_safe(output_path), cif_dataset)

    # test H2b H3b have been added
    for atom_name in ("H2b", "H3b"):
        uiso_matches_uaniso(atom_name, block)
    # test H2a H3b have not been added
    aniso_labels = list(block["_atom_site_aniso.label"])
    assert all(label not in aniso_labels for label in ("H2a", "H3a"))
