# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from iotbx.cif import model

from qcrboxtools.robots.eval import RmatFile

rmat_content = """
# Created by peakref (version 1.4 2021091400) at 16-Nov-2023 13:49:55
# Host DW-PF3E3G40 User niklas WD /home/niklas/messing_around/eval/Ylid_OD_Images/
# mmAng(3337,0.06487) rotall(3337,0.02326) res=0.08813
RMAT P
   0.1600435   0.0135282   0.0149924
  -0.0090313  -0.0167682   0.1045150
   0.0466450  -0.0496632  -0.0312043
TMAT P mmm
   1.0000000   0.0000000   0.0000000
   0.0000000   1.0000000   0.0000000
   0.0000000   0.0000000   1.0000000
CELL 5.98990 18.47230 9.08270 90.00000 90.00000 90.00000 1004.97960
SIGMACELL 0.02070 0.03351 0.02659 0.18847 0.30094 0.22203 4.65000
"""

expected_rmat = np.array(
    [
        [0.1600435, 0.0135282, 0.0149924],
        [-0.0090313, -0.0167682, 0.1045150],
        [0.0466450, -0.0496632, -0.0312043],
    ]
)

expected_tmat = np.array(
    [
        [1.0000000, 0.0000000, 0.0000000],
        [0.0000000, 1.0000000, 0.0000000],
        [0.0000000, 0.0000000, 1.0000000],
    ]
)

expected_cell = np.array([5.9899, 18.4723, 9.0827, 90.0000, 90.0000, 90.0000, 1004.9796])

expected_sigmacell = np.array([0.02070, 0.03351, 0.02659, 0.18847, 0.30094, 0.22203, 4.6500])

cif_rmat_data = {
    "_diffrn_orient_matrix.ub_11": 0.1600435,
    "_diffrn_orient_matrix.ub_12": 0.0135282,
    "_diffrn_orient_matrix.ub_13": 0.0149924,
    "_diffrn_orient_matrix.ub_21": -0.0090313,
    "_diffrn_orient_matrix.ub_22": -0.0167682,
    "_diffrn_orient_matrix.ub_23": 0.1045150,
    "_diffrn_orient_matrix.ub_31": 0.0466450,
    "_diffrn_orient_matrix.ub_32": -0.0496632,
    "_diffrn_orient_matrix.ub_33": -0.0312043,
}

cif_cell_data = {
    "_cell.length_a": 5.9899,
    "_cell.length_b": 18.4723,
    "_cell.length_c": 9.0827,
    "_cell.angle_alpha": 90.0000,
    "_cell.angle_beta": 90.0000,
    "_cell.angle_gamma": 90.0000,
    "_cell.volume": 1004.9796,
}

cif_sigmacell_data = {
    "_cell.length_a_su": 0.02070,
    "_cell.length_b_su": 0.03351,
    "_cell.length_c_su": 0.02659,
    "_cell.angle_alpha_su": 0.18847,
    "_cell.angle_beta_su": 0.30094,
    "_cell.angle_gamma_su": 0.22203,
    "_cell.volume_su": 4.6500,
}

cif_space_group_data = {
    "_space_group.point_group_h-m": "mmm",
    "_space_group.centring_type": "P",
}


def test_extract_data():
    rmat = RmatFile("test.rmat", rmat_content)

    # Check if all expected keys are present in the data
    expected_keys = ["RMAT", "TMAT", "CELL", "SIGMACELL"]
    for key in expected_keys:
        assert key in rmat, f"Key '{key}' not found in extracted data"

    # Test RMAT values
    assert np.allclose(rmat["RMAT"], expected_rmat), "RMAT values do not match expected values"

    # Test TMAT values
    assert np.allclose(rmat["TMAT"], expected_tmat), "TMAT values do not match expected values"

    # Test CELL values
    assert np.allclose(rmat["CELL"], expected_cell), "CELL values do not match expected values"

    # Test SIGMACELL values
    assert np.allclose(rmat["SIGMACELL"], expected_sigmacell), "SIGMACELL values do not match expected values"


def test_to_cif_dict():
    rmat = RmatFile("test.rmat", rmat_content)
    cif_dict = rmat.to_cif_dict()

    for key, expected_value in cif_rmat_data.items():
        assert np.isclose(cif_dict[key], expected_value), f"CIF entry {key} does not match expected value"

    for key, expected_value in cif_cell_data.items():
        assert np.isclose(cif_dict[key], expected_value), f"CIF entry {key} does not match expected value"

    for key, expected_value in cif_sigmacell_data.items():
        assert np.isclose(cif_dict[key], expected_value), f"CIF entry {key} does not match expected value"


def test_from_cif_dict_and_file(tmp_path):
    # Merge all CIF dictionaries
    complete_cif_dict = {
        **cif_rmat_data,
        **cif_cell_data,
        **cif_sigmacell_data,
        **cif_space_group_data,
    }

    # Create a cif file from the dict
    block = model.block()
    for name, entry in complete_cif_dict.items():
        block.add_data_item(name, entry)
    cif = model.cif()
    cif["newblock"] = block
    cif_path = tmp_path / "test.cif"
    cif_path.write_text(str(cif))

    # test for creation from dict or cif file
    rmats = (
        RmatFile.from_cif_dict("test.rmat", complete_cif_dict),
        RmatFile.from_cif_file("test.rmat", cif_path),
    )
    for rmat in rmats:
        # Test RMAT values
        assert np.allclose(rmat["RMAT"], expected_rmat), "RMAT values do not match expected CIF data"

        # Test CELL values
        assert np.allclose(rmat["CELL"], expected_cell), "CELL values do not match expected CIF data"

        # Test SIGMACELL values
        assert np.allclose(rmat["SIGMACELL"], expected_sigmacell), "SIGMACELL values do not match expected CIF data"


def test_to_rmat_file(tmp_path):
    rmat = RmatFile("test.rmat", rmat_content)

    # Use pytest's tmp_path fixture to create a temporary file
    rmat.to_file(tmp_path)

    # Read back the content from the file
    with open(tmp_path / "test.rmat", "r", encoding="UTF-8") as file:
        content = [line for line in file.readlines() if len(line.strip()) > 0]

    # Extract the expected content from the original rmat_content
    expected_content = [line for line in rmat_content.strip().splitlines(keepends=True) if not line.startswith("#")]

    # Compare each line of the file content with the expected content
    for line, expected_line in zip(content, expected_content):
        assert (
            line == expected_line
        ), f"Line in file does not match expected line: {line.strip()} != {expected_line.strip()}"
