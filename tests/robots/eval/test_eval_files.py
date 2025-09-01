# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
from iotbx.cif import model, reader

from qcrboxtools.robots.eval import PicFile, RmatFile, SettingsVicFile, TextFile
from qcrboxtools.robots.eval.eval_files import infer_and_cast


# General tests
def test_infer_and_cast():
    assert infer_and_cast("1") == 1
    assert infer_and_cast("1.0") == 1.0
    assert np.allclose(infer_and_cast("1.0 2.0 3.0"), np.array([1.0, 2.0, 3.0]))
    assert infer_and_cast("true") is True
    assert infer_and_cast("false") is False
    assert infer_and_cast("True") is True
    assert infer_and_cast("Something") == "Something"


# PIC tests
def test_pic_string_representation_and_file_writing(tmp_path):
    content = "model 4\nxa 0.2\nxb 0.2\nxc 0.2\nlambdainit\nmicavec 0 0 1"
    pic_file = PicFile("output.pic", content)
    pic_file["micavec"][0].options[2] = 2
    pic_file.to_file(tmp_path)
    with open(tmp_path / "output.pic", "r", encoding="UTF-8") as f:
        written_content = f.read()
    assert written_content.strip()[-1] == "2"
    assert written_content.strip()[:-1] == content.strip()[:-1]


def test_pic_command_parameters_parsing():
    content = "model 4\nxa 0.2 xb 0.2 xc 0.2\n\nlambdainit\nmicavec 0 0 1"
    pic_file = PicFile("test.pic", content)
    assert pic_file["model"][0].options == 4
    assert pic_file["xa"][0].options == 0.2
    assert pic_file["xb"][0].options == 0.2
    assert pic_file["xc"][0].options == 0.2
    assert pic_file["lambdainit"][0].options is None
    assert pic_file["micavec"][0].options[0] == 0
    assert pic_file["micavec"][0].options[1] == 0
    assert pic_file["micavec"][0].options[2] == 1

    with pytest.raises(KeyError):
        pic_file["non_existing_command"]


# RMAT tests
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


cif_trans_matrix_data = {
    "_diffrn_reflns_transf_matrix.11": 1.0,
    "_diffrn_reflns_transf_matrix.12": 0.0,
    "_diffrn_reflns_transf_matrix.13": 0.0,
    "_diffrn_reflns_transf_matrix.21": 0.0,
    "_diffrn_reflns_transf_matrix.22": 1.0,
    "_diffrn_reflns_transf_matrix.23": 0.0,
    "_diffrn_reflns_transf_matrix.31": 0.0,
    "_diffrn_reflns_transf_matrix.32": 0.0,
    "_diffrn_reflns_transf_matrix.33": 1.0,
}


def test_rmat_extract_data():
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


def test_rmat_to_cif_dict():
    rmat = RmatFile("test.rmat", rmat_content)
    cif_dict = rmat.to_cif_dict()

    for key, expected_value in cif_rmat_data.items():
        assert np.isclose(cif_dict[key], expected_value), f"CIF entry {key} does not match expected value"

    for key, expected_value in cif_cell_data.items():
        assert np.isclose(cif_dict[key], expected_value), f"CIF entry {key} does not match expected value"

    for key, expected_value in cif_sigmacell_data.items():
        assert np.isclose(cif_dict[key], expected_value), f"CIF entry {key} does not match expected value"


@pytest.mark.parametrize(
    "cif_space_group_data",
    [
        {
            "_space_group.point_group_h-m": "mmm",
            "_space_group.centring_type": "P",
        },
        {
            "_space_group.point_group_h-m": "mmm",
            "_space_group.name_h-m_full": "P m m m",
        },
    ],
)
def test_rmat_from_cif_dict_and_file(cif_space_group_data, tmp_path):
    # Merge all CIF dictionaries
    complete_cif_dict = {
        **cif_rmat_data,
        **cif_cell_data,
        **cif_sigmacell_data,
        **cif_space_group_data,
        **cif_trans_matrix_data,
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

        # Test TMAT values
        if "_space_group.centring_type" in cif_space_group_data:
            assert np.allclose(rmat["TMAT"], expected_tmat), "TMAT values do not match expected CIF data"


def test_rmat_from_cif_dict_centring_missing():
    incomplete_cif_dict = {
        **cif_rmat_data,
        **cif_cell_data,
        **cif_sigmacell_data,
        **cif_trans_matrix_data,
    }

    with pytest.raises(KeyError):
        RmatFile.from_cif_dict("test.rmat", incomplete_cif_dict)


def test_rmat_to_rmat_file(tmp_path):
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
        assert line == expected_line, (
            f"Line in file does not match expected line: {line.strip()} != {expected_line.strip()}"
        )


def test_rmat_to_cif_file(tmp_path):
    rmat = RmatFile("test.rmat", rmat_content)
    cif_path = tmp_path / "test.cif"

    # Use pytest's tmp_path fixture to create a temporary file
    rmat.to_cif_file(cif_path, "test_block")

    # Read back the content from the file
    block = reader(str(cif_path)).model()["test_block"]

    for key, value in rmat.to_cif_dict().items():
        assert key in block, f"Key '{key}' not found in CIF block"
        assert infer_and_cast(block[key]) == value, f"Value for key '{key}' does not match expected value"


# Test SettingsVicFile

beamstop_vic = dedent("""\
    ! Created by view (version 1.4 2023091300) at 16-Nov-2023 13:39:41
    ! Host DW-PF3E3G40 User niklas WD /home/niklas/messing_around/eval/Ylid_OD_Images/
    beamstopid oxford_e_81
    beamstopcolour red
    beamstopdiameter 0.1
    beamstopwidth 1.0
    beamstop 0.0 0.0
    beamstopshift 0.0 0.0
    beamstopangle 0.0
    beamstopdistance 10.0
""")

detalign_vic = dedent("""\
    ! Created by peakref (version 1.4 2021091400) at 16-Nov-2023 15:09:03
    ! Host DW-PF3E3G40 User niklas WD /home/niklas/messing_around/eval/Ylid_OD_Images/
    ! mmAng(3737,0.04442) rotpartial(632,0.12806) res=0.17248
    ALIGNDETID oxford_e_81
    ! Displacements in mm
    DETZEROX -0.229091
    DETZEROY -0.471509
    DETZEROZ 0.506907
    ! Rotations in degrees
    DETROTX 0.569459
    DETROTY -0.099205
    DETROTZ -0.086215
""")


def test_vic_parsing_file_content():
    vic_file = SettingsVicFile("beamstop.vic", beamstop_vic)
    assert isinstance(vic_file, SettingsVicFile)
    assert "!" not in vic_file


@pytest.mark.parametrize(
    "content, expected_data",
    [
        (
            beamstop_vic,
            {
                "beamstopid": "oxford_e_81",
                "beamstopcolour": "red",
                "beamstopdiameter": 0.1,
                "beamstopwidth": 1.0,
                "beamstop": np.array([0.0, 0.0]),
                "beamstopshift": np.array([0.0, 0.0]),
                "beamstopangle": 0.0,
                "beamstopdistance": 10.0,
            },
        ),
        (
            detalign_vic,
            {
                "ALIGNDETID": "oxford_e_81",
                "DETZEROX": -0.229091,
                "DETZEROY": -0.471509,
                "DETZEROZ": 0.506907,
                "DETROTX": 0.569459,
                "DETROTY": -0.099205,
                "DETROTZ": -0.086215,
            },
        ),
    ],
)
def test_vic_data_integrity(content, expected_data):
    vic_file = SettingsVicFile("test.vic", content)
    for key, value in expected_data.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(vic_file[key], value)
        else:
            assert vic_file[key] == value


def test_vic_data_modification():
    vic_file = SettingsVicFile("beamstop.vic", beamstop_vic)
    vic_file["beamstopid"] = "new_id"
    assert vic_file["beamstopid"] == "new_id"


def test_vic_string_representation_and_file_writing(tmp_path):
    vic_file = SettingsVicFile("beamstop.vic", beamstop_vic)
    expected_content = "\n".join(line for line in beamstop_vic.split("\n") if not line.startswith("!"))
    vic_file.to_file(tmp_path)
    with open(tmp_path / "beamstop.vic", "r", encoding="UTF-8") as f:
        written_content = f.read()
    assert written_content.strip() == expected_content.strip()


# Test TextFile
@pytest.fixture(name="sample_file")
def fixture_sample_file(tmp_path):
    sample_text = "This is a sample text."
    file = tmp_path / "sample.txt"
    file.write_text(sample_text, encoding="UTF-8")
    return file


def test_text_reading_file(sample_file):
    text_file = TextFile.from_file(str(sample_file))
    assert text_file.text == "This is a sample text."
    assert text_file.filename == "sample.txt"


def test_text_writing_file(tmp_path, sample_file):
    text_file = TextFile.from_file(str(sample_file))
    new_content = "This is a new text."
    text_file.text = new_content
    text_file.to_file(str(tmp_path))
    new_file_path = tmp_path / "sample.txt"
    assert new_file_path.read_text(encoding="UTF-8") == new_content


def test_text_writing_to_different_directory(tmp_path, sample_file):
    text_file = TextFile.from_file(str(sample_file))
    new_content = "Modified text."
    text_file.text = new_content
    new_directory = tmp_path / "subfolder"
    new_directory.mkdir()
    text_file.to_file(str(new_directory))
    new_file_path = new_directory / "sample.txt"
    assert new_file_path.read_text(encoding="UTF-8") == new_content


def test_text_writing_to_current_directory(sample_file, monkeypatch):
    text_file = TextFile.from_file(str(sample_file))
    new_content = "Content for current directory."
    text_file.text = new_content
    with monkeypatch.context() as m:
        m.chdir(sample_file.parent)
        text_file.to_file()
        assert Path("sample.txt").read_text(encoding="UTF-8") == new_content
