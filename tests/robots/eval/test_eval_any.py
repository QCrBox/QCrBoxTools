# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

import subprocess

import pytest

from qcrboxtools.robots.eval import EvalAnyRobot

mock_sad_content = """
   0  -3  -110126.41   89.28   1-0.81459 0.81523-0.21272-0.02270 0.53963-0.57870  31.24 211.16    3.27 -18.62 1678
  -1   6   810009.44  151.56   1-0.81423 0.69449-0.21117 0.68121 0.54078-0.23161 342.91 318.27    2.38 -18.62 4048
   1  -5   621293.28  118.33   1-0.81327 0.93188-0.20712-0.18515 0.54377-0.31198  51.17  58.23    2.68 -18.62 3312
  -3   7   2 2382.12   77.34   1-0.81421 0.45683-0.21112 0.75953 0.54082-0.46307 311.54 458.72    2.39 -18.62 4640
  -2   7   6   24.66   27.14   1-0.81382 0.57486-0.20945 0.75734 0.54205-0.30981 349.93 384.10    2.51 -18.62 4515
   1   2  14 7662.79  128.02   1-0.81403 0.93050-0.21033 0.36629 0.54140-0.00081 319.29 141.24    2.44 -18.62 4047
  -3   8   6 1045.44   71.59   1-0.81282 0.45595-0.20522 0.83297 0.54517-0.31349 384.17 441.87    2.81 -18.62 5334
   1   1  20  351.88   48.53   1-0.81375 0.93012-0.20915 0.28682 0.54228 0.22936 379.51  41.73    2.53 -18.62 5524
   0   4  21 1115.39   88.82   1-0.81367 0.81073-0.20880 0.52079 0.54253 0.26738 470.43 121.06    2.55 -18.62 6117
   1   2  15 5719.27  116.66   1-0.81384 0.93011-0.20952 0.36540 0.54201 0.03709 331.44 130.28    2.50 -18.62 4303
   1   2  13 1211.48   55.96   1-0.81347 0.93044-0.20796 0.36412 0.54315-0.04109 307.32 151.79    2.62 -18.62 3793
"""[1:-1]

@pytest.fixture(name='robot')
def fixture_robot(tmp_path):
    return EvalAnyRobot(tmp_path)

def test_init(robot, tmp_path):
    assert robot.work_folder == tmp_path

def test_create_abs_with_monkeypatch(monkeypatch, robot):
    # Replace subprocess.call with a dummy function that does nothing
    monkeypatch.setattr(subprocess, "call", lambda *args, **kwargs: None)

    robot.create_abs()

    # Check if the expected file is created
    assert (robot.work_folder / 'any_output.log').exists()

def test_create_cif_dict(monkeypatch, robot, tmp_path):
    # Replace the create_abs method with a dummy function
    monkeypatch.setattr(robot, 'create_abs', lambda: None)

    # Write mock_sad_content to a temporary file
    sad_path = tmp_path / 'shelx.sad'
    with open(sad_path, 'w', encoding='UTF-8') as f:
        f.write(mock_sad_content)

    # Execute the method and validate the output
    result = robot.create_cif_dict()

    # Verify the returned dictionary
    assert result['_diffrn_refln.index_h'][0] == 0
    assert result['_diffrn_refln.index_k'][1] == 6
    assert result['_diffrn_refln.index_l'][2] == 6
    assert result['_diffrn_refln.intensity_net'][3] == 2382.12
    assert result['_diffrn_refln.intensity_net_su'][4] == 27.14
    assert result['_diffrn_refln.class_code'][4] == 1
    assert result['_qcrbox.diffrn_refln.direction_cosine_incid_x'][6] == -0.81282
    assert result['_qcrbox.diffrn_refln.direction_cosine_incid_y'][7] == 0.93012
    assert result['_qcrbox.diffrn_refln.direction_cosine_incid_z'][8] == -0.20880
    assert result['_qcrbox.diffrn_refln.direction_cosine_diffrn_x'][9] == 0.36540
    assert result['_qcrbox.diffrn_refln.direction_cosine_diffrn_y'][10] == 0.54315
    assert result['_qcrbox.diffrn_refln.direction_cosine_diffrn_z'][10] == -0.04109
    assert result['_qcrbox.diffrn_refln.detector_px_x_obs'][9] == 331.44
    assert result['_qcrbox.diffrn_refln.detector_px_y_obs'][8] == 121.06
    assert result['_qcrbox.diffrn_refln.detector_frame_obs'][7] == 2.53
    assert result['_qcrbox.diffrn_refln.evalsad_mystery_val1'][6] == -18.62
    assert result['_qcrbox.diffrn_refln.evalsad_mystery_val2'][5] == 4047

def test_create_cif_file(monkeypatch, robot, tmp_path):
    # Mock the data returned by create_abs
    monkeypatch.setattr(robot, 'create_abs', lambda: None)

    # Write mock_sad_content to a temporary file
    sad_path = tmp_path / 'shelx.sad'
    with open(sad_path, 'w', encoding='UTF-8') as f:
        f.write(mock_sad_content)

    file_path = tmp_path / 'output.cif'
    robot.create_cif_file(file_path)

    assert file_path.exists()

    #TODO add more verification
