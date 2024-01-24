from textwrap import dedent

import numpy as np
import pytest

from qcrboxtools.robots.eval import SettingsVicFile

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

def test_parsing_file_content():
    vic_file = SettingsVicFile('beamstop.vic', beamstop_vic)
    assert isinstance(vic_file, SettingsVicFile)
    assert '!' not in vic_file

@pytest.mark.parametrize("content, expected_data", [
    (beamstop_vic, {
        'beamstopid': 'oxford_e_81',
        'beamstopcolour': 'red',
        'beamstopdiameter': 0.1,
        'beamstopwidth': 1.0,
        'beamstop': np.array([0.0, 0.0]),
        'beamstopshift': np.array([0.0, 0.0]),
        'beamstopangle': 0.0,
        'beamstopdistance': 10.0
    }),
    (detalign_vic, {
        'ALIGNDETID': 'oxford_e_81',
        'DETZEROX': -0.229091,
        'DETZEROY': -0.471509,
        'DETZEROZ': 0.506907,
        'DETROTX': 0.569459,
        'DETROTY': -0.099205,
        'DETROTZ': -0.086215
    })
])
def test_data_integrity(content, expected_data):
    vic_file = SettingsVicFile('test.vic', content)
    for key, value in expected_data.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(vic_file[key], value)
        else:
            assert vic_file[key] == value

def test_data_modification():
    vic_file = SettingsVicFile('beamstop.vic', beamstop_vic)
    vic_file['beamstopid'] = 'new_id'
    assert vic_file['beamstopid'] == 'new_id'

def test_string_representation_and_file_writing(tmp_path):
    vic_file = SettingsVicFile('beamstop.vic', beamstop_vic)
    expected_content = "\n".join(line for line in beamstop_vic.split("\n") if not line.startswith('!'))
    vic_file.to_file(tmp_path)
    with open(tmp_path / "beamstop.vic", 'r', encoding='UTF-8') as f:
        written_content = f.read()
    assert written_content.strip() == expected_content.strip()
