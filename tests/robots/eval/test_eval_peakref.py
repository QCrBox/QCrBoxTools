import textwrap

import pytest
from qcrboxtools.robots.eval import EvalPeakrefRobot, RmatFile

mock_peakref_output = textwrap.dedent("""\
    Refining 11 parameters: 226 evaluations in 140 iterations. residue 0.07535
    Refining 11 parameters: 187 evaluations in 119 iterations. residue 0.07518
    Refining 11 parameters: 114 evaluations in 71 iterations. residue 0.07512
    One matrix. 420395 reflections. (forbidden mm:1 weak:43 total:44)
    Used: 420351 mm 91845 rot
            ref   current  previous   change   initial   change   shift
    =====================================================================
    a         Yes  57.68429  57.68800 -0.00370  57.69090 -0.00661 0.28845
    b         Con  57.68429                     57.69084          0.28845
    c         Yes 149.68896 149.69516 -0.00620 149.68208  0.00688 0.74841
    alpha     Fix  90.00000                     89.99943          1.00000
    beta      Fix  90.00000                     89.99971          1.00000
    gamma     Fix  90.00000                     89.99989          1.00000
    orx       Yes   0.60693   0.60695 -0.00002   0.60687  0.00006 0.00426
    ory       Yes  -0.67102  -0.67102 -0.00001  -0.67112  0.00010 0.00426
    ora       Yes 173.03036 173.02960  0.00076 173.04532 -0.01496 1.00000
    xtalx      No   0.00000                      0.00000          0.10000
    xtaly      No   0.00000                      0.00000          0.10000
    xtalz     Fix   0.00000                      0.00000          0.10000
    zerodist  Yes  -0.00999   0.00000 -0.00999   0.00000 -0.00999 0.10000
    zerohor   Yes   0.05432   0.06053 -0.00621   0.00000  0.05432 0.10000
    zerover   Yes   0.06026   0.06670 -0.00644   0.00000  0.06026 0.10000
    detrotx   Yes   0.00696   0.00890 -0.00194   0.00000  0.00696 0.20000
    detroty   Yes  -0.00445  -0.00654  0.00209   0.00000 -0.00445 0.20000
    detrotz   Yes  -0.00256  -0.00055 -0.00202   0.00000 -0.00256 0.20000
    =====================================================================
    Vol           498086.63 498171.31   -84.69 498177.34   -90.72  420351
    =====================================================================
    mm              0.06754   0.06740  0.00014   0.11092 -0.04339  420351
    mmAng       +   0.03339   0.03332  0.00007   0.05483 -0.02145  420351
    rotpartial  +   0.04173   0.04148  0.00025   0.04906 -0.00733   91845
    rotoutside      0.00000   0.00000  0.00000   0.12527 -0.12527       3
    rotinside       0.00000   0.00000  0.00000   0.00000  0.00000  328503
    rotall          0.00912   0.00906  0.00005   0.01072 -0.00160  420351
    res             0.07512   0.07480  0.00032   0.10389 -0.02878
    b constrained to [a]. alpha remains fixed. beta remains fixed. gamma remains fixed.
    Calculating sigma's using sigrnd.
    50 cycles of refinement using a random subset (fraction=0.1 n=42039) of all 420395 reflections.
    res=0.07511 nref(mm)=420351
                a         c     Volume
    refined 57.68429 149.68896 498086.625
    sigma    0.00026   0.00299      7.000
    420351 reflections. Resolution(d) from 45.69169 to 1.6 Angstrom. Theta from 0.93299 to 27.71021 degrees.
""")

mock_rmat = textwrap.dedent("""\
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
""")

@pytest.fixture(name='robot')
def fixture_robot(tmp_path):
    rmat_file = tmp_path / "example.rmat"
    rmat_file.write_text(mock_rmat)
    return EvalPeakrefRobot(tmp_path, rmat_file)

def test_init(robot, tmp_path):
    assert robot.work_folder == tmp_path
    assert isinstance(robot.rmat_file, RmatFile)

def test_cell_cif_from_log(robot, tmp_path):
    # Create a mock peakref_output.log file with sample data
    log_file = tmp_path / 'peakref_output.log'
    log_file.write_text(mock_peakref_output)

    # Test the method
    result = robot.cell_cif_from_log()

    # Add assertions here to verify the returned dictionary
    assert result['_cell.length_a'] == result['_cell.length_b']
    assert result['_cell.angle_gamma_su'] == 0.0
    assert result['_cell.length_a'] == 57.68429
    assert result['_cell.volume_su'] == 7.0
