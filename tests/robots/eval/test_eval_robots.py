# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

import textwrap
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from qcrboxtools.robots.eval import (
    Eval15AllRobot,
    EvalAnyRobot,
    EvalBuilddatcolRobot,
    EvalBuildeval15Robot,
    EvalPeakrefRobot,
    EvalScandbRobot,
    EvalViewRobot,
    PicFile,
    RmatFile,
)
from qcrboxtools.robots.eval.eval_robots import EvalBaseRobot


def mock_for_subprocess_call_factory(
    expected_program_name: str, expected_init_content: str = None, raise_os_error: bool = False
):
    """
    Factory function to flexibly create mocked subprocess.call functions for the Eval components
    """

    def mocked_subprocess_call(
        program_name: str,
        cwd: Path,
        *args,
        shell: bool = False,
        **kwargs,
    ):
        """
        Instead of calling the program, check if init file was created.
        """
        init_file = Path(cwd) / f"{program_name}.init"
        assert (
            program_name == expected_program_name
        ), f"Expected program name {expected_program_name}, got {program_name}"
        assert init_file.exists(), f"Init file {init_file} was not created."
        if expected_init_content is not None:
            assert (
                init_file.read_text(encoding="UTF-8") == expected_init_content
            ), "Init file content does not match the expected value."
        if raise_os_error and not shell:
            raise OSError("Mocked OS error")

    mock = Mock(side_effect=mocked_subprocess_call)
    return mock


# Test EvalBaseRobot


@pytest.fixture(name="base_robot")
def fixture_base_robot(tmp_path):
    return EvalBaseRobot(tmp_path)


def test_base_init(base_robot, tmp_path):
    assert base_robot.work_folder == tmp_path

    base_robot.work_folder = tmp_path / "subfolder"

    assert base_robot.work_folder == tmp_path / "subfolder"


def test_base_run_program_with_commands(base_robot, tmp_path):
    # Mock the subprocess.call function
    program_name = "test"

    mock = mock_for_subprocess_call_factory(
        expected_program_name=program_name, expected_init_content="command1\ncommand2\n", raise_os_error=False
    )

    with patch("subprocess.call", mock):
        base_robot._run_program_with_commands("test", ["command1", "command2"])

    assert mock.called

    # test if shell is used when an OSError is raised
    mock_raise = mock_for_subprocess_call_factory(
        expected_program_name=program_name, expected_init_content="command1\ncommand2\n", raise_os_error=True
    )

    with patch("subprocess.call", mock_raise):
        base_robot._run_program_with_commands("test", ["command1", "command2"])

    assert mock_raise.call_count == 2

    # test if existing init file is kept
    (tmp_path / "test.init").write_text("existing content", encoding="UTF-8")

    with patch("subprocess.call", mock):
        base_robot._run_program_with_commands("test", ["command1", "command2"])

    assert (tmp_path / "test.init").read_text(encoding="UTF-8") == "existing content"


@pytest.fixture(name="pic_file")
def fixture_pic_file():
    pic_file = PicFile("test.pic", "command\ncommand")
    pic_file.to_file = lambda path: None  # remove output as it will not be used
    return pic_file


# Test Eval15AllRobot
@pytest.fixture(name="robot15")
def fixture_robot15(tmp_path, pic_file):
    file_list = [pic_file]
    return Eval15AllRobot(tmp_path, file_list)


def test_eval15_init(robot15, tmp_path):
    assert robot15.work_folder == tmp_path


def test_integrate_shoes(robot15, tmp_path):
    # Mock the subprocess.call function
    mock = mock_for_subprocess_call_factory(
        expected_program_name="eval15all", expected_init_content="\n\n\n\n\n\n\n\n\n\n", raise_os_error=False
    )

    with patch("subprocess.call", mock):
        robot15.integrate_shoes()

    assert mock.called

    # test that the additional dependency eval15.init file is retained

    (tmp_path / "eval15.init").write_text("existing eval content", encoding="UTF-8")

    with patch("subprocess.call", mock):
        robot15.integrate_shoes()

    assert (tmp_path / "eval15.init").read_text(encoding="UTF-8") == "existing eval content"


# Test EvalViewRobot
@pytest.fixture(name="robot_view")
def fixture_robot_view(tmp_path, pic_file):
    file_list = [pic_file]
    return EvalViewRobot(tmp_path, file_list)


def test_view_init(robot_view, tmp_path):
    assert robot_view.work_folder == tmp_path


def test_view_create_shoes(robot_view):
    # Mock the subprocess.call function
    mock = mock_for_subprocess_call_factory(
        expected_program_name="view", expected_init_content="@datcol\nexit\n", raise_os_error=False
    )

    with patch("subprocess.call", mock):
        robot_view.create_shoes()

    assert mock.called


# Test EvalAnyRobot

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


@pytest.fixture(name="any_robot")
def fixture_any_robot(tmp_path):
    return EvalAnyRobot(tmp_path)


def test_any_init(any_robot, tmp_path):
    assert any_robot.work_folder == tmp_path


@pytest.fixture(name="any_abs_mock")
def fixture_any_abs_mock(any_robot):
    # Mock the subprocess.call function
    mock = mock_for_subprocess_call_factory("any", expected_init_content="read final\nsadabs\nexit\n")
    return mock


def test_any_create_abs(any_robot, any_abs_mock):
    # Mock the subprocess.call function
    with patch("subprocess.call", any_abs_mock):
        any_robot.create_abs()

    # Check if the expected file is created
    assert (any_robot.work_folder / "any_output.log").exists()


def test_any_create_cif_dict(any_robot, any_abs_mock, tmp_path):
    # Write mock_sad_content to a temporary file
    sad_path = tmp_path / "shelx.sad"
    with open(sad_path, "w", encoding="UTF-8") as f:
        f.write(mock_sad_content)

    # Execute the method and validate the output
    with patch("subprocess.call", any_abs_mock):
        result = any_robot.create_cif_dict()

    # Verify the returned dictionary
    assert result["_diffrn_refln.index_h"][0] == 0
    assert result["_diffrn_refln.index_k"][1] == 6
    assert result["_diffrn_refln.index_l"][2] == 6
    assert result["_diffrn_refln.intensity_net"][3] == 2382.12
    assert result["_diffrn_refln.intensity_net_su"][4] == 27.14
    assert result["_diffrn_refln.class_code"][4] == 1
    assert result["_qcrbox.diffrn_refln.direction_cosine_incid_x"][6] == -0.81282
    assert result["_qcrbox.diffrn_refln.direction_cosine_incid_y"][7] == 0.93012
    assert result["_qcrbox.diffrn_refln.direction_cosine_incid_z"][8] == -0.20880
    assert result["_qcrbox.diffrn_refln.direction_cosine_diffrn_x"][9] == 0.36540
    assert result["_qcrbox.diffrn_refln.direction_cosine_diffrn_y"][10] == 0.54315
    assert result["_qcrbox.diffrn_refln.direction_cosine_diffrn_z"][10] == -0.04109
    assert result["_qcrbox.diffrn_refln.detector_px_x_obs"][9] == 331.44
    assert result["_qcrbox.diffrn_refln.detector_px_y_obs"][8] == 121.06
    assert result["_qcrbox.diffrn_refln.detector_frame_obs"][7] == 2.53
    assert result["_qcrbox.diffrn_refln.evalsad_mystery_val1"][6] == -18.62
    assert result["_qcrbox.diffrn_refln.evalsad_mystery_val2"][5] == 4047


def test_any_create_cif_file(any_robot, any_abs_mock, tmp_path):
    # Write mock_sad_content to a temporary file
    sad_path = tmp_path / "shelx.sad"
    with open(sad_path, "w", encoding="UTF-8") as f:
        f.write(mock_sad_content)

    file_path = tmp_path / "output.cif"
    with patch("subprocess.call", any_abs_mock):
        any_robot.create_cif_file(file_path)

    assert file_path.exists()

    content = file_path.read_text(encoding="UTF-8")

    assert "_diffrn_refln.index_h" in content
    assert "_diffrn_refln.index_k" in content
    assert "_diffrn_refln.index_l" in content


def test_any_create_pk(any_robot):
    # Mock the subprocess.call function
    mock = mock_for_subprocess_call_factory("any", expected_init_content="read final\npkrestfrac 0.2\npk\nexit\n")

    with patch("subprocess.call", mock):
        any_robot.create_pk()

    # Check if the expected file is created
    assert (any_robot.work_folder / "any_output.log").exists()


# Test EvalPeakrefRobot

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


@pytest.fixture(name="peakref_robot", params=[True, False])
def fixture_peakref_robot(request, tmp_path):
    if request.param:
        rmat_file = tmp_path / "example.rmat"
        rmat_file.write_text(mock_rmat)
    else:
        rmat_file = RmatFile("example.rmat", mock_rmat)
    return EvalPeakrefRobot(tmp_path, rmat_file)


def test_peakref_init(peakref_robot, tmp_path):
    assert peakref_robot.work_folder == tmp_path
    assert isinstance(peakref_robot.rmat_file, RmatFile)


def test_peakref_cell_cif_from_log(peakref_robot, tmp_path):
    with pytest.raises(FileNotFoundError):
        peakref_robot.cell_cif_from_log()
    # Create a mock peakref_output.log file with sample data
    log_file = tmp_path / "peakref_output.log"
    log_file.write_text(mock_peakref_output)

    # Test the method
    result = peakref_robot.cell_cif_from_log()

    # Add assertions here to verify the returned dictionary
    assert result["_cell.length_a"] == result["_cell.length_b"]
    assert result["_cell.angle_gamma_su"] == 0.0
    assert result["_cell.length_a"] == 57.68429
    assert result["_cell.volume_su"] == 7.0


@pytest.mark.parametrize("reuse_rmat", [True, False])
@pytest.mark.parametrize("explicit_strategy", [True, False])
@patch("qcrboxtools.robots.eval.EvalPeakrefRobot._run_program_with_commands")
@patch("os.remove")
def test_refine_parameters(mock_remove, mock_run_program, explicit_strategy, reuse_rmat, peakref_robot, tmp_path):
    peakfile_path = "peakfile"
    if explicit_strategy:
        refinement_strategy = (("zerohor", "zerover"), ("rmat",), ("detrot",), ("zerodist",))
    else:
        refinement_strategy = "default"
    point_group_tolerance = (0.1, 0.2)
    end_with_cell = True
    rewrite_detalign = True
    rewrite_goniostat = True
    if reuse_rmat:
        new_rmat_filename = "new.rmat"
        new_rmat_file = tmp_path / new_rmat_filename
        new_rmat_file.write_text(mock_rmat)
        peakref_robot.rmat_file = RmatFile(new_rmat_filename, mock_rmat)
        new_rmat_filename = None
    else:
        new_rmat_filename = "new.rmat"
        new_rmat_file = tmp_path / new_rmat_filename
        new_rmat_file.write_text(mock_rmat)

    peakref_robot.refine_parameters(
        peakfile_path,
        refinement_strategy,
        point_group_tolerance,
        end_with_cell,
        new_rmat_filename,
        rewrite_detalign,
        rewrite_goniostat,
    )

    expected_commands = [
        "rmat transfer.rmat",
        "pk peakfile",
        "fix all",
        "free zerohor",
        "free zerover",
        "gox",
        "free rmat",
        "gox",
        "free detrot",
        "gox",
        "free zerodist",
        "gox",
        "pgzero 0.1 0.2",
        "gox",
        "reind",
        "gox",
        "fix all",
        "free cell",
        "sigrnd 0.1 50",
        "save detalign.vic",
        "savegonio goniostat.vic",
        f"savermat {peakref_robot.rmat_file.filename}",
        "exit",
    ]

    mock_run_program.assert_called_once_with("peakref", expected_commands)
    mock_remove.assert_called_once_with(tmp_path / "transfer.rmat")
    if not reuse_rmat:
        assert peakref_robot.rmat_file.filename == new_rmat_filename


def test_peakref_folder_to_cif(peakref_robot, tmp_path):
    # Create a mock peakref_output.log file with sample data
    log_file = tmp_path / "peakref_output.log"
    log_file.write_text(mock_peakref_output)

    # Test the method
    peakref_robot.folder_to_cif("peakref_output.cif")

    # Check if the expected file is created
    assert (tmp_path / "peakref_output.cif").exists()

    content = (tmp_path / "peakref_output.cif").read_text(encoding="UTF-8")

    assert "_cell.length_a" in content
    assert "_cell.length_b" in content
    assert "_cell.length_c" in content
    assert "_cell.angle_alpha" in content
    assert "_cell.angle_beta" in content
    assert "_cell.angle_gamma" in content
    assert "_cell.volume" in content
    assert "_cell.volume_su" in content


# Test EvalScandbRobot
def test_scandb_run(tmp_path):
    mock = mock_for_subprocess_call_factory("scandb", "\n", False)
    robot = EvalScandbRobot(tmp_path)

    view_init_path = tmp_path / "view.init"
    view_init_path.write_text("Test Content", encoding="UTF-8")

    with patch("subprocess.call", mock):
        robot.run()

    assert mock.called
    assert view_init_path.read_text(encoding="UTF-8") == "Test Content"


# Test BuilddatcolRobot
@pytest.fixture(name="builddatcol_robot")
def fixture_builddatcol_robot(tmp_path):
    return EvalBuilddatcolRobot(tmp_path)


# TODO Complete
def test_builddatcol_create_datcol_files(builddatcol_robot, tmp_path):
    # Mock the subprocess.call function
    mock_builddatcol = mock_for_subprocess_call_factory("builddatcol", expected_init_content="\n\n\n\n\n\n\n\n\n\n")
    mock_scandb = Mock(side_effect=lambda *args: None)

    # entry, value, search_string
    test_data = [
        ("rmat_file", RmatFile("example.rmat", mock_rmat), "rmat example.rmat"),
        ("minimum_res", 9.6, "resomin 9.6"),
        ("maximum_res", 0.84, "resomax 0.84"),
        ("box_size", 0.1, "boxsizemm 0.1"),
        ("box_depth", 5, "boxdepth 5"),
        ("maximum_duration", 6, "durationmax 6"),
        ("min_refln_in_box", 3, "boxrefl 3"),
    ]

    test_kws = {key: value for key, value, _ in test_data}
    with patch("subprocess.call", mock_builddatcol), patch("qcrboxtools.robots.eval.EvalScandbRobot.run", mock_scandb):
        builddatcol_robot.create_datcol_files(**test_kws)

    assert mock_builddatcol.called
    assert mock_scandb.called

    vic_path = tmp_path / "datcolsetup.vic"
    vic_content = vic_path.read_text(encoding="UTF-8")

    for _, _, search_string in test_data:
        assert search_string in vic_content

    mock_scandb.reset_mock()

    # Check that scandb not called if scaninfo.txt exists
    (tmp_path / "scaninfo.txt").write_text("Test Content", encoding="UTF-8")
    with patch("subprocess.call", mock_builddatcol), patch("qcrboxtools.robots.eval.EvalScandbRobot.run", mock_scandb):
        builddatcol_robot.create_datcol_files(**test_kws)

    assert mock_builddatcol.called
    assert not mock_scandb.called

    # Check that ValueError raised if minimum_res < maximum_res

    with pytest.raises(ValueError):
        wrong_test_kws = {key: value for key, value, _ in test_data}
        wrong_test_kws["minimum_res"] = 0.83
        builddatcol_robot.create_datcol_files(**wrong_test_kws)


mock_datcol_output = textwrap.dedent("""\
    ! Created by builddatcol at 3-Apr-2024 17:00:52
    ! Host 68c97441bae5 User WD /mnt/qcrbox/shared_files/examples_eval/run_integrate/
    ! abort on warnings
    abort on
    badpixel on
    rmat ic.rmat
    ! load scan specific rmat if it exists
    \if file 'scan'.rmat rmat 'scan'.rmat
    ! scandependent beamstop
    &beamstop'scan'.vic
    ! scandependent goniostat
    &goniostat'scan'.vic
    resomin 50.0
    resomax 0.79
    boxsizemm 1.2
    boxdepth 5
    durationmax 5.0
    boxrefl 1000
    ! display only
    datcolboxes 1
    output none
""")


def test_builddatcol_extract_vars(builddatcol_robot, tmp_path):
    # Create a mock datcolsetup.vic file with sample data
    vic_file = tmp_path / "datcolsetup.vic"
    vic_file.write_text(mock_datcol_output)

    # Test the method
    result = builddatcol_robot.extract_vars()

    # Add assertions here to verify the returned dictionary
    assert result["minimum_res"] == 50.0
    assert result["maximum_res"] == 0.79
    assert result["box_size"] == 1.2
    assert result["box_depth"] == 5
    assert result["maximum_duration"] == 5.0
    assert result["min_refln_in_box"] == 1000

    # Check that KeyError is raised if essential line missing
    vic_file.write_text(mock_datcol_output.replace("resomin 50.0", ""), encoding="UTF-8")

    with pytest.raises(KeyError):
        builddatcol_robot.extract_vars()


# test EvalBuildEval15Robot


@pytest.fixture(name="robot_buildeval15")
def fixture_robot_buildeval15(tmp_path):
    return EvalBuildeval15Robot(tmp_path)


def test_evalbuildeval15_init(tmp_path, robot_buildeval15):
    assert robot_buildeval15.work_folder == tmp_path


def test_evalbuildeval15_run(robot_buildeval15, tmp_path):
    # Mock the subprocess.call function
    mock_nop4p = mock_for_subprocess_call_factory(
        "buildeval15", expected_init_content="tube\nnone\n0.8\n2.0\n0.2\n0.3\n"
    )
    with patch("subprocess.call", mock_nop4p):
        robot_buildeval15.run(
            focus_type="tube",
            polarisation_type=None,
            pointspread_gamma=0.8,
            acdnoise=2.0,
            crystal_dimension=0.2,
            mosaic=0.3,
        )

    assert mock_nop4p.called

    mock_p4p = mock_for_subprocess_call_factory(
        "buildeval15", expected_init_content="rotating\nparallel\n0.8\n2.0\n0.3\n"
    )

    (tmp_path / "test.p4p").touch()

    with patch("subprocess.call", mock_p4p):
        with pytest.warns(UserWarning):
            robot_buildeval15.p4p_file = "test.p4p"
            robot_buildeval15.run(
                focus_type="rotating", polarisation_type="parallel", pointspread_gamma=0.8, acdnoise=2.0, mosaic=0.3
            )
    assert mock_p4p.called

    # Test that an invalid focus_type raises a ValueError
    with pytest.raises(ValueError):
        robot_buildeval15.run(focus_type="nonsense")

    # Test that an invalid polarisation_type raises a ValueError
    with pytest.raises(ValueError):
        robot_buildeval15.run(polarisation_type="nonsense")
