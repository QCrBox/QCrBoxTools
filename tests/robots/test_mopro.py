import os
from pathlib import Path, PureWindowsPath
from textwrap import dedent
from unittest.mock import Mock, patch

import pytest

from qcrboxtools.robots.mopro import (
    MoProImportRobot,
    MoProInpFile,
    MoProRobot,
    out2inp_file,
    out2last_par_file,
    parse_out,
)
from qcrboxtools.util.wine import OptionalWineExecutor


# Test out2last_par_file function
def test_out2last_par_file():
    mock_content = "Some content\nLast molecular file : C:\\path\\to\\file.par\nMore content"
    mock_path = Mock(spec=Path)
    mock_path.read_text.return_value = mock_content

    result = out2last_par_file(mock_path)
    assert isinstance(result, PureWindowsPath)
    assert str(result) == "C:\\path\\to\\file.par"


def test_out2last_par_file_no_match():
    mock_content = "Some content without a match"
    mock_path = Mock(spec=Path)
    mock_path.read_text.return_value = mock_content

    result = out2last_par_file(mock_path)
    assert result is None


@pytest.fixture(name="mock_out_file")
def fixture_mock_out_file(tmp_path):
    mock_path = tmp_path / "mopro.out"
    mock_path.write_text(
        dedent(r"""
        Working Directory : Y:\workdir\\
        =============================================================
        ============  MoPro v22                 ===============
        ============   Job starting at :  26/06/2024   14:36:15
        =============================================================

        ===================================================
        List of commands in MoPro input file :  Y:\workdir\\mopro.inp

        FILE PARA Y:\workdir\\work_05.par
        FILE DATA Y:\workdir\epoxide.hkl
        FILE TABL Y:\MoProDir\LibMoPro\mopro_v20.tab
        FILE WAVE Y:\MoProDir\LibMoPro\WAVEF
        FILE ANOM Y:\MoProDir\LibMoPro\asf_Kissel.dat
        AUTO REFI
        WRIT CIFM
        ===================================================

        -------------------------------------------------------------------------------
        |--> Applying : FILE PARA Y:\workdir\\work_05.par   |
        -------------------------------------------------------------------------------
        Opened PARA   File : Y:\workdir\\work_05.par
        reading Molecular Parameters input file: work_05.par

        ... (more content)

        Estimate of average error on the electron density map
        sigma(delta_rho) =  0.0520  e/A3
        Rees, B. (1976). Acta Cryst. A32, 483-488
        sigma(delta_rho) = 2/V  [ sum(k.Fo-Fc)**2 ] **1/2
        computed with sin(Theta)/lambda limits:    0.098    0.995  A-1
        Max |Fo-Fc|      1.67  for:  -1   1   0 k*Yo k*Yc Fo Fc    687.51    778.67     25.62     27.26

        =============================================================
        ----------- End  of MoPro job :  26/06/2024      14:36:36
        ----------- Total CPU time needed :       0 min   15.6875 s

        --- Writing new output file : mopro_010.out
        Last molecular file : Y:\workdir\\work_10.par
        =============================================================



        Check for  9  WARNINGs in 'mopro.out'
    """)
    )
    return mock_path


def test_out2inp_file(mock_out_file):
    result = out2inp_file(mock_out_file)
    assert isinstance(result, MoProInpFile)
    assert result.files["PARA"] == Path(r"Y:\workdir\\work_05.par")
    assert result.files["DATA"] == Path(r"Y:\workdir\epoxide.hkl")
    assert result.body.strip() == "AUTO REFI\nWRIT CIFM"


def test_parse_out(mock_out_file):
    inp_file, last_par_file = parse_out(mock_out_file)

    assert isinstance(inp_file, MoProInpFile)
    assert inp_file.files["PARA"] == Path(r"Y:\workdir\\work_05.par")
    assert inp_file.body.strip() == "AUTO REFI\nWRIT CIFM"
    assert last_par_file == r"work_10.par"


def test_parse_out_no_par(tmp_path):
    mock_out_file = tmp_path / "mopro.out"
    mock_out_file.write_text(
        dedent(r"""
        Working Directory : Y:\workdir\\
        =============================================================
        ============  MoPro v22                 ===============
        ============   Job starting at :  26/06/2024   14:36:15
        =============================================================

        ===================================================
        List of commands in MoPro input file :  Y:\workdir\\mopro.inp

        FILE PARA Y:\workdir\\work_05.par
        FILE DATA Y:\workdir\epoxide.hkl
        FILE TABL Y:\MoProDir\LibMoPro\mopro_v20.tab
        FILE WAVE Y:\MoProDir\LibMoPro\WAVEF
        FILE ANOM Y:\MoProDir\LibMoPro\asf_Kissel.dat
        AUTO REFI
        WRIT CIFM
        ===================================================
        """),
        encoding="UTF-8",
    )
    inp_file, last_par_file = parse_out(mock_out_file)
    assert inp_file.body.strip() == "AUTO REFI\nWRIT CIFM"
    assert last_par_file is None


@pytest.fixture
def mock_executor():
    return Mock(spec=OptionalWineExecutor)


def test_mopro_import_robot_init(mock_executor, tmp_path):
    mock_executable = tmp_path / "imopro"
    mock_executable.touch()
    with patch.dict(os.environ, {"IMOPRO_PATH": str(mock_executable)}):
        robot = MoProImportRobot(executor=mock_executor)
    assert str(robot.executable_path) == str(mock_executable)


def test_mopro_import_robot_init_missing_env_var():
    with patch.dict(os.environ, clear=True):
        with pytest.raises(ValueError) as excinfo:
            MoProImportRobot()
        assert "IMOPRO_PATH environment variable not set" in str(excinfo.value)


def test_mopro_import_robot_init_nonexistent_path():
    with patch.dict(os.environ, {"IMOPRO_PATH": str(Path("/nonexistent/path"))}):
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError) as excinfo:
                MoProImportRobot()
            assert f"No executable not found at {str(Path('/nonexistent/path'))}" in str(excinfo.value)


def test_mopro_import_robot_cif2par(mock_executor, tmp_path):
    executable_path = tmp_path / "imopro"
    executable_path.touch()
    robot = MoProImportRobot(executor=mock_executor, executable_path=executable_path)
    cif_path = Path("/path/to/input.cif")

    result = robot.cif2par(cif_path)

    mock_executor.execute.assert_called_once_with(
        [str(executable_path), f"%{cif_path.absolute()}"], cwd=cif_path.parent
    )
    assert result == cif_path.with_name(cif_path.stem + "_00.par")


def test_mopro_inp_file_init():
    inp_file = MoProInpFile(files={"PARA": Path("file.par"), "DATA": Path("data.hkl")}, body="AUTO REFI\nWRIT CIFM")
    assert inp_file.files["PARA"] == Path("file.par")
    assert inp_file.files["DATA"] == Path("data.hkl")
    assert inp_file.body == "AUTO REFI\nWRIT CIFM"


def test_mopro_inp_from_string():
    inp_file = MoProInpFile.from_string(
        dedent(r"""
        FILE PARA file.par
        FILE DATA data.hkl
        AUTO REFI
        WRIT CIFM
    """)
    )
    assert inp_file.files["PARA"] == Path("file.par")
    assert inp_file.files["DATA"] == Path("data.hkl")
    assert inp_file.body == "AUTO REFI\nWRIT CIFM"


def test_mopro_inp_from_file(tmp_path):
    inp_path = tmp_path / "mopro.inp"
    inp_path.write_text(
        dedent(r"""
        FILE PARA file.par
        FILE DATA data.hkl
        AUTO REFI
        WRIT CIFM
    """)
    )
    inp_file = MoProInpFile.from_file(inp_path)
    assert inp_file.files["PARA"] == Path("file.par")
    assert inp_file.files["DATA"] == Path("data.hkl")
    assert inp_file.body == "AUTO REFI\nWRIT CIFM"


def test_mopro_inp_write(tmp_path):
    inp_file = MoProInpFile(files={"PARA": Path("file.par"), "DATA": Path("data.hkl")}, body="AUTO REFI\nWRIT CIFM")
    inp_path = tmp_path / "mopro.inp"
    inp_file.write(inp_path)
    assert inp_path.read_text() == dedent("""\
        FILE PARA file.par
        FILE DATA data.hkl
        AUTO REFI
        WRIT CIFM
    """)


def test_mopro_robot_init(mock_executor, tmp_path):
    mock_executable = tmp_path / "mopro"
    mock_executable.touch()
    with patch.dict(os.environ, {"MOPRO_PATH": str(mock_executable)}):
        robot = MoProRobot(executor=mock_executor)
    assert str(robot.executable_path) == str(mock_executable)


def test_mopro_robot_init_missing_env_var():
    with patch.dict(os.environ, clear=True):
        with pytest.raises(ValueError) as excinfo:
            MoProRobot()
        assert "MOPRO_PATH environment variable not set" in str(excinfo.value)


def test_mopro_robot_init_nonexistent_path():
    with patch.dict(os.environ, {"MOPRO_PATH": str(Path("/nonexistent/path"))}):
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError) as excinfo:
                MoProRobot()
            assert f"No executable not found at {str(Path('/nonexistent/path'))}" in str(excinfo.value)


def test_mopro_robot_run_file(mock_executor, tmp_path, mock_out_file):
    executable_path = tmp_path / "mopro"
    executable_path.touch()
    robot = MoProRobot(executor=mock_executor, executable_path=executable_path)
    file_path = Path(tmp_path / "mopro_00.inp")
    mock_out_file.touch()

    robot.run_file(file_path)

    mock_executor.execute.assert_called_once_with([str(executable_path), str(file_path)])
