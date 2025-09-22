# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

"""
This test module provides functionalities to test the interactions of `Olex2Socket` with a running
Olex2 server. It contains tests to check the Olex2 server's availability and functionality of
headless refinement using the socket.
"""

import os
import shutil
import warnings
from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np
import pytest
from iotbx import cif

from qcrboxtools.robots.olex2 import Olex2Socket


@pytest.mark.program_dependent
def test_olex2server_avail():
    """
    Tests the connection to the Olex2 server using the `Olex2Socket` class. The test checks whether
    the server is ready and responsive. It expects Olex2 socket server to be started and if using
    a non-default address or port that the OLEX2SERVER and OLEX2PORT environment variables are set.
    """
    olex2 = Olex2Socket()
    message = (
        "Server did not respond with ready to status check, are environment variables OLEX2SERVER "
        + "and OLEX2PORT available and is the Olex2 socket server started?"
    )
    assert olex2.check_connection(), message


@pytest.mark.program_dependent
def test_olex2_refine_live(tmp_path):
    """
    Tests the refinement functionality of `Olex2Socket`. The function simulates a
    refinement process by copying a non-converged CIF file to a temporary working path, and
    then triggering refinement using the `Olex2Socket` class. The test ensures that the refined
    values match the target values within a specified tolerance.

    Args:
    - tmp_path: A fixture provided by pytest for temporary directories.
    """
    work_path = tmp_path / "work.cif"
    shutil.copy("./tests/robots/olex/cif_files/refine_nonconv_nonHaniso.cif", work_path)

    # create these files to check that loading works if they already exist
    work_path.with_suffix(".hkl").touch()
    work_path.with_suffix(".ins").touch()

    olex2 = Olex2Socket()
    olex2.run_full_refinement(
        work_path,
        None,
        n_cycles=8,
        refine_starts=5,
    )

    target_path = "./tests/robots/olex/cif_files/refine_conv_nonHaniso.cif"
    cif_target = cif.reader(str(target_path)).model()["epoxide"]

    cif_refined = cif.reader(str(work_path)).model()["work"]

    for ij in (11, 22, 33, 12, 13, 23):
        key = f"_atom_site_aniso_U_{ij}"
        refined_vals = np.array([float(val.split("(")[0]) for val in cif_refined[key]])
        target_vals = np.array([float(val.split("(")[0]) for val in cif_target[key]])
        assert max(abs(refined_vals - target_vals)) < 1.1e-4


@pytest.mark.program_dependent
def test_olex2_refine_tsc_live(tmp_path):
    work_path = tmp_path / "work.cif"
    shutil.copy("./tests/robots/olex/cif_files/refine_nonconv_allaniso.cif", work_path)

    input_tsc_path = Path("./tests/robots/olex/cif_files/refine_allaniso.tscb").absolute()
    shutil.copy(input_tsc_path, tmp_path / input_tsc_path.name)
    tsc_path = tmp_path / input_tsc_path.name

    olex2 = Olex2Socket()
    olex2.run_full_refinement(
        work_path,
        tsc_path,
        n_cycles=8,
        refine_starts=5,
    )

    target_path = "./tests/robots/olex/cif_files/refine_conv_allaniso.cif"
    cif_target = cif.reader(str(target_path)).model()["epoxide"]

    cif_refined = cif.reader(str(work_path)).model()["work"]

    for ij in (11, 22, 33, 12, 13, 23):
        key = f"_atom_site_aniso_U_{ij}"
        refined_vals = np.array([float(val.split("(")[0]) for val in cif_refined[key]])
        target_vals = np.array([float(val.split("(")[0]) for val in cif_target[key]])
        assert max(abs(refined_vals - target_vals)) < 1.1e-4


# dry run tests that only test behavior without connecting to Olex2


def test_olex2_init():
    olex2 = Olex2Socket()
    assert olex2.server == "localhost"
    assert olex2.port == 8899

    olex2 = Olex2Socket("test_server", 1234)
    assert olex2.server == "test_server"
    assert olex2.port == 1234


@pytest.fixture(name="olex2_env_vars")
def fixture_olex2_env_vars():
    os.environ["OLEX2SERVER"] = "another_server"
    os.environ["OLEX2PORT"] = "12345"
    yield
    del os.environ["OLEX2SERVER"]
    del os.environ["OLEX2PORT"]


def test_olex2_init_env(olex2_env_vars):
    olex2 = Olex2Socket()
    assert olex2.server == "another_server"
    assert olex2.port == 12345


def test_olex2_wait_for_completion():
    # create a mock answer generator that returns "test_response" 10 times and then "ready"
    def mock_send_input_func():
        n = 0
        while n < 10:
            n += 1
            yield "test_response"
        yield "ready"

    mock_send_input_iter = mock_send_input_func()

    mock_send_input = Mock(side_effect=lambda *args: next(mock_send_input_iter))

    olex2 = Olex2Socket()
    with patch("qcrboxtools.robots.olex2.Olex2Socket._send_input", mock_send_input):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            olex2.wait_for_completion(11, "test_input")

        assert mock_send_input.call_count == 11
        assert mock_send_input.call_args_list == [call("status\n")] * 11

        mock_send_input_iter = mock_send_input_func()
        mock_send_input.side_effect = lambda *args: next(mock_send_input_iter)
        with pytest.warns(UserWarning):
            olex2.wait_for_completion(1, "test_input")


def test_wait_for_completion_failed():
    olex2 = Olex2Socket()

    with patch("qcrboxtools.robots.olex2.Olex2Socket._send_input", lambda *args: "failed"):
        with pytest.raises(RuntimeError):
            olex2.wait_for_completion(1, "test_input")


@patch("qcrboxtools.robots.olex2.Olex2Socket.wait_for_completion")
@patch("qcrboxtools.robots.olex2.Olex2Socket._send_input")
@patch("qcrboxtools.robots.olex2.Olex2Socket.reserve_new_job")
@patch("qcrboxtools.robots.olex2.Olex2Socket.move_refined_files")
@patch("qcrboxtools.robots.olex2.Olex2Socket.create_startup_cmds")
@patch("qcrboxtools.robots.olex2.Olex2Socket.create_structure_load_cmds")
@patch("qcrboxtools.robots.olex2.Olex2Socket.create_nosphera2_cmds")
def test_olex2_send_command_orchestration(
    mock_create_nosphera2_cmds,
    mock_create_structure_load_cmds,
    mock_create_startup_cmds,
    mock_move_refined_files,
    mock_reserve_new_job,
    mock_send_input,
    mock_wait_for_completion,
    tmp_path,
):
    """Test that send_command orchestrates all steps in correct order."""
    # Setup mock returns
    mock_create_startup_cmds.return_value = ["startup command"]
    mock_create_structure_load_cmds.return_value = ["load command"]
    mock_create_nosphera2_cmds.return_value = ["nosphera command"]

    # Create test files and log
    cif_path = tmp_path / "test.cif"
    cif_path.touch()
    tsc_path = tmp_path / "aspherical.tscb"
    tsc_path.touch()
    log_path = tmp_path / "task_0.log"
    log_path.write_text("test_log_content")

    olex2 = Olex2Socket()
    result = olex2.send_command(cif_path, tsc_path, "custom_input")

    # Check that task counter increments
    assert olex2._task_id_counter.__repr__() == "count(1)"

    # Check that all orchestration steps are called in order
    mock_move_refined_files.assert_called_once_with(cif_path.absolute())
    mock_create_startup_cmds.assert_called_once_with(cif_path.absolute(), 0)
    mock_create_structure_load_cmds.assert_called_once_with(cif_path.absolute())
    mock_create_nosphera2_cmds.assert_called_once_with(tsc_path)
    mock_reserve_new_job.assert_called_once_with("task_0")
    mock_wait_for_completion.assert_called_once()

    # Check that log content is returned
    assert result == "test_log_content"


@patch("qcrboxtools.robots.olex2.Olex2Socket.wait_for_completion")
@patch("qcrboxtools.robots.olex2.Olex2Socket._send_input")
@patch("qcrboxtools.robots.olex2.Olex2Socket.reserve_new_job")
@patch("qcrboxtools.robots.olex2.Olex2Socket.move_refined_files")
@patch("qcrboxtools.robots.olex2.Olex2Socket.create_startup_cmds")
@patch("qcrboxtools.robots.olex2.Olex2Socket.create_structure_load_cmds")
@patch("qcrboxtools.robots.olex2.Olex2Socket.create_nosphera2_cmds")
def test_olex2_send_command_task_id_increment(
    mock_create_nosphera2_cmds,
    mock_create_structure_load_cmds,
    mock_create_startup_cmds,
    mock_move_refined_files,
    mock_reserve_new_job,
    mock_send_input,
    mock_wait_for_completion,
    tmp_path,
):
    """Test that task IDs increment correctly across multiple calls."""
    # Setup mocks
    mock_create_startup_cmds.return_value = ["startup"]
    mock_create_structure_load_cmds.return_value = ["load"]
    mock_create_nosphera2_cmds.return_value = ["nosphera"]

    cif_path = tmp_path / "test.cif"
    cif_path.touch()

    # Create log files for both calls
    (tmp_path / "task_0.log").write_text("first call")
    (tmp_path / "task_1.log").write_text("second call")

    olex2 = Olex2Socket()

    # First call
    result1 = olex2.send_command(cif_path, None, "input1")
    assert result1 == "first call"
    mock_create_startup_cmds.assert_called_with(cif_path.absolute(), 0)
    mock_reserve_new_job.assert_called_with("task_0")

    # Second call
    result2 = olex2.send_command(cif_path, None, "input2")
    assert result2 == "second call"
    # Check that task ID incremented
    assert mock_create_startup_cmds.call_args_list[-1][0][1] == 1  # Second call uses job_id=1
    assert mock_reserve_new_job.call_args_list[-1][0][0] == "task_1"  # Second call uses task_1


@patch("qcrboxtools.robots.olex2.Olex2Socket.wait_for_completion")
@patch("qcrboxtools.robots.olex2.Olex2Socket._send_input")
@patch("qcrboxtools.robots.olex2.Olex2Socket.reserve_new_job")
@patch("qcrboxtools.robots.olex2.Olex2Socket.move_refined_files")
@patch("qcrboxtools.robots.olex2.Olex2Socket.create_startup_cmds")
@patch("qcrboxtools.robots.olex2.Olex2Socket.create_structure_load_cmds")
@patch("qcrboxtools.robots.olex2.Olex2Socket.create_nosphera2_cmds")
def test_olex2_send_command_no_log(
    mock_create_nosphera2_cmds,
    mock_create_structure_load_cmds,
    mock_create_startup_cmds,
    mock_move_refined_files,
    mock_reserve_new_job,
    mock_send_input,
    mock_wait_for_completion,
    tmp_path,
):
    """Test send_command when log file doesn't exist."""
    # Setup mocks
    mock_create_startup_cmds.return_value = ["startup"]
    mock_create_structure_load_cmds.return_value = ["load"]
    mock_create_nosphera2_cmds.return_value = ["nosphera"]

    cif_path = tmp_path / "test.cif"
    cif_path.touch()
    # Note: no log file created

    olex2 = Olex2Socket()
    result = olex2.send_command(cif_path, None, "test_input")

    # Should return the default message when log file not found
    assert result == "No log file found."


@patch("qcrboxtools.robots.olex2.Olex2Socket._send_input", new_callable=lambda: Mock(return_value="ready"))
def test_check_connection(mock_send_input):
    olex2 = Olex2Socket()
    assert olex2.check_connection()
    assert mock_send_input.call_args_list == [call("status")]


@patch("qcrboxtools.robots.olex2.Olex2Socket._send_input")
def test_shutdown(mock_send_input):
    olex2 = Olex2Socket()
    olex2._shutdown_server()
    assert mock_send_input.call_args_list == [call("stop")]


def test_move_refined_files(tmp_path):
    """Test that move_refined_files correctly handles existing files."""
    olex2 = Olex2Socket()

    # Create test files
    test_cif = tmp_path / "test.cif"
    test_ins = tmp_path / "test.ins"
    test_hkl = tmp_path / "test.hkl"

    test_cif.touch()
    test_ins.write_text("test ins content")
    test_hkl.write_text("test hkl content")

    # Run the method
    olex2.move_refined_files(test_cif)

    # Check implementation behavior: files should be moved, not copied
    assert not test_ins.exists()
    assert not test_hkl.exists()

    # Check that moved files exist with expected naming pattern
    moved_ins = tmp_path / "test_moved.ins"
    moved_hkl = tmp_path / "test_moved.hkl"

    assert moved_ins.exists()
    assert moved_hkl.exists()
    assert moved_ins.read_text() == "test ins content"
    assert moved_hkl.read_text() == "test hkl content"


def test_move_refined_files_no_files(tmp_path):
    """Test that move_refined_files handles missing files gracefully."""
    olex2 = Olex2Socket()
    test_cif = tmp_path / "test.cif"
    test_cif.touch()

    # Should not raise an error when files don't exist
    olex2.move_refined_files(test_cif)


def test_move_refined_files_partial_files(tmp_path):
    """Test move_refined_files when only some files exist."""
    olex2 = Olex2Socket()

    test_cif = tmp_path / "test.cif"
    test_ins = tmp_path / "test.ins"
    # Note: no .hkl file

    test_cif.touch()
    test_ins.write_text("only ins content")

    olex2.move_refined_files(test_cif)

    # Should move the existing file
    assert not test_ins.exists()
    moved_ins = tmp_path / "test_moved.ins"
    assert moved_ins.exists()
    assert moved_ins.read_text() == "only ins content"

    # Should not create .hkl files where none existed
    moved_hkl = tmp_path / "test_moved.hkl"
    assert not moved_hkl.exists()


def test_create_structure_load_cmds():
    """Test that create_structure_load_cmds includes essential loading commands."""
    olex2 = Olex2Socket()
    cif_path = Path("/test/path/structure.cif")

    cmds = olex2.create_structure_load_cmds(cif_path)

    # Check that essential commands are present
    cmd_string = "\n".join(cmds)
    assert "user /test/path" in cmd_string  # Sets working directory
    assert "@reap /test/path/structure.cif" in cmd_string  # Loads the CIF
    assert "file structure.ins" in cmd_string  # Loads INS file
    assert "@export structure.hkl" in cmd_string  # Exports HKL data
    assert "@reap /test/path/structure.ins" in cmd_string  # Loads INS file


def test_create_startup_cmds():
    """Test that create_startup_cmds includes essential startup commands."""
    olex2 = Olex2Socket()
    cif_path = Path("/test/path/structure.cif")
    job_id = 42

    cmds = olex2.create_startup_cmds(cif_path, job_id)

    cmd_string = "\n".join(cmds)
    # Check for essential startup elements
    assert f"run:task_{job_id}" in cmd_string  # Sets task ID
    assert f"xlog:/test/path/task_{job_id}.log" in cmd_string  # Sets log file
    assert "user /test/path" in cmd_string  # Sets working directory
    assert f"task_{job_id}" in cmd_string  # Job ID is referenced
    assert "remote_mode true" in cmd_string  # Remote mode enabled


def test_create_nosphera2_cmds_no_tsc():
    """Test create_nosphera2_cmds disables aspherical refinement when no tsc file."""
    olex2 = Olex2Socket()

    cmds = olex2.create_nosphera2_cmds(None)

    cmd_string = "\n".join(cmds)
    # Should disable aspherical refinement
    assert "use_aspherical', False" in cmd_string
    assert "Calculate', False" in cmd_string


def test_create_nosphera2_cmds_with_tsc():
    """Test create_nosphera2_cmds enables aspherical refinement with tsc file."""
    olex2 = Olex2Socket()
    tsc_path = Path("/test/path/aspherical.tscb")

    cmds = olex2.create_nosphera2_cmds(tsc_path)

    cmd_string = "\n".join(cmds)
    # Should enable aspherical refinement
    assert "use_aspherical', True" in cmd_string
    assert "Calculate', False" in cmd_string
    assert "aspherical.tscb" in cmd_string


def test_create_refine_cmds():
    """Test that create_refine_cmds produces correct number of refinement cycles."""
    olex2 = Olex2Socket()

    cmds = olex2.create_refine_cmds(n_cycles=15, refine_starts=4)

    # Should have setup commands + refinement commands
    refine_commands = [cmd for cmd in cmds if cmd.startswith("refine ")]
    setup_commands = [cmd for cmd in cmds if not cmd.startswith("refine ")]

    # Check implementation details
    assert len(refine_commands) == 4  # Should have 4 refine starts
    assert all("refine 15" == cmd for cmd in refine_commands)  # Each should use 15 cycles
    assert len(setup_commands) >= 3  # Should have setup commands (refinement program, ACTA handling)

    # Check that essential setup is present
    setup_string = "\n".join(setup_commands)
    assert "olex2.refine" in setup_string  # Uses olex2 refinement
    assert "ACTA" in setup_string  # Handles ACTA instructions


@patch("qcrboxtools.robots.olex2.time.sleep")
@patch("qcrboxtools.robots.olex2.Olex2Socket._send_input")
def test_reserve_new_job_success(mock_send_input, mock_sleep):
    """Test reserve_new_job when reservation is successful immediately."""
    olex2 = Olex2Socket()
    mock_send_input.return_value = "reserved"

    olex2.reserve_new_job("test_task")

    mock_send_input.assert_called_once_with("reserve:test_task")
    mock_sleep.assert_not_called()


@patch("qcrboxtools.robots.olex2.time.sleep")
@patch("qcrboxtools.robots.olex2.Olex2Socket._send_input")
def test_reserve_new_job_busy_then_success(mock_send_input, mock_sleep):
    """Test reserve_new_job when server is busy then becomes available."""
    olex2 = Olex2Socket()

    mock_send_input.side_effect = ["busy", "reserved"]

    olex2.reserve_new_job("test_task")

    assert mock_send_input.call_count == 2
    mock_send_input.assert_any_call("reserve:test_task")
    mock_sleep.assert_called_once_with(1)


@patch("qcrboxtools.robots.olex2.Olex2Socket.send_command")
@patch("qcrboxtools.robots.olex2.Olex2Socket.create_refine_cmds")
def test_run_full_refinement(mock_create_refine_cmds, mock_send_command):
    """Test that run_full_refinement orchestrates refinement correctly."""
    olex2 = Olex2Socket()

    # Setup mock returns
    mock_create_refine_cmds.return_value = ["setup", "refine 25", "refine 25", "refine 25"]
    mock_send_command.return_value = "refinement log output"

    cif_path = Path("/test/structure.cif")
    tsc_path = Path("/test/aspherical.tscb")

    result = olex2.run_full_refinement(cif_path, tsc_path, n_cycles=25, refine_starts=3)

    # Check that refinement commands are created with correct parameters
    mock_create_refine_cmds.assert_called_once_with(25, 3)

    # Check that send_command is called with paths and joined commands
    mock_send_command.assert_called_once_with(cif_path, tsc_path, "setup\nrefine 25\nrefine 25\nrefine 25")

    # Check return value passes through
    assert result == "refinement log output"


@patch("qcrboxtools.robots.olex2.Olex2Socket.send_command")
@patch("qcrboxtools.robots.olex2.Olex2Socket.create_refine_cmds")
def test_run_full_refinement_defaults(mock_create_refine_cmds, mock_send_command):
    """Test run_full_refinement with default parameters."""
    olex2 = Olex2Socket()

    mock_create_refine_cmds.return_value = ["refine 20"] * 5  # Default 5 starts, 20 cycles
    mock_send_command.return_value = "default refinement"

    cif_path = Path("/test/structure.cif")

    result = olex2.run_full_refinement(cif_path)  # Use defaults

    # Should use default parameters
    mock_create_refine_cmds.assert_called_once_with(20, 5)  # Default n_cycles=20, refine_starts=5
    mock_send_command.assert_called_once_with(cif_path, None, "refine 20\nrefine 20\nrefine 20\nrefine 20\nrefine 20")
    assert result == "default refinement"
