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
from unittest.mock import Mock, PropertyMock, call, patch

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
    olex2.structure_path = work_path
    _ = olex2.refine()

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
    work_path = os.path.join(tmp_path, "work.cif")
    shutil.copy("./tests/robots/olex/cif_files/refine_nonconv_allaniso.cif", work_path)

    tsc_path = Path("./tests/robots/olex/cif_files/refine_allaniso.tscb").absolute()

    olex2 = Olex2Socket()
    olex2.structure_path = work_path

    olex2.tsc_path = tsc_path
    _ = olex2.refine()

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
    assert olex2.structure_path is None

    olex2 = Olex2Socket("test_server", 1234)
    assert olex2.server == "test_server"
    assert olex2.port == 1234
    assert olex2.structure_path is None


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
    assert olex2.port == "12345"
    assert olex2.structure_path is None


def test_olex2_wait_for_completion():
    # create a mock answer generator that returns "test_response" 10 times and then "finished"
    def mock_send_input_func():
        n = 0
        while n < 10:
            n += 1
            yield "test_response"
        yield "finished"

    mock_send_input_iter = mock_send_input_func()

    mock_send_input = Mock(side_effect=lambda *args: next(mock_send_input_iter))

    olex2 = Olex2Socket()
    with patch("qcrboxtools.robots.olex2.Olex2Socket._send_input", mock_send_input):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            olex2.wait_for_completion(11, "test", "test_input")

        assert mock_send_input.call_count == 11
        assert mock_send_input.call_args_list == [call("status:test")] * 11

        mock_send_input_iter = mock_send_input_func()
        mock_send_input.side_effect = lambda *args: next(mock_send_input_iter)
        with pytest.warns(UserWarning):
            olex2.wait_for_completion(1, "test", "test_input")


def test_wait_for_completion_failed():
    olex2 = Olex2Socket()

    with patch("qcrboxtools.robots.olex2.Olex2Socket._send_input", lambda *args: "failed"):
        with pytest.raises(RuntimeError):
            olex2.wait_for_completion(1, "test", "test_input")


@patch("qcrboxtools.robots.olex2.Olex2Socket.wait_for_completion")
@patch("qcrboxtools.robots.olex2.Olex2Socket._send_input")
def test_olex2_send_command(mock_send_input, mock_wait_for_completion, tmp_path):
    # test that the task id is incremented and the command is sent, no log is read as none exists
    # mock_structure_path = PropertyMock(return_value= tmp_path / "test.cif")

    with patch.object(Olex2Socket, "structure_path", PropertyMock):
        olex2 = Olex2Socket(structure_path=tmp_path / "test.cif")
        assert olex2._task_id_counter.__repr__() == "count(0)"
        assert olex2.send_command("test_input") is None
        mock_send_input.assert_called_once_with("run:0\nlog:task_0.log\ntest_input")
        assert mock_wait_for_completion.called
        assert olex2._task_id_counter.__repr__() == "count(1)"

        # test that log is read when exists
        (tmp_path / "task_1.log").write_text("test_log_content")
        assert olex2.send_command("test_input") == "test_log_content"
        assert olex2._task_id_counter.__repr__() == "count(2)"
        assert mock_send_input.call_count == 2


@patch("qcrboxtools.robots.olex2.Olex2Socket.wait_for_completion")
@patch("qcrboxtools.robots.olex2.Olex2Socket.send_command")
@patch("qcrboxtools.robots.olex2.Olex2Socket._send_input")
def test_olex2_structure_path_setter(mock_send_input, mock_send_command, mock_wait_for_input, tmp_path):
    (tmp_path / "test.cif").touch()
    (tmp_path / "test.ins").touch()
    (tmp_path / "test.hkl").touch()

    olex2 = Olex2Socket()
    olex2.structure_path = tmp_path / "test.cif"
    assert olex2.structure_path == tmp_path / "test.cif"
    assert mock_send_command.call_count == 1
    assert mock_send_input.call_count == 1
    assert mock_wait_for_input.call_count == 1
    # Check the existence of moved files
    assert (tmp_path / "test_moved.ins").exists()
    assert (tmp_path / "test_moved.hkl").exists()

    (tmp_path / "test2.cif").touch()
    olex2.structure_path = tmp_path / "test2.cif"

    assert olex2.structure_path == tmp_path / "test2.cif"
    assert mock_send_command.call_count == 2
    assert mock_send_input.call_count == 2
    assert mock_wait_for_input.call_count == 2

    assert not (tmp_path / "test_moved2.ins").exists()
    assert not (tmp_path / "test_moved2.hkl").exists()

    # test that the task id is set to a higher value when log files exist
    (tmp_path / "task_1.log").touch()
    olex2.structure_path = tmp_path / "test.cif"
    assert olex2._task_id_counter.__repr__() == "count(2)"

    # test that a tsc_file is copied to the structure_path folder if it is not there
    tsc_path = tmp_path / "test.tscb"
    tsc_path.touch()
    olex2.tsc_path = tsc_path
    subfolder = tmp_path / "subfolder"
    subfolder.mkdir()
    subfolder_cif = subfolder / "test.cif"
    subfolder_cif.touch()
    olex2.structure_path = subfolder_cif
    assert (subfolder / tsc_path.name).exists()

    # test that warning is raised, when send_command raises a RuntimeError
    olex2.tsc_path = None  #  Runtime Errors are only catched while loading a structure
    mock_send_command.side_effect = RuntimeError
    with pytest.warns(UserWarning):
        olex2.structure_path = tmp_path / "test.cif"


@patch("qcrboxtools.robots.olex2.Olex2Socket.send_command")
def test_olex2_tsc_path_setter(mock_send_command, tmp_path):
    with patch.object(Olex2Socket, "structure_path", PropertyMock):
        # test that ValueError is raised when setting tsc_path without setting structure_path
        olex2 = Olex2Socket()
        olex2.structure_path = None
        with pytest.raises(ValueError):
            olex2.tsc_path = tmp_path / "test.tscb"

        # test that tsc_path is set correctly in intended case

        cif1_path = tmp_path / "test1.cif"
        cif1_path.touch()

        olex2 = Olex2Socket(structure_path=cif1_path)

        tsc_path = tmp_path / "test.tscb"
        tsc_path.touch()

        olex2.tsc_path = tsc_path
        assert olex2.tsc_path == tsc_path
        assert mock_send_command.call_args_list[-1] == call(
            "\n".join(
                [
                    "spy.SetParam('snum.NoSpherA2.use_aspherical', True)",
                    "spy.SetParam('snum.NoSpherA2.Calculate', False)",
                    "spy.SetParam('snum.NoSpherA2.source', 'tsc_file')",
                    f"spy.SetParam('snum.NoSpherA2.file', '{tsc_path.name}')",
                ]
            )
        )

        # test that the tsc file is not set if it does not exist
        tsc_path.unlink()
        with pytest.raises(FileNotFoundError):
            olex2.tsc_path = tsc_path

        # test that the tsc file is copied to the structure_path folder if somewhere else
        structure_subfolder = tmp_path / "structure"
        structure_subfolder.mkdir()
        (structure_subfolder / "test.cif").touch()
        olex2.structure_path = structure_subfolder / "test.cif"
        tsc_path.touch()
        olex2.tsc_path = tsc_path
        assert (structure_subfolder / tsc_path.name).exists()

        # test that NoSpherA2 commands are send when tsc_path is set to None
        olex2.tsc_path = None
        assert olex2.tsc_path is None
        assert mock_send_command.call_args_list[-1] == call(
            "\n".join(
                [
                    "spy.SetParam('snum.NoSpherA2.use_aspherical', False)",
                    "spy.SetParam('snum.NoSpherA2.Calculate', False)",
                    "spy.SetParam('snum.NoSpherA2.file', None)",
                ]
            )
        )


@patch("qcrboxtools.robots.olex2.Olex2Socket._send_input", new_callable=lambda: Mock(return_value="ready"))
def test_check_connection(mock_send_input):
    olex2 = Olex2Socket()
    assert olex2.check_connection()
    assert mock_send_input.call_args_list == [call("status")]


@patch("qcrboxtools.robots.olex2.Olex2Socket.send_command")
def test_olex2_refine(mock_send_command, tmp_path):
    with patch.object(Olex2Socket, "structure_path", PropertyMock):
        olex2 = Olex2Socket()

        # test that ValueError is raised when no structure loaded
        with pytest.raises(ValueError):
            olex2.structure_path = None
            olex2.refine()

        (tmp_path / "test.cif").touch()
        olex2.structure_path = tmp_path / "test.cif"
        olex2.refine(n_cycles=5, refine_starts=4)
        cmd_string = "\n".join(["refine 5"] * 4)
        assert cmd_string in mock_send_command.call_args[0][0]


@patch("qcrboxtools.robots.olex2.Olex2Socket._send_input")
def test_shutdown(mock_send_input):
    olex2 = Olex2Socket()
    olex2._shutdown_server()
    assert mock_send_input.call_args_list == [call("stop")]
