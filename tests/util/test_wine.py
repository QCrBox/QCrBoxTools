import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath
from unittest.mock import Mock, patch

import pytest

from qcrboxtools.util.wine import DefaultExecutor, OptionalWineExecutor, WinePathHelper


@pytest.fixture(name="wine_path_helper")
def fixture_wine_path_helper():
    return WinePathHelper()


def test_wine_path_helper_get_windows_path(wine_path_helper):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "Z:\\home\\user\\file.txt\n"
        unix_path = PurePosixPath("/home/user/file.txt")
        result = wine_path_helper.get_windows_path(unix_path)
        assert isinstance(result, PureWindowsPath)
        assert str(result) == "Z:\\home\\user\\file.txt"
        mock_run.assert_called_once_with(
            ["winepath", "-w", "/home/user/file.txt"], text=True, capture_output=True, check=True
        )


def test_wine_path_helper_get_wine_path(wine_path_helper):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "/home/user/file.txt\n"
        windows_path = PureWindowsPath("Z:\\home\\user\\file.txt")
        result = wine_path_helper.get_unix_path(windows_path)
        assert isinstance(result, PurePosixPath)
        assert str(result) == "/home/user/file.txt"
        mock_run.assert_called_once_with(
            ["winepath", "-u", "Z:\\home\\user\\file.txt"], text=True, capture_output=True, check=True
        )


def test_wine_path_helper_winepath_error(wine_path_helper):
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "winepath")
        unix_path = Path("/home/user/file.txt")
        with pytest.raises(subprocess.CalledProcessError):
            wine_path_helper.get_windows_path(unix_path)


@pytest.mark.parametrize(
    "use_wine, wine_available, expected_use_wine",
    [
        (True, True, True),  # Explicitly set to use wine, wine is available
        (True, False, True),  # Explicitly set to use wine, even if wine is not available
        (False, True, False),  # Explicitly set to not use wine, even if wine is available
        (False, False, False),  # Explicitly set to not use wine, wine is not available
        (None, True, True),  # Not specified, wine is available
        (None, False, False),  # Not specified, wine is not available
    ],
)
def test_wine_executor_init(use_wine, wine_available, expected_use_wine):
    with patch("subprocess.run") as mock_run:
        if wine_available:
            mock_run.return_value = Mock(returncode=0)
        else:
            mock_run.side_effect = subprocess.CalledProcessError(1, "wine --version")

        executor = OptionalWineExecutor(use_wine=use_wine)
        assert executor.use_wine == expected_use_wine

        if use_wine is None:
            mock_run.assert_called_once_with("wine --version", shell=True, check=True)
        else:
            mock_run.assert_not_called()


def test_wine_executor_to_cmd_args_with_wine():
    executor = OptionalWineExecutor(use_wine=True)
    result = executor.to_cmd_args(["mopro", "input.par"])
    assert result == ["wine", "mopro", "input.par"]


def test_wine_executor_to_cmd_args_without_wine():
    executor = OptionalWineExecutor(use_wine=False)
    result = executor.to_cmd_args(["mopro", "input.par"])
    assert result == ["mopro", "input.par"]


def test_wine_executor_convert_if_path_with_wine():
    with patch("qcrboxtools.util.wine.WinePathHelper") as MockWinePathHelper:
        mock_helper = MockWinePathHelper.return_value
        mock_helper.get_windows_path.return_value = PureWindowsPath("Z:\\path\\to\\file")

        executor = OptionalWineExecutor(use_wine=True)
        unix_path = Path("/path/to/file")
        result = executor.convert_if_path(unix_path)

        assert result == "Z:\\path\\to\\file"
        mock_helper.get_windows_path.assert_called_once_with(unix_path)
        MockWinePathHelper.assert_called_once()


def test_wine_executor_convert_if_path_without_wine():
    executor = OptionalWineExecutor(use_wine=False)
    unix_path = Path("/path/to/file")
    result = executor.convert_if_path(unix_path)
    assert result == unix_path


def test_wine_executor_convert_if_path_non_path():
    executor = OptionalWineExecutor(use_wine=True)
    non_path = "some_string"
    result = executor.convert_if_path(non_path)
    assert result == non_path


def test_wine_executor_execute_success():
    executor = OptionalWineExecutor(use_wine=False)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Success output"
        mock_run.return_value.stderr = ""
        result = executor.execute(["echo", "test"])
        assert result.returncode == 0
        assert result.stdout == "Success output"
        mock_run.assert_called_once_with(["echo", "test"], text=True, capture_output=True, check=False)


def test_wine_executor_execute_failure():
    executor = OptionalWineExecutor(use_wine=False)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = "Error output"
        with pytest.raises(RuntimeError) as excinfo:
            executor.execute(["false"])
        assert "Error when running command" in str(excinfo.value)
        assert "Error output" in str(excinfo.value)


def test_wine_executor_execute_with_wine():
    executor = OptionalWineExecutor(use_wine=True)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        executor.execute(["mopro", "input.par"])
        mock_run.assert_called_once_with(["wine", "mopro", "input.par"], text=True, capture_output=True, check=False)


def test_default_executor_execute_success():
    executor = DefaultExecutor()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Success output"
        mock_run.return_value.stderr = ""

        result = executor.execute(["echo", "test"])

        assert result.returncode == 0
        assert result.stdout == "Success output"
        mock_run.assert_called_once_with(["echo", "test"], text=True, capture_output=True, check=False)


def test_default_executor_execute_failure():
    executor = DefaultExecutor()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = "Error output"

        with pytest.raises(RuntimeError) as excinfo:
            executor.execute(["false"])

        assert "Error when running command" in str(excinfo.value)
        assert "Error output" in str(excinfo.value)


def test_default_executor_convert_if_path():
    executor = DefaultExecutor()
    path = Path("/some/path")
    assert executor.convert_if_path(path) == path
