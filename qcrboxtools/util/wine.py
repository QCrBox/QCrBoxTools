"""
This module provides classes and functions to help working with programs started
with WINE.

Classes
-------
WinePathHelper
    Helper class for converting file paths between Unix and Windows formats using winepath.
OptionalWineExecutor
    Executor that optionally uses Wine to run commands.
"""

import shlex
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, List, Optional


class WinePathHelper:
    """
    Helper class for converting file paths between Unix and Windows formats using winepath.
    """

    def __init__(self, winepath_executable: str = "winepath"):
        """
        Initializes WinePathHelper with the specified winepath executable.

        Parameters
        ----------
        winepath_executable : str, optional
            Path to the winepath executable (default is "winepath").
        """
        self.winepath_executable = winepath_executable

    def get_windows_path(self, unix_path: Path) -> PureWindowsPath:
        """
        Converts a Unix path to a Windows path using winepath.ein

        Parameters
        ----------
        unix_path : Path
            The Unix path to convert.

        Returns
        -------
        PureWindowsPath
            The converted Windows path.
        """
        process = subprocess.run(
            [self.winepath_executable, "-w", str(unix_path)], text=True, capture_output=True, check=True
        )
        return PureWindowsPath(process.stdout.strip())

    def get_unix_path(self, windows_path: PureWindowsPath) -> PurePosixPath:
        """
        Converts a Windows path to a Unix path using winepath.

        Parameters
        ----------
        windows_path : PureWindowsPath
            The Windows path to convert.

        Returns
        -------
        PurePosixPath
            The converted Unix path.
        """
        process = subprocess.run(
            [self.winepath_executable, "-u", str(windows_path)], text=True, capture_output=True, check=True
        )
        return PurePosixPath(process.stdout.strip())


class Executor(ABC):
    """
    Abstract base class for command executors.
    """

    @abstractmethod
    def execute(self, cmd_args: List[str], **kwargs) -> subprocess.CompletedProcess:
        """
        Executes a command with the specified arguments.

        Parameters
        ----------
        cmd_args : List[str]
            The command arguments.
        **kwargs
            Additional keyword arguments for subprocess.run.

        Returns
        -------
        subprocess.CompletedProcess
            The result of the subprocess execution.
        """
        pass

    @abstractmethod
    def convert_if_path(self, arg: Any) -> Any:
        """
        Converts an argument to a path if applicable.

        Parameters
        ----------
        arg : Any
            The argument to convert.

        Returns
        -------
        Any
            The converted argument.
        """
        pass


class OptionalWineExecutor(Executor):
    """
    Executor that optionally uses Wine to run commands.
    """

    def __init__(self, use_wine: Optional[bool] = None):
        """
        Initializes OptionalWineExecutor with an option to use Wine.

        Parameters
        ----------
        use_wine : Optional[bool], optional
            Whether to use Wine (default is None, auto-detection).
        """
        if use_wine is None:
            try:
                subprocess.run("wine --version", shell=True, check=True)
                use_wine = True
            except subprocess.CalledProcessError:
                use_wine = False
        self.use_wine = use_wine

    def to_cmd_args(self, original_cmd_args: List[str]) -> List[str]:
        """
        Converts command arguments to include Wine if needed. Also converts paths if Wine is used.

        Parameters
        ----------
        original_cmd_args : List[str]
            The original command arguments.

        Returns
        -------
        List[str]
            The modified command arguments.
        """
        original_cmd_args = [original_cmd_args[0]] + [self.convert_if_path(arg) for arg in original_cmd_args[1:]]
        if self.use_wine:
            return ["wine", *original_cmd_args]
        return original_cmd_args

    def convert_if_path(self, arg: Any) -> Any:
        """
        Converts an argument to a Windows path if applicable and Wine is used.

        Parameters
        ----------
        arg : Any
            The argument to convert.

        Returns
        -------
        Any
            The converted argument.
        """
        if isinstance(arg, (Path, PurePosixPath)) and self.use_wine:
            helper = WinePathHelper()
            return str(helper.get_windows_path(arg))
        return arg

    def execute(self, cmd_args: List[str], **kwargs) -> subprocess.CompletedProcess:
        """
        Executes a command with optional Wine usage.

        Parameters
        ----------
        cmd_args : List[str]
            The command arguments.
        **kwargs
            Additional keyword arguments for subprocess.run.

        Returns
        -------
        subprocess.CompletedProcess
            The result of the subprocess execution.

        Raises
        ------
        RuntimeError
            If the command execution fails.
        """
        cmd_args = self.to_cmd_args(cmd_args)
        process = subprocess.run(cmd_args, text=True, capture_output=True, check=False, **kwargs)

        if process.returncode != 0:
            cmd = shlex.join(cmd_args)
            raise RuntimeError(
                f"Error when running command\n{cmd}\n"
                + f"\nSTDERR:\n{process.stderr}"
                + f"\n\nSTDOUT:\n{process.stdout}"
            )

        return process


class DefaultExecutor(Executor):
    """
    Default command executor without Wine support.
    """

    def execute(self, cmd_args: List[str], **kwargs) -> subprocess.CompletedProcess:
        """
        Executes a command.

        Parameters
        ----------
        cmd_args : List[str]
            The command arguments.
        **kwargs
            Additional keyword arguments for subprocess.run.

        Returns
        -------
        subprocess.CompletedProcess
            The result of the subprocess execution.

        Raises
        ------
        RuntimeError
            If the command execution fails.
        """
        process = subprocess.run(cmd_args, text=True, capture_output=True, check=False, **kwargs)

        if process.returncode != 0:
            cmd = shlex.join(cmd_args)
            raise RuntimeError(
                f"Error when running command\n{cmd}\n"
                + f"\nSTDERR:\n{process.stderr}"
                + f"\n\nSTDOUT:\n{process.stdout}"
            )

        return process

    def convert_if_path(self, arg: Any) -> Any:
        """
        Compatibility function, returns the argument as is.

        Parameters
        ----------
        arg : Any
            The argument to convert.

        Returns
        -------
        Any
            The converted argument.
        """
        return arg
