"""
This module provides classes and functions to interact with MoProSuite executables
and files. It includes utilities to parse output files, execute commands using
optional Wine support, and manage MoPro input and output files.

Functions
---------
out2last_par_file(out_path: Path) -> Optional[PureWindowsPath]:
    Extracts the path of the last molecular .par file from the output file.

out2inp_file(out_path: Path) -> 'MoProInpFile':
    Extracts the content of the MoPro input file from the output file.

parse_out(out_path: Path) -> Tuple['MoProInpFile', Optional[str]]:
    Parses the output file to retrieve the input file and the last .par file.

Classes
-------
WinePathHelper:
    Helper class for converting file paths between Unix and Windows formats using winepath.

Executor(ABC):
    Abstract base class for command executors.

OptionalWineExecutor(Executor):
    Executor that optionally uses Wine to run commands.

DefaultExecutor(Executor):
    Default command executor without Wine support.

MoProImportRobot:
    Robot for importing files into MoProSuite.

MoProInpFile:
    Represents a MoPro input file.

MoProRobot:
    Robot for executing MoProSuite commands.
"""

import os
import re
import shlex
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional, Tuple


def out2last_par_file(out_path: Path) -> Optional[PureWindowsPath]:
    """
    Extracts the path of the last molecular .par file from the output file.

    Parameters
    ----------
    out_path : Path
        Path to the output file.

    Returns
    -------
    Optional[PureWindowsPath]
        Path to the last molecular .par file, or None if not found.
    """
    content = out_path.read_text(encoding="UTF-8")
    output_search = re.search(r"Last molecular file :\s?(.*?\.par)", content)
    if output_search:
        output_par_file = output_search.group(1).strip()
        return PureWindowsPath(output_par_file)
    return None


def out2inp_file(out_path: Path) -> "MoProInpFile":
    """
    Extracts the content of the MoPro input file from the output file.

    Parameters
    ----------
    out_path : Path
        Path to the output file.

    Returns
    -------
    MoProInpFile
        An instance of MoProInpFile containing the parsed input file content.
    """
    content = out_path.read_text(encoding="UTF-8")
    inp_content = re.search(
        (
            r"List of commands in MoPro input file : "
            + r".*?\n\s?\n?(.*?)\n"  # first name (not included), then content
            + r"===================================================\n"
        ),
        content,
        flags=re.DOTALL,
    ).group(1)
    return MoProInpFile.from_string(inp_content)


def parse_out(out_path: Path) -> Tuple["MoProInpFile", Optional[str]]:
    """
    Parses the output file to retrieve the input file and the last .par file.

    Parameters
    ----------
    out_path : Path
        Path to the output file.

    Returns
    -------
    Tuple[MoProInpFile, Optional[str]]
        A tuple containing the MoPro input file and the name of the last .par file.
    """
    inp_file = out2inp_file(out_path)
    last_par_file = out2last_par_file(out_path)
    if last_par_file is None:
        return inp_file, None
    return inp_file, last_par_file.name


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
        Converts a Unix path to a Windows path using winepath.

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

    def get_wine_path(self, windows_path: PureWindowsPath) -> PurePosixPath:
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


class MoProImportRobot:
    """
    Robot for importing CIF files into MoProSuite.
    """

    def __init__(self, executor: Executor = OptionalWineExecutor(), executable_path: Optional[Path] = None):
        """
        Initializes MoProImportRobot with an executor and executable path.

        Parameters
        ----------
        executor : Executor, optional
            The command executor to use (default is OptionalWineExecutor()).
        executable_path : Optional[Path], optional
            Path to the executable (default is None, using IMOPRO_PATH environment variable).
        """
        self.executor = executor
        if executable_path is None:
            try:
                executable_path = Path(os.environ["IMOPRO_PATH"])
            except KeyError as e:
                raise ValueError("IMOPRO_PATH environment variable not set. Please set or use executable_path.") from e
        else:
            executable_path = Path(executable_path)
        if not executable_path.exists():
            raise FileNotFoundError(f"No executable not found at {executable_path}")
        self.executable_path = executable_path

    def cif2par(self, cif_path: Path) -> Path:
        """
        Converts a CIF file to a PAR file.

        Parameters
        ----------
        cif_path : Path
            Path to the CIF file.

        Returns
        -------
        Path
            Path to the generated PAR file.
        """
        cmd_args = [str(self.executable_path), f"%{cif_path.absolute()}"]
        self.executor.execute(cmd_args, cwd=cif_path.parent)
        return cif_path.with_name(cif_path.stem + "_00.par")


class MoProInpFile:
    """
    Represents a MoPro input file.
    """

    files: Dict[str, Path]
    body: str

    def __init__(self, files: Dict[str, Path], body: str):
        """
        Initializes MoProInpFile with file paths and body content.

        Parameters
        ----------
        files : Dict[str, Path]
            Dictionary of file identifiers and paths.
        body : str
            The body content of the input file.
        """
        self.files = files
        self.body = body

    @classmethod
    def from_string(cls, string: str) -> "MoProInpFile":
        """
        Creates a MoProInpFile instance from a string.

        Parameters
        ----------
        string : str
            The string content of the input file.

        Returns
        -------
        MoProInpFile
            The created MoProInpFile instance.
        """
        lines = string.splitlines()
        files = {line[5:9]: Path(line[9:].strip()) for line in lines if line.startswith("FILE")}
        body = "\n".join(line for line in lines if not line.startswith("FILE"))
        return cls(files=files, body=body.strip())

    @classmethod
    def from_file(cls, file_path: Path) -> "MoProInpFile":
        """
        Creates a MoProInpFile instance from a file.

        Parameters
        ----------
        file_path : Path
            Path to the input file.

        Returns
        -------
        MoProInpFile
            The created MoProInpFile instance.
        """
        return cls.from_string(file_path.read_text())

    def write(self, file_path: Path):
        """
        Writes the MoPro input file to a specified path.

        Parameters
        ----------
        file_path : Path
            The path where the input file should be written.
        """
        string = "\n".join(f"FILE {key} {value}" for key, value in self.files.items())
        string += "\n" + self.body
        string += "\n"
        file_path.write_text(string)


class MoProRobot:
    """
    Robot for executing MoProSuite commands.
    """

    def __init__(self, executor: Executor = OptionalWineExecutor(), executable_path: Optional[Path] = None):
        """
        Initializes MoProRobot with an executor and executable path.

        Parameters
        ----------
        executor : Executor, optional
            The command executor to use (default is OptionalWineExecutor()).
        executable_path : Optional[Path], optional
            Path to the executable (default is None, using MOPRO_PATH environment variable).
        """
        self.executor = executor
        if executable_path is None:
            try:
                executable_path = Path(os.environ["MOPRO_PATH"])
            except KeyError as e:
                raise ValueError("MOPRO_PATH environment variable not set. Please set or use executable_path.") from e
        else:
            executable_path = Path(executable_path)
        if not executable_path.exists():
            raise FileNotFoundError(f"No executable not found at {executable_path}")
        self.executable_path = executable_path

    def run_file(self, file_path: Path) -> Optional[PureWindowsPath]:
        """
        Runs a MoProSuite input file and retrieves the last .par file.

        Parameters
        ----------
        file_path : Path
            Path to the input file.

        Returns
        -------
        Optional[PureWindowsPath]
            Path to the last .par file generated, or None if not found.
        """
        cmd_args = [str(self.executable_path), str(file_path)]
        self.executor.execute(cmd_args)

        work_folder = file_path.parent
        out_path = work_folder / "mopro.out"
        return out2last_par_file(out_path)
