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
from pathlib import Path, PureWindowsPath
from typing import Dict, Optional, Tuple

from ..util.wine import Executor, OptionalWineExecutor


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
