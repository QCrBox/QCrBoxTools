import os
import re
import shlex
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional


def out2last_par_file(out_path: Path):
    content = out_path.read_text(encoding="UTF-8")
    output_search = re.search(r"Last molecular file :\s?(.*?\.par)", content)
    if output_search:
        output_par_file = output_search.group(1).strip()
        return PureWindowsPath(output_par_file)
    return None


def out2inp_file(out_path: Path):
    content = out_path.read_text(encoding="UTF-8")
    inp_content = re.search(
        (
            r"List of commands in MoPro input file : "
            + ".*?\n\s?\n?(.*?)\n"
            + "===================================================\n",
        ),
        content,
        flags=re.DOTALL,
    ).group(1)
    return MoProInpFile.from_string(inp_content)


def parse_out(out_path: Path):
    inp_file = out2inp_file(out_path)
    last_par_file = out2last_par_file(out_path)
    if last_par_file is None:
        return inp_file, None
    return inp_file, last_par_file.name


class WinePathHelper:
    def __init__(self, winepath_executable="winepath"):
        self.winepath_executable = winepath_executable

    def get_windows_path(self, unix_path: Path) -> PureWindowsPath:
        process = subprocess.run(
            [self.winepath_executable, "-w", str(unix_path)], text=True, capture_output=True, check=True
        )
        return PureWindowsPath(process.stdout.strip())

    def get_wine_path(self, windows_path: PureWindowsPath) -> Path:
        process = subprocess.run(
            [self.winepath_executable, "-u", str(windows_path)], text=True, capture_output=True, check=True
        )
        return Path(process.stdout.strip())


class Executor(ABC):
    @abstractmethod
    def execute(self, cmd_args: List[str], **kwargs): ...

    @abstractmethod
    def convert_if_path(self, arg: Any): ...


class OptionalWineExecutor(Executor):
    def __init__(self, use_wine=None):
        if use_wine is None:
            try:
                subprocess.run("wine --version", shell=True, check=True)
                use_wine = True
            except subprocess.CalledProcessError:
                use_wine = False
        self.use_wine = use_wine

    def to_cmd_args(self, original_cmd_args: List[str]):
        original_cmd_args = [original_cmd_args[0]] + [self.convert_if_path(arg) for arg in original_cmd_args[1:]]
        if self.use_wine:
            return ["wine", *original_cmd_args]
        return original_cmd_args

    def convert_if_path(self, arg: Any):
        if isinstance(arg, Path) and self.use_wine:
            helper = WinePathHelper()
            return str(helper.get_windows_path(arg))
        return arg

    def execute(self, cmd_args: List[str], **kwargs):
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
    def execute(self, cmd_args: List[str], **kwargs):
        process = subprocess.run(cmd_args, text=True, capture_output=True, check=False, **kwargs)

        if process.returncode != 0:
            cmd = shlex.join(cmd_args)
            raise RuntimeError(
                f"Error when running command\n{cmd}\n"
                + f"\nSTDERR:\n{process.stderr}"
                + f"\n\nSTDOUT:\n{process.stdout}"
            )

        return process

    def convert_if_path(self, arg: Any):
        return arg



class MoProImportRobot:
    def __init__(self, executor: Executor = OptionalWineExecutor(), executable_path: Optional[Path] = None):
        self.executor = executor
        if executable_path is None:
            try:
                executable_path = Path(os.environ["MOPRO_IMPORT_EXECUTABLE_PATH"])
            except KeyError as e:
                raise ValueError(
                    "MOPRO_IMPORT_EXECUTABLE_PATH environment variable not set. Please set or use executable_path."
                ) from e
        else:
            executable_path = Path(executable_path)
        if not executable_path.exists():
            raise FileNotFoundError(f"No executable not found at {executable_path}")
        self.executable_path = executable_path

    def cif2par(self, cif_path: Path):
        cmd_args = [str(self.executable_path), f"%{cif_path.absolute()}"]
        self.executor.execute(cmd_args, cwd=cif_path.parent)
        return cif_path.with_name(cif_path.stem + "_00.par")

class MoProInpFile:
    files: Dict[str, Path]
    body: str

    def __init__(self, files: Dict[str, Path], body: str):
        self.files = files
        self.body = body

    @classmethod
    def from_string(cls, string: str):
        lines = string.splitlines()
        files = {line[5:9]: Path(line[9:].strip()) for line in lines if line.startswith("FILE")}
        body = "\n".join(line for line in lines if not line.startswith("FILE"))
        return cls(files=files, body=body)

    @classmethod
    def from_file(cls, file_path: Path):
        return cls.from_string(file_path.read_text())

    def write(self, file_path: Path):
        string = "\n".join(f"FILE {key} {value}" for key, value in self.files.items())
        string += "\n" + self.body
        string += "\n"
        file_path.write_text(string)


class MoProRobot:
    def __init__(self, executor: Executor = OptionalWineExecutor(), executable_path: Optional[Path] = None):
        self.executor = executor
        if executable_path is None:
            try:
                executable_path = Path(os.environ["MOPRO_EXECUTABLE_PATH"])
            except KeyError as e:
                raise ValueError(
                    "MOPRO_EXECUTABLE_PATH environment variable not set. Please set or use executable_path."
                ) from e
        else:
            executable_path = Path(executable_path)
        if not executable_path.exists():
            raise FileNotFoundError(f"No executable not found at {executable_path}")
        self.executable_path = executable_path

    def run_file(self, file_path: Path):
        cmd_args = [str(self.executable_path), str(file_path)]
        self.executor.execute(cmd_args)

        work_folder = file_path.parent
        out_path = work_folder / "mopro.out"
        return out2last_par_file(out_path)
