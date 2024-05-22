from pathlib import Path
import time
from typing import Optional

#class CAPCommand:
#    def __init__(self, command_folder: Path):
#        self.command_folder = command_folder

class NoWorkFolder(FileNotFoundError):
    pass

class ListeningModeInactive(Exception):
    pass

class CAPBusy(Exception):
    pass

class CommandFailedException(Exception):
    pass


class CAPRobot:
    _dataset_par = "default"
    def __init__(self, command_folder: Path):
        self.command_folder = Path(command_folder)

    @property
    def dataset_par(self):
        return self._dataset_par

    @dataset_par.setter
    def dataset_par(self, par_path: Path):
        par_path = Path(par_path)
        if not par_path.exists():
            raise FileNotFoundError("Dataset parameter file not found")
        self._dataset_par = Path(par_path)
        self.send_command(f"xx selectexpnogui {str(par_path)}")

    @property
    def is_busy(self):
        if (self.command_folder / "command.busy").exists():
            return True
        else:
            return False

    def _check_ready_for_command(self):
        if not self.command_folder.exists():
            raise NoWorkFolder("No work folder found")

        if (self.command_folder / "command.closed").exists():
            raise ListeningModeInactive("Listening mode is inactive")

        if (self.command_folder / "command.busy").exists():
            raise CAPBusy("CryAlisPro is already busy executing a command")

    def stop(self):
        if self.is_busy:
            (self.command_folder / "command.stop").touch()
            return True
        else:
            return False

    def _clean_up_folder(self):
        possible_files = ["command.stop", "command.error", "command.done"]
        for file in possible_files:
            if (self.command_folder / file).exists():
                (self.command_folder / file).unlink()

    def send_command(self, command: str, in_background=False, test_interval=0.1, timeout=600):
        self._check_ready_for_command()
        self._clean_up_folder()
        (self.command_folder / "command.in").write_text(command)
        time.sleep(0.5)
        if not in_background:
            try:
                return self.wait_for_command_to_finish(test_interval=test_interval, timeout=timeout)
            except CommandFailedException as e:
                raise CommandFailedException(f"Command failed: {command}") from e
            except TimeoutError as e:
                raise TimeoutError(f"Timeout ({timeout} s) waiting for command to finish: {command}") from e

    def wait_for_command_to_finish(self, test_interval=0.1, timeout=600):
        if timeout is None:
            timeout_time = float("inf")
        else:
            timeout_time = time.time() + timeout
        while self.is_busy:
            time.sleep(test_interval)
            if time.time() > timeout_time:
                raise TimeoutError("Timeout waiting for command to finish")
        if (self.command_folder / "command.error").exists():
            raise CommandFailedException("Command failed")

        if (self.command_folder / "command.done").exists():
            (self.command_folder / "command.done").unlink()
        return True

    def create_xml_file(self, xml_file: Path, file_type: Optional[str] = None, file_path: Optional[Path] = None, output_name: Optional[str] = None):
        if file_type is None and file_path is None:
            raise ValueError("Either file_type or file_path must be provided")
        if file_type is not None and file_path is not None:
            raise ValueError("Only one of file_type or file_path can be provided")
        if file_type is not None and self.dataset_par == "default":
            raise ValueError("dataset_par must be set before creating xml file by file type only")
        if file_type is not None:
            file_path = self.dataset_par.with_suffix(f".{file_type}")
        else:
            file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError("File not found")
        xml_file = Path(xml_file)
        if output_name is None:
            output_name = ''
        else:
            output_name = f" {output_name}"
        self.send_command(f"xx partoxml {file_path} {xml_file}{output_name}")

    def run_from_xml_file(self, xml_file: Path):
        xml_file = Path(xml_file)
        if not xml_file.exists():
            raise FileNotFoundError(f"XML file not found at {xml_file}")
        self.send_command(f"dc proffit xml {xml_file}")

    def fullautoanalyse(self):
        self.send_command("dc fullautoanalyse")

    def run_script(self, script_file: Path):
        script_file = Path(script_file)
        if not script_file.exists():
            raise FileNotFoundError(f"Script file not found at {script_file}")
        if not script_file.suffix == ".mac":
            raise ValueError("Script file must have .mac extension")
        self.send_command(f"dc runscript {script_file.with_suffix('')}")

