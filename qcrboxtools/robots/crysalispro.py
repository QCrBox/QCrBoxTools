"""
CrysalisPro Module
=================

This module provides classes and methods to interact with and control the
Rigaku CrysAlisPro program via scripted commands. The central class, `CAPRobot`,
allows for seamless automation of data integration tasks from Rigaku Synergy
X-ray diffractometers. Additional helper classes facilitate the management of
XML configuration files used by CrysAlisPro.

Classes
-------
CAPRobot
    A class to interface with CrysAlisPro, allowing for the automation of
    command execution and data integration processes.

CAPXml
    A class to handle CAP XML files for CrysAlisPro, enabling read and write
    operations on parameters related to data analysis and AutoChem procedures.

CAPSubParameters
    A helper class to manage subsections of the XML file, providing dictionary-like
    access to the parameters.

CAPIndexParameter
    A class to handle indexed parameters within the XML file, providing easy
    translation between index and its corresponding value.

Exceptions
-----------
NoWorkFolder
    Raised when the specified work folder does not exist.

ListeningModeInactive
    Raised when the listening mode is inactive.

CAPBusy
    Raised when CrysAlisPro is busy executing another command.

CommandFailedException
    Raised when a command execution fails.

Usage
-----
To use this module, create an instance of `CAPRobot` with the path to the
command folder. You can then use its methods to execute various commands
and control the CrysAlisPro program. Additionally, the `CAPXml` class can be
used to manipulate XML configuration files as needed.
"""

import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Union

from .utils import infer_and_cast


class CAPIndexParameter:
    """
    A class to handle index parameters with translations for CrysAlisPro.
    """

    def __init__(self, index: int, trans_dict: Dict[str, str]):
        """
        Initialize a CAPIndexParameter.

        Parameters
        ----------
        index : int
            The index value.
        trans_dict : Dict[str, str]
            A dictionary for translating the index value.
        """
        self.index = index
        self._trans_dict = trans_dict

    @property
    def translated(self) -> str:
        """
        Get the translated value of the index.

        Returns
        -------
        str
            The translated value.
        """
        return self._trans_dict[str(self.index)]

    def __str__(self) -> str:
        """
        Get the string representation of the index.

        Returns
        -------
        str
            The string representation.
        """
        return str(self.index)

    def __repr__(self) -> str:
        """
        Get the representation of the CAPIndexParameter.

        Returns
        -------
        str
            The representation of the CAPIndexParameter.
        """
        return f"{self.__class__.__name__}({self.index}:{self.translated})"


class CAPSubParameters:
    """
    A class to handle sub-parameters section in a CAP XML file for CrysAlisPro.
    """

    def __init__(self, sub_et_element: ET.Element, tree: ET.ElementTree, xml_file: Path):
        """
        Initialize CAPSubParameters.

        Parameters
        ----------
        sub_et_element : ET.Element
            The XML element containing sub-parameters.
        tree : ET.ElementTree
            The reference to the complete XML tree.
        xml_file : Path
            The path to the complete XML file.
        """
        self._sub_et_element = sub_et_element
        self._tree = tree
        self._xml_file = xml_file

    def _to_et_keyword(self, keyword: str) -> str:
        """
        Convert a keyword to its XML element representation.

        Parameters
        ----------
        keyword : str
            The keyword to convert.

        Returns
        -------
        str
            The XML element representation of the keyword.
        """
        return f"__{keyword}__"

    def _to_dict_keyword(self, key: str) -> str:
        """
        Convert an XML element tag to a dictionary keyword.

        Parameters
        ----------
        key : str
            The XML element tag to convert.

        Returns
        -------
        str
            The dictionary keyword representation of the tag.
        """
        return key[2:-2]

    def __getitem__(self, key: str) -> Any:
        """
        Get a sub-parameter value by its keyword.

        Parameters
        ----------
        item : str
            The keyword of the sub-parameter.

        Returns
        -------
        Any
            The value of the sub-parameter.

        Raises
        ------
        KeyError
            If the sub-parameter keyword is not found in the XML file.
        """
        et_value = self._sub_et_element.find(self._to_et_keyword(key))
        if et_value is None:
            raise KeyError(f"Key {key} not found")

        et_indexinfo = self._sub_et_element.find(self._to_et_keyword(f"{key}__indexinfo"))
        if et_indexinfo is not None:
            split = (val.split("-") for val in et_indexinfo.text.split(";"))
            trans_dict = {key.strip(): "-".join(values).strip() for key, *values in split}
            return CAPIndexParameter(int(et_value.text), trans_dict)

        return infer_and_cast(et_value.text, cast_np=False)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set a parameter value within the XML section by its keyword.

        Parameters
        ----------
        key : str
            The keyword of the parameter within the XML section.
        value : Any
            The value to set.

        Raises
        ------
        KeyError
            If the keyword is not found within the XML section.
        """
        et_value = self._sub_et_element.find(self._to_et_keyword(key))

        if et_value is None:
            raise KeyError(f"Key {key} not found")
        et_value.text = str(value)
        self._tree.write(self._xml_file)

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over the keywords within the XML section.

        Returns
        -------
        Iterator[str]
            An iterator over the keywords within the XML section.
        """
        for el in self._sub_et_element:
            if el.tag.endswith("indexinfo__"):
                continue
            yield self._to_dict_keyword(el.tag)

    def __len__(self) -> int:
        """
        Get the number of parameters in the XML section.

        Returns
        -------
        int
            The number of parameters in the XML section.
        """
        return len(list(self.__iter__()))

    def __repr__(self) -> str:
        """
        Get the representation of the CAPSubParameters.

        Returns
        -------
        str
            The representation of the CAPSubParameters.
        """
        return f"{self.__class__.__name__}({dict(self)})"

    def __str__(self) -> str:
        """
        Get the string representation of the CAPSubParameters.

        Returns
        -------
        str
            The string representation of the CAPSubParameters.
        """
        return str(dict(self))

    def __contains__(self, item: str) -> bool:
        """
        Check if a keyword is in the XML section.

        Parameters
        ----------
        item : str
            The keyword to check.

        Returns
        -------
        bool
            True if the keyword is in the XML section, False otherwise.
        """
        return item in list(self.__iter__())

    def __dict__(self) -> Dict[str, Any]:
        """
        Get a dictionary representation of the XML section.

        Returns
        -------
        Dict[str, Any]
            A dictionary representation of the XML section.
        """
        return {key: val for key, val in self.items()}

    def keys(self) -> Iterator[str]:
        """
        Get the XML section's parameter keywords.

        Returns
        -------
        Iterator[str]
            An iterator over the XML section's keywords.
        """
        return list(iter(self))

    def values(self) -> Iterator[Any]:
        """
        Get the XML section's parameter values.

        Returns
        -------
        Iterator[Any]
            An iterator over the XML section's parameter values.
        """
        return [val for _, val in self.items()]

    def items(self) -> Iterator[Tuple[str, Any]]:
        """
        Get the XML section's keys and values as iterator of tuples.

        Returns
        -------
        Iterator[Tuple[str, Any]]
            An iterator over the XML section's parameter items.
        """
        return [(key, self[key]) for key in self.keys()]


class CAPXml:
    """
    A class to handle CAP XML files for CrysAlisPro. This class provides an interface
    to read from and write to XML configuration files used by CrysAlisPro, a program
    for integrating frames from Rigaku Synergy X-ray diffractometers. All modifications
    to the properties of this object will be immediately written to the XML file.

    The class allows for setting and getting the output file name used during data
    analysis and provides access to various parameters for AutoChem procedures and
    general data reduction. These parameters can be accessed and modified via
    CAPSubParameters instances.

    Attributes
    ----------
    output_file_name : str
        The base name used for files created during data analysis.
    autochem_parameters : CAPSubParameters
        The parameters for the AutoChem procedure in CrysAlisPro. Interact with them like you
        would with a python dictionary.
    proffit_parameters : CAPSubParameters
        The general parameters for data reduction in CrysAlisPro. Interact with them like you
        would with a python dictionary.

    Methods
    -------
    output_file_name : str
        Get or set the base name used for files created during data analysis.
    autochem_parameters : CAPSubParameters
        Get or set the parameters for the AutoChem procedure.
    proffit_parameters : CAPSubParameters
        Get or set the general parameters for data reduction.
    """

    def __init__(self, xml_file: Path):
        """
        Initialize CAPXml.

        Parameters
        ----------
        xml_file : Path
            The path to the XML file, must be readable and writeable.
        """
        self.xml_file = Path(xml_file)
        self._tree = ET.parse(xml_file)
        self._root = self._tree.getroot()

    @property
    def output_file_name(self) -> str:
        """
        Get the base name used for files created when doing the data analysis in
        accordance with the settings in the XML file.

        Returns
        -------
        str
            The output file name.
        """
        out_name_el = self._root.find("__PROFFIT__OUTPUT__NAME__")
        return out_name_el.find("__output_file_name__").text

    @output_file_name.setter
    def output_file_name(self, new_name: str) -> None:
        """
        Set the base name used for files created when doing the data analysis in
        accordance with the settings in the XML file.

        Parameters
        ----------
        new_name : str
            The new output file name.
        """
        out_name_el = self._root.find("__PROFFIT__OUTPUT__NAME__")
        out_name_el.find("__output_file_name__").text = new_name
        self._tree.write(self.xml_file)

    @property
    def autochem_parameters(self) -> CAPSubParameters:
        """
        Get the parameters concerning the AutoChem procedure in CrysAlisPro.

        Returns
        -------
        CAPSubParameters
            The AutoChem parameters.
        """
        autochem_el = self._root.find("__AUTOCHEM__SETTING__SECTION__")
        return CAPSubParameters(autochem_el, self._tree, self.xml_file)

    @autochem_parameters.setter
    def autochem_parameters(self, new_params: Union[CAPSubParameters, dict]):
        """
        Set the parameters concerning the AutoChem procedure in CrysAlisPro.

        Parameters
        ----------
        new_params : CAPSubParameters
            The new AUTOCHEM parameters.
        """
        autochem_el = self._root.find("__AUTOCHEM__SETTING__SECTION__")
        for key, value in new_params.items():
            autochem_el.find(f"__{key}__").text = str(value)
        self._tree.write(self.xml_file)

    @property
    def proffit_parameters(self) -> CAPSubParameters:
        """
        Get the general parameters for the data reduction in CrysAlisPro.

        Returns
        -------
        CAPSubParameters
            The data reduction parameters.
        """
        proffit_el = self._root.find("__PROFFIT__PARAMETERS__")
        return CAPSubParameters(proffit_el, self._tree, self.xml_file)

    @proffit_parameters.setter
    def proffit_parameters(self, new_params: CAPSubParameters) -> None:
        """
        Set the general parameters for the data reduction in CrysAlisPro.

        Parameters
        ----------
        new_params : CAPSubParameters
            The new data reduction parameters.
        """
        proffit_el = self._root.find("__PROFFIT__PARAMETERS__")
        for key, value in new_params.items():
            proffit_el.find(f"__{key}__").text = value
        self._tree.write(self.xml_file)


class NoWorkFolder(FileNotFoundError):
    """Exception raised when the work folder is not found."""


class ListeningModeInactive(Exception):
    """Exception raised when the listening mode is inactive."""


class CAPBusy(Exception):
    """Exception raised when CrysAlisPro is busy / is already executing a command."""


class CommandFailedException(Exception):
    """Exception raised when a command to CrysAlisPro fails."""


class CAPRobot:
    """
    A class to control the CrysAlisPro program for integrating frames from
    Rigaku Synergy X-ray diffractometers in a scripted fashion. Its functionality
    will be limited unless the dataset parameter file is set explicitly.
    """

    _dataset_par = "default"

    def __init__(self, command_folder: Path):
        """
        Initialize the CAPRobot with a command folder, this needs to match the command folder
        set as the CAP Listen mode was started.

        Parameters
        ----------
        command_folder : Path
            The path to the command folder.
        """
        self.command_folder = Path(command_folder)

    @property
    def dataset_par(self) -> Path:
        """
        Get the dataset parameter path.

        Returns
        -------
        str
            The dataset parameter path.
        """
        return self._dataset_par

    @dataset_par.setter
    def dataset_par(self, par_path: Path):
        """
        Set the dataset parameter path and load the dataset in CrysAlisPro.

        Parameters
        ----------
        par_path : Path
            The path to the dataset parameter file.

        Raises
        ------
        FileNotFoundError
            If the dataset parameter file is not found.
        """
        par_path = Path(par_path)
        if not par_path.exists():
            raise FileNotFoundError("Dataset parameter file not found")
        self._dataset_par = Path(par_path)
        self.send_command(f"xx selectexpnogui {str(par_path)}")

    @property
    def is_busy(self) -> bool:
        """
        Check if CrysAlisPro is busy.

        Returns
        -------
        bool
            True if CrysAlisPro is busy, False otherwise.
        """
        if (self.command_folder / "command.busy").exists():
            return True
        else:
            return False

    def _check_ready_for_command(self) -> None:
        """
        Check if CrysAlisPro is ready for a new command and raise the correct
        error if not.

        Raises
        ------
        NoWorkFolder
            If the work folder is not found.
        ListeningModeInactive
            If the listening mode is inactive.
        CAPBusy
            If CrysAlisPro is already busy executing a command.
        """
        if not self.command_folder.exists():
            raise NoWorkFolder("No work folder found")

        if (self.command_folder / "command.closed").exists():
            raise ListeningModeInactive("Listening mode is inactive")

        if (self.command_folder / "command.busy").exists():
            raise CAPBusy("CryAlisPro is already busy executing a command")

    def stop(self) -> bool:
        """
        Stop execution of the current command in CrysAlisPro if it is busy.

        Returns
        -------
        bool
            True if the stop command was sent, False otherwise.
        """
        if self.is_busy:
            (self.command_folder / "command.stop").touch()
            return True
        return False

    def _clean_up_folder(self) -> None:
        """Clean up the command folder by removing left over files."""
        possible_files = ["command.stop", "command.error", "command.done"]
        for file in possible_files:
            if (self.command_folder / file).exists():
                (self.command_folder / file).unlink()

    def send_command(self, command: str, in_background: bool = False, test_interval: float = 0.1, timeout: float = 600):
        """
        Send a command to CrysAlisPro.

        Parameters
        ----------
        command : str
            The command to send.
        in_background : bool, optional
            Whether to run the command in the background instead of blocking the further execution
            of the python script, by default False.
        test_interval : float, optional
            The interval to test for command completion, by default 0.1.
        timeout : float, optional
            The timeout for the command to complete, by default 600. Can also be set to None to
            wait indefinitely.

        Raises
        ------
        CommandFailedException
            If the command fails.
        TimeoutError
            If the command times out.
        """
        self._check_ready_for_command()
        self._clean_up_folder()
        (self.command_folder / "command.in").write_text(command)
        time.sleep(0.5)
        if not in_background:
            try:
                self.wait_for_command_to_finish(test_interval=test_interval, timeout=timeout)
            except CommandFailedException as e:
                raise CommandFailedException(f"Command failed: {command}") from e
            except TimeoutError as e:
                raise TimeoutError(f"Timeout ({timeout} s) waiting for command to finish: {command}") from e

    def wait_for_command_to_finish(self, test_interval: float = 0.1, timeout: float = 600) -> bool:
        """
        Wait for the command to finish. Only needed to be called by a user, if the command
        was sent in the background.

        Parameters
        ----------
        test_interval : float, optional
            The interval to test for command completion, by default 0.1.
        timeout : float, optional
            The timeout for the command to complete, by default 600. Can also be set to None to
            wait indefinitely.

        Returns
        -------
        bool
            True if the command completed successfully.

        Raises
        ------
        TimeoutError
            If the command times out.
        CommandFailedException
            If the command fails.
        """
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

    def create_xml_file(
        self,
        xml_file: Path,
        file_type: Optional[str] = None,
        file_path: Optional[Path] = None,
        output_name: Optional[str] = None,
    ) -> CAPXml:
        """
        Create an XML file from a CrysAlisPro binary settings file. Several options are available
        to specify the file to create the XML from. File type and file path are mutually exclusive.

        Parameters
        ----------
        xml_file : Path
            The path to the XML file to create.
        file_type : Optional[str], optional
            The file type to create XML from. This assumes the file is located in the same directory
            as the robots par file. Possible options for file type are 'proffitgui' for exporting
            the options selected in the 'Data reductions with options' and 'proffitpars' for a file
            created by the 'xx proffitloop' procedure, by default None.
        file_path : Optional[Path], optional
            The file path to create XML from for the two types see file_type, by default None.
        output_name : Optional[str], optional
            The output name for the files generated when executing the newly created XML file, by
            default None which will use the datasets default.

        Returns
        -------
        CAPXml
            The CAPXml object created from the XML file.

        Raises
        ------
        ValueError
            If neither or both file_type and file_path are provided, or if dataset_par is not set.
        FileNotFoundError
            If the specified file path does not exist.
        """
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
            output_name = ""
        else:
            output_name = f" {output_name}"
        self.send_command(f"xx partoxml {file_path} {xml_file}{output_name}")

        return CAPXml(xml_file)

    def run_from_xml_file(self, xml_file: Union[Path, CAPXml]) -> None:
        """
        Run a data reduction from frames using the settings from an XML file.
        Valid XML files can be created using the create_xml_file method.

        Parameters
        ----------
        xml_file : Path
            The path to the XML file.

        Raises
        ------
        FileNotFoundError
            If the XML file is not found.
        """
        if isinstance(xml_file, CAPXml):
            xml_file = xml_file.xml_file

        xml_file = Path(xml_file)
        if not xml_file.exists():
            raise FileNotFoundError(f"XML file not found at {xml_file}")
        self.send_command(f"dc proffit xml {xml_file}")

    def fullautoanalyse(self) -> None:
        """Run the full auto analysis command."""
        self.send_command("dc fullautoanalyse")

    def run_script(self, script_file: Path) -> None:
        """
        Run a script file. All script files must have a .mac extension.

        Parameters
        ----------
        script_file : Path
            The path to the script file.

        Raises
        ------
        FileNotFoundError
            If the script file is not found.
        ValueError
            If the script file does not have a .mac extension.
        """
        script_file = Path(script_file)
        if not script_file.exists():
            raise FileNotFoundError(f"Script file not found at {script_file}")
        if script_file.suffix != ".mac":
            raise ValueError("Script file must have .mac extension")
        self.send_command(f"dc runscript {script_file.with_suffix('')}")
