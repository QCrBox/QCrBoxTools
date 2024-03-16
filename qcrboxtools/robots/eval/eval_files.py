# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

import re
from itertools import product
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from iotbx.cif import model, reader


def infer_and_cast(s: str) -> Union[float, int, bool, np.ndarray, str]:
    """
    Attempts to cast a string to an integer, float, boolean, or retains it as
    a string based on its value.

    Parameters
    ----------
    s : str
        The string to be cast.

    Returns
    -------
    Union[int, float, bool, np.ndarray, str]
        The cast value.
    """
    # Try to cast to integer
    try:
        return int(s)
    except ValueError:
        pass

    # Try to cast to float
    try:
        return float(s)
    except ValueError:
        pass

    # Try to cast to boolean
    if s.lower() in ["true", "false"]:
        return s.lower() == "true"

    # Try to cast to numpy array (if contains space-separated values)
    if " " in s:
        return np.array([infer_and_cast(sub_s) for sub_s in s.split()])

    # If all else fails, return as string
    return s


class RelativePathFile:
    """
    A class representing a relative path file.

    This class provides a basic framework for managing file-related operations
    such as reading from and writing to files based on relative paths Only meant
    as BaseClass for the other classes in this file.

    Attributes
    ----------
    filename : str
        The name of the file.

    Methods
    -------
    __init__(self, filename: str, content: str)
        Initializes a new instance of the RelativePathFile class.
    from_file(cls, file_path: str) -> 'RelativePathFile'
        Reads a file from the given path and creates a RelativePathFile instance.
    to_file(self, directory: Optional[str] = None)
        Writes the RelativePathFile content to a text file in the specified directory.
    """

    def __init__(self, filename: str, content: str):
        """
        Initializes a new instance of the TextFile class.

        Parameters
        ----------
        filename : str
            The name of the text file.
        content : str
            The content of the file as a string.
        """
        self.filename = filename

    @classmethod
    def from_file(cls, file_path: str) -> "RelativePathFile":
        """
        Reads a file from the given path and creates a RelativePathFile instance.

        Parameters
        ----------
        file_path : str
            The path to the file to be read.

        Returns
        -------
        RelativePathFile
            An instance of RelativePathFile initialized with the contents of the file.
        """
        path = Path(file_path)
        with path.open("r", encoding="UTF-8") as fobj:
            text = fobj.read()
        return cls(path.name, text)

    def to_file(self, directory: Optional[str] = None):
        """
        Writes the RelativePathFile content to a text file.

        If the directory is None, writes to the current directory.

        Parameters
        ----------
        directory : Optional[str], default None
            The directory where the file should be written. If None, writes to the
            current directory.
        """
        file_path = Path(directory) / self.filename if directory else Path(self.filename)
        with file_path.open("w", encoding="UTF-8") as fobj:
            fobj.write(str(self))


class TextFile(RelativePathFile):
    """
    A subclass of RelativePathFile representing a text file.

    This class extends the RelativePathFile with specific methods and attributes
    relevant to text file operations.

    Attributes
    ----------
    text : str
        The content of the text file as a string.

    Methods
    -------
    __init__(self, filename: str, text: str)
        Initializes a new instance of the TextFile class.
    __str__(self) -> str
        Returns a string representation of the TextFile content.
    """

    def __init__(self, filename: str, content: str):
        """
        Initializes a new instance of the TextFile class.

        Parameters
        ----------
        filename : str
            The name of the text file.
        content : str
            The content of the text file as a string.
        """
        super().__init__(filename, content)
        self.text = content

    def __str__(self):
        return self.text


class EvalCommand:
    """
    A class representing a command for files within Eval.

    Attributes
    ----------
    name : str
        The name of the Eval command.
    options : Union[str, List[Union[str, int, float]], None]
        The options for the command, which can be a string, list of values, or None.

    Methods
    -------
    __init__(self, name: str, options: Union[str, List[Union[str, int, float]], None])
        Initializes an EvalCommand with a command name and associated options.
    __str__(self) -> str
        Returns a string representation of the Eval command.
    """

    def __init__(self, name: str, options: Union[str, List[Union[str, int, float]], None]):
        """
        Initializes the EvalCommand with a command name and associated options.

        Parameters
        ----------
        name : str
            The name of the Eval command.
        options : Union[str, List[Union[str, int, float]], None]
            The options for the command, which can be a single string, a list of
            values (strings, integers, or floats), or None if there are no options.
        """
        self.name = name
        self.options = options

    def __str__(self):
        """
        Returns a string representation of the Eval command.

        The method constructs a string by concatenating the command name with its
        options. If options are provided as a list, they are joined by spaces.
        If no options are provided, only the command name is returned.

        Returns
        -------
        str
            The string representation of the Eval command.
        """
        if self.options is None:
            return self.name
        if isinstance(self.options, list):
            options_str = " ".join(str(val) for val in self.options if val is not None)
        else:
            options_str = str(self.options)
        return f"{self.name} {options_str}"


class PicFile(RelativePathFile):
    """
    A class representing the content of a PIC file as a dictionary.

    This class allows for reading, modifying, and writing PIC file data.

    Attributes
    ----------
    _eval15_commands : List[str]
        A list of recognized commands for the PIC file format.

    Methods
    -------
    __init__(content_str)
        Constructs a PicFile object from a string.
    __str__()
        Returns a formatted string representation of the PIC file content.
    infer_and_cast(s)
        Casts a string to an int, float, bool, or retains it as a string.
    add_data(new_key, options)
        Adds or updates a command in the PIC file data.
    formatted_entry(key)
        Formats a single data entry for output as a string.
    to_file(file_path)
        Writes the PIC file content to a file.
    from_file(file_path)
        Creates a PicFile instance from the contents of a specified file.
    """

    _eval15_commands = [
        "abort",
        "adcnoise",
        "addconstant",
        "ambiguity",
        "animo",
        "anivec",
        "autofirst",
        "autorestore",
        "badresponsthreshold",
        "batch",
        "beamgridsize",
        "beampixelsize",
        "beampos",
        "boxclose",
        "boxfixed",
        "boximpactdiff",
        "boxnext",
        "boxopen",
        "boxscan",
        "boxsize",
        "cell",
        "centraloverlap",
        "changecoef",
        "check",
        "clear",
        "cntclose",
        "cntfile",
        "cntwrite",
        "collimatordiameter",
        "collimatortheight",
        "collimatortype",
        "collimatorwidth",
        "colour",
        "colourscheme",
        "contour",
        "correct",
        "cutoff",
        "cvectorrot",
        "debugimpact",
        "delay",
        "dict",
        "dictname",
        "diffarg",
        "diffracpos",
        "diffdividesigma",
        "diffscale",
        "difftype",
        "dirax",
        "diraxclose",
        "diraxopen",
        "dist",
        "distribution",
        "divmax",
        "draw",
        "dump",
        "dur",
        "edgefraction",
        "epsilon",
        "eval14",
        "eval14pointspread",
        "eval14threshold",
        "eval14type",
        "exhor",
        "exit",
        "exrot",
        "exver",
        "facetest",
        "faceconvert",
        "fbg",
        "fbragg",
        "fibre",
        "fibreaxis",
        "file",
        "fix",
        "flex",
        "focfile",
        "focus",
        "focuschange",
        "focuslambda",
        "focustype",
        "follow",
        "fomtype",
        "force",
        "forever",
        "framefraction",
        "free",
        "gain",
        "go",
        "goniostatchange",
        "goniostatprint",
        "goniostattype",
        "graphics",
        "gravity",
        "gravitycycles",
        "gravitytype",
        "help",
        "hkl",
        "ignore",
        "imagefilename",
        "impact",
        "impactcolourtype",
        "impactdotsize",
        "impactlabel",
        "impactnb",
        "impactrow",
        "incidencecoefficient",
        "incidencecoefficient2",
        "incidencecoefficient3",
        "inner",
        "intermediate",
        "isigrefine",
        "kappasupport",
        "lambdaadd",
        "lambdaaddfull",
        "lambdacalcmean",
        "lambdahigh",
        "lambdainit",
        "lambdalow",
        "lambdameanmid",
        "lambdaprint",
        "lambdarange",
        "lambdasigma",
        "lambdatype",
        "lambdavalue",
        "lambdaweight",
        "largelegenda",
        "latt",
        "lattfraction",
        "latthigh",
        "lattiso",
        "lattlow",
        "latttype",
        "lattvec",
        "legenda",
        "limits",
        "link",
        "listoff",
        "liston",
        "loadexppic",
        "loadrefine",
        "loadxtalevc",
        "lock",
        "longdelay",
        "low",
        "lpfinal",
        "lsqtype",
        "manual",
        "markdefault",
        "maxtry",
        "menu",
        "mica",
        "micahigh",
        "micalow",
        "micascale",
        "micatype",
        "micavec",
        "model",
        "mosaic",
        "mosaicadd",
        "mosaicinit",
        "mosaicrotangle",
        "mosaicrotaxis",
        "mosaictype",
        "mosaicweight",
        "msa1",
        "msa2",
        "mu",
        "nbcommonfraction",
        "nbcovariance",
        "nbdeltaindex",
        "nbeffective",
        "nbexpandhv",
        "nbexpandr",
        "nboverlapfraction",
        "nbsimfraction",
        "nbstillfraction",
        "nbtype",
        "nbvolume",
        "ncell",
        "ndivinner",
        "ndivouter",
        "netto",
        "newworld",
        "next",
        "nmosaic",
        "nrefineouter",
        "nop",
        "noprintcontour",
        "normalize",
        "nrmat",
        "obsminfrac",
        "oneframe",
        "onerot",
        "onescan",
        "outer",
        "output",
        "overflowthreshold",
        "penalty",
        "pick",
        "play",
        "plot",
        "plot0",
        "plot000",
        "plotrot",
        "pointspreadgammma",
        "pointspreadmoment",
        "pointspreadthreshold",
        "pointspreadtype",
        "polarisation",
        "pq",
        "printcontour",
        "printextreme",
        "printframes",
        "printprojection",
        "printposmin",
        "printsliceframe",
        "printslicelimit",
        "printslices",
        "printxtalsample",
        "projection",
        "ps",
        "qhor",
        "qrefine",
        "qrot",
        "qshift",
        "qvec",
        "qver",
        "randomcache",
        "randominit",
        "randomseed",
        "read",
        "readcorners",
        "readdetalign",
        "readfaces",
        "readfcalc",
        "readicalc",
        "readins",
        "refine",
        "refinebox",
        "refinefile",
        "refineouter",
        "refineshift",
        "refinestatus",
        "refinetol",
        "refinetol2",
        "refinetypeinner",
        "refinetypeouter",
        "refinewrite",
        "reflnr",
        "reject",
        "reset",
        "responsfraction",
        "restore",
        "rmat",
        "rock",
        "rotatermat",
        "rotaxnr",
        "sampleplot",
        "sampleplotcolourtype",
        "sampleplotdotsize",
        "sampleplotfloor",
        "sampleplotscale",
        "save",
        "savemeanshift",
        "savemeanshiftfloor",
        "scaletype",
        "scanfomtype",
        "searchanivec",
        "sensordepth",
        "sensoroffset",
        "shextra",
        "shhor",
        "shift",
        "shiftgrid",
        "shiftrestfracmin",
        "shiftrmat",
        "shinit",
        "shinitsector",
        "show",
        "showfaces",
        "shrot",
        "shver",
        "simulate",
        "sliceclose",
        "slicefactor",
        "slicefile",
        "slicethreshold",
        "slicewrite",
        "slicewritefilter",
        "splitshift",
        "square",
        "still",
        "stillframes",
        "stillinc",
        "stilltype",
        "store",
        "storestatus",
        "sumhkl",
        "svdtol",
        "swing",
        "sync",
        "target",
        "targetname",
        "test",
        "trace",
        "unlink",
        "vecscan",
        "vecscanfile",
        "vector",
        "view",
        "viewrot",
        "volcortype",
        "wait",
        "weightscheme",
        "window",
        "xa",
        "xb",
        "xc",
        "xrot",
        "xrotx",
        "xroty",
        "xrotz",
        "xtalax",
        "xtalcfac",
        "xtalplot",
        "xtalsamplesize",
        "xtalscale",
        "xtalshift",
        "xtalvec",
        "xyztohkl",
        "zero",
        "zoom",
        "zoomfraction",
        "zoompick",
        "zoomtitle",
    ]

    _commands_with_X = [
        "Sample{}",
        "dur{}",
        "exhor{}",
        "exrot{}",
        "exver{}",
        "lambdahigh{}",
        "lambdalow{}",
        "lambdasigma{}",
        "lambdatype{}",
        "lambdavalue{}",
        "lambdaweight{}",
        "lambda{}",
        "lamsig{}",
        "lamwght{}",
        "lattice{}",
        "mosaicrotangle{}",
        "mosaicrotaxis{}",
        "mosaicweight{}",
        "mosaic{}",
        "mosrot{}",
        "moswght{}",
        "rmat{}",
        "shextra{}",
        "shhor{}",
        "shiftrmat{}",
        "shinitsector{}",
        "shinit{}",
        "shrot{}",
        "shver{}",
    ]

    _eval15_commands += [
        cmd.format(number) for cmd, number in product(_commands_with_X, range(2, 10))
    ]

    def __init__(
        self,
        filename,
        content_str: str,
    ):
        """
        Initializes a new instance of the PicFile class.

        Parameters
        ----------
        content_str : str
            A string representation of the contents of a PIC file.
        """
        super().__init__(filename=filename, content=content_str)

        line_contents = [
            line.strip().split() for line in content_str.split("\n") if not line.startswith("!")
        ]
        self.command_list = []

        options = []
        for line in line_contents:
            if len(line) == 0:
                continue
            cmd = line[0]
            for item in line[1:]:
                if item in self._eval15_commands:
                    if len(options) == 0:
                        options.append(infer_and_cast(item))
                        continue
                    self.add_data(cmd, options)
                    options = []
                    cmd = item
                else:
                    options.append(infer_and_cast(item))

            self.add_data(cmd, options)
            options = []

    def __str__(self) -> str:
        """
        Returns a formatted string representation of the PIC file content.

        Returns
        -------
        str
            The formatted content of the PIC file.
        """
        return "\n".join(str(val) for val in self.command_list)

    def add_data(self, name: str, options: List[Union[int, float, bool, str]]):
        """
        Adds a new command to the PIC file data or updates an existing command.

        Parameters
        ----------
        name : str
            The command key to add or update.
        options : List[Union[int, float, bool, str]]
            The list of values associated with the command.
        """

        if len(options) == 0:
            self.command_list.append(EvalCommand(name, None))
        elif len(options) == 1:
            self.command_list.append(EvalCommand(name, options[0]))
        else:
            self.command_list.append(EvalCommand(name, options))

    def __getitem__(self, key):
        found_cmds = [cmd for cmd in self.command_list if cmd.name == key]
        if len(found_cmds) == 0:
            raise KeyError("No command with that name in file")
        return found_cmds


class SettingsVicFile(RelativePathFile, dict):
    """
    A class representing the content of a VIC file as a dictionary.

    This class allows for reading, modifying, and writing VIC file data.
    It can be initialized from a string representation of a VIC file or directly from
    a file.

    Methods
    -------
    __init__(content_str)
        Constructs a SettingsVicFile object from a string.
    __str__()
        Returns a formatted string representation of the VIC file content.
    from_file(file_path)
        Creates a SettingsVicFile instance from the contents of a specified file.
    to_file(file_path)
        Writes the VIC file content to a file.
    """

    def __init__(self, filename: str, content: str):
        """
        Initializes a new instance of the SettingsVicFile class.

        Parameters
        ----------
        content : str
            A string representation of the contents of a VIC file.
        """
        super().__init__(filename, content)
        for line in content.split("\n"):
            if not line.startswith("!"):
                parts = line.split()
                if len(parts) > 1:
                    self[parts[0]] = infer_and_cast(" ".join(parts[1:]))

    def __str__(self) -> str:
        """
        Returns a formatted string representation of the VIC file content.

        Returns
        -------
        str
            The formatted content of the VIC file.
        """
        return "\n".join(
            f"{key} {' '.join(map(str, values)) if isinstance(values, np.ndarray) else values}"
            for key, values in self.items()
        )


class RmatFile(RelativePathFile, dict):
    """
    A class for handling and manipulating RMAT file data.

    RmatFile extends RelativePathFile and utilizes dictionary functionality for
    storing and managing RMAT file data. It offers specialized methods for reading,
    processing, and writing RMAT file data in both RMAT and CIF (Crystallographic
    Information File) formats.

    Methods
    -------
    __init__(filename: str, content: Optional[str] = None)
        Initializes an RmatFile object with the content of an RMAT file.
    convert_to_numpy(array_str: str) -> np.ndarray
        Converts a string representation of a matrix to a numpy array.
    extract_data(text: str)
        Extracts and processes RMAT data from the provided text.
    value_as_string(key: str) -> str
        Converts RMAT data values to a formatted string representation.
    to_cif_dict() -> dict
        Converts RMAT data to a dictionary in CIF format.
    from_cif_dict(filename: str, cif_dict: dict) -> 'RmatFile'
        Creates an RmatFile instance from CIF dictionary data.
    """

    def __init__(self, filename: str, content: Optional[str] = None):
        """
        Initializes an RmatFile object with the content of an RMAT file.

        Parameters
        ----------
        filename : str
            The name of the RMAT file.
        content : Optional[str], default None
            The text content of the RMAT file. If None, the class will be initialized empty.
        """
        super().__init__(filename, content)
        if content is not None:
            self.extract_data(content)

    @staticmethod
    def convert_to_numpy(array_str: str) -> np.ndarray:
        """
        Converts a string representation of a matrix to a numpy array.

        Parameters
        ----------
        array_str : str
            A string containing matrix data, separated by whitespace and newlines.

        Returns
        -------
        np.ndarray
            A numpy array representation of the matrix.
        """
        lines = array_str.strip().split("\n")
        return np.squeeze(np.array([list(map(float, row.split())) for row in lines]))

    def extract_data(self, text: str):
        """
        Extracts and processes RMAT data from a given string.

        Updates the class's internal state (dictionary) with processed data.

        Parameters
        ----------
        text : str
            A string containing the RMAT file content.
        """
        patterns = {
            "RMAT": r"RMAT ([ABCFIPR])\n(.*?\n.*?\n.*?)\n",
            "TMAT": r"TMAT .*? (.*?)\n(.*?\n.*?\n.*?)\n",
            "CELL": r"\nCELL ([\d\s\.\-]+)",
            "SIGMACELL": r"SIGMACELL ([\d\s\.\-]+)",
            "QVECType": r"QVECType (.*?)\n",
            "QVEC": r"QVEC (.*?)\n",
            "QVECRADIUS": r"QVECRADIUS (.*?)\n",
            "QVC": r"QVC (.*?)\n",
        }

        second_entry = {"RMAT": "CENTRING", "TMAT": "POINTGROUP"}
        extracted_data = {}

        for key, pattern in patterns.items():
            matches = re.findall(pattern, text, re.DOTALL)
            for i, match in enumerate(matches):
                formatted_key = f"{key}_{i+1}" if key == "QVEC" and len(matches) > 1 else key
                if key in second_entry:
                    extracted_data[second_entry[key]] = match[0].strip()
                    extracted_data[formatted_key] = self.convert_to_numpy(match[1])
                else:
                    extracted_data[formatted_key] = self.convert_to_numpy(match)

        self.update(extracted_data)

    def value_as_string(self, key: str) -> str:
        """
        Converts a specific RMAT data value to a formatted string representation.

        Parameters
        ----------
        key : str
            The key for the data item in the dictionary.

        Returns
        -------
        str
            A string representation of the data item associated with the given key.
        """
        value = self[key]
        if isinstance(value, np.ndarray) and value.ndim == 2:
            if key == "RMAT":
                key_string = f'RMAT {self["CENTRING"]}\n'
            elif key == "TMAT":
                key_string = f'TMAT {self["CENTRING"]} {self["POINTGROUP"]}\n'
            else:
                key_string = key + " "
            return key_string + "\n".join("".join(f"{num: 12.7f}" for num in row) for row in value)
        elif isinstance(value, np.ndarray) and value.ndim == 1:
            key = "QVEC" if key.startswith("QVEC") else key
            return f'{key} {" ".join(f"{num:.5f}" for num in value)}'
        else:
            return key + " " + " ".join(value)

    def __str__(self) -> str:
        """
        String representation of RmatFile.

        Returns
        -------
        str
            Formatted string of the RMAT data.
        """
        output = [
            self.value_as_string(key) for key in self if key not in ("CENTRING", "POINTGROUP")
        ]
        return "\n".join(output)

    def to_cif_file(self, cif_path: str, block_name: str) -> None:
        """
        Writes a cif file from the data with the RMAT object.

        Parameters
        ----------
        cif_path : str
            Path of newly created cif file
        block_name :
            Name of data block within the cif file
        """

        cif = model.cif()
        block = model.block()
        for name, entry in self.to_cif_dict().items():
            block.add_data_item(name, entry)
        cif[block_name] = block
        cif_path = Path(cif_path)
        cif_path.write_text(str(cif), encoding="utf-8")

    def to_cif_dict(self) -> dict:
        """
        Converts the RMAT data to a dictionary in CIF format.

        Returns
        -------
        dict
            A dictionary containing the RMAT data in CIF format.
        """
        cif_dict = {
            "_diffrn_orient_matrix.ub_11": self["RMAT"][0][0],
            "_diffrn_orient_matrix.ub_12": self["RMAT"][0][1],
            "_diffrn_orient_matrix.ub_13": self["RMAT"][0][2],
            "_diffrn_orient_matrix.ub_21": self["RMAT"][1][0],
            "_diffrn_orient_matrix.ub_22": self["RMAT"][1][1],
            "_diffrn_orient_matrix.ub_23": self["RMAT"][1][2],
            "_diffrn_orient_matrix.ub_31": self["RMAT"][2][0],
            "_diffrn_orient_matrix.ub_32": self["RMAT"][2][1],
            "_diffrn_orient_matrix.ub_33": self["RMAT"][2][2],
            "_space_group.centring_type": self["CENTRING"],
        }
        if "TMAT" in self:
            cif_dict.update(
                {
                    "_diffrn_reflns_transf_matrix.11": self["TMAT"][0][0],
                    "_diffrn_reflns_transf_matrix.12": self["TMAT"][0][1],
                    "_diffrn_reflns_transf_matrix.13": self["TMAT"][0][2],
                    "_diffrn_reflns_transf_matrix.21": self["TMAT"][1][0],
                    "_diffrn_reflns_transf_matrix.22": self["TMAT"][1][1],
                    "_diffrn_reflns_transf_matrix.23": self["TMAT"][1][2],
                    "_diffrn_reflns_transf_matrix.31": self["TMAT"][2][0],
                    "_diffrn_reflns_transf_matrix.32": self["TMAT"][2][1],
                    "_diffrn_reflns_transf_matrix.33": self["TMAT"][2][2],
                }
            )

        if "POINTGROUP" in self:
            cif_dict["_space_group.point_group_h-m"] = self["POINTGROUP"]

        if "CELL" in self:
            cif_dict.update(
                {
                    "_cell.length_a": self["CELL"][0],
                    "_cell.length_b": self["CELL"][1],
                    "_cell.length_c": self["CELL"][2],
                    "_cell.angle_alpha": self["CELL"][3],
                    "_cell.angle_beta": self["CELL"][4],
                    "_cell.angle_gamma": self["CELL"][5],
                    "_cell.volume": self["CELL"][6],
                }
            )

        if "SIGMACELL" in self:
            cif_dict.update(
                {
                    "_cell.length_a_su": self["SIGMACELL"][0],
                    "_cell.length_b_su": self["SIGMACELL"][1],
                    "_cell.length_c_su": self["SIGMACELL"][2],
                    "_cell.angle_alpha_su": self["SIGMACELL"][3],
                    "_cell.angle_beta_su": self["SIGMACELL"][4],
                    "_cell.angle_gamma_su": self["SIGMACELL"][5],
                    "_cell.volume_su": self["SIGMACELL"][6],
                }
            )
        return cif_dict

    @classmethod
    def from_cif_file(cls, rmat_filename: str, cif_filename: str):
        """
        Creates an RmatFile instance from a CIF file.

        Parameters
        ----------
        rmat_filename : str
            The name for the RMAT file to create.
        cif_filename : str
            The name for the cif file to read from. Needs to use the
            unified keywords.
        cif_dict : dict
            A dictionary containing CIF data.

        Returns
        -------
        RmatFile
            An instance of RmatFile created from the CIF dictionary.
        """
        cif = reader(str(cif_filename))
        block = list(cif.model().values())[0]
        return cls.from_cif_dict(rmat_filename, block)

    @classmethod
    def from_cif_dict(cls, rmat_filename: str, cif_dict: dict) -> "RmatFile":
        """
        Creates an RmatFile instance from a CIF format dictionary.

        Parameters
        ----------
        filename : str
            The name for the RMAT file.
        cif_dict : dict
            A dictionary containing CIF data.

        Returns
        -------
        RmatFile
            An instance of RmatFile created from the CIF dictionary.
        """
        new = cls(rmat_filename)
        new["RMAT"] = np.array(
            [
                [float(cif_dict[f"_diffrn_orient_matrix.ub_{i}{j}"]) for j in range(1, 4)]
                for i in range(1, 4)
            ]
        )
        if "_space_group.centring_type" in cif_dict:
            new["CENTRING"] = cif_dict["_space_group.centring_type"]
        else:
            space_group_entries = (
                "_space_group.name_h-m_alt",
                "_space_group.name_h-m_full",
                "_space_group.name_hall",
            )
            for entry in space_group_entries:
                if entry in cif_dict:
                    new["CENTRING"] = entry[0]
                    break
            else:
                raise KeyError("No lattice centring found in cif (_space_group.centring_type)")

        tmat_entries = [
            f"_diffrn_reflns_transf_matrix.{i}{j}" for i, j in product(range(1, 4), repeat=2)
        ]
        tmat_entries.append("_space_group.point_group_h-m")
        if all(entry in cif_dict for entry in tmat_entries):
            new["TMAT"] = np.array(
                [
                    [float(cif_dict[f"_diffrn_reflns_transf_matrix.{i}{j}"]) for j in range(1, 4)]
                    for i in range(1, 4)
                ]
            )
            new["POINTGROUP"] = cif_dict["_space_group.point_group_h-m"]

        cell_entries = (
            "_cell.length_a",
            "_cell.length_b",
            "_cell.length_c",
            "_cell.angle_alpha",
            "_cell.angle_beta",
            "_cell.angle_gamma",
            "_cell.volume",
        )

        if all(entry in cif_dict for entry in cell_entries):
            new["CELL"] = np.array([float(cif_dict[entry]) for entry in cell_entries])

        sigmacell_entries = tuple(entry + "_su" for entry in cell_entries)
        if all(entry in cif_dict for entry in sigmacell_entries):
            new["SIGMACELL"] = np.array([float(cif_dict[entry]) for entry in sigmacell_entries])
        return new
