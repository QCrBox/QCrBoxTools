"""
This module serves as a Python interface for automating the Eval program,
for the integration and data reduction of crystallographic
diffraction images. It offers a suite of classes that help reading and
modifying and writing the necessary components for a simple
interaction with different components of the Eval program, including Eval15All,
EvalView, and other related functionalities.

Classes
-------
RelativePathFile
    A base class for handling file operations using relative paths.
TextFile
    Extends RelativePathFile for operations specific to text files.
PicFile
    Manages PIC file data, enabling reading and writing operations.
SettingsVicFile
    Handles operations specific to (settings) VIC file data.
RmatFile
    Provides functionalities for RMAT file data manipulation and CIF format conversion.
Eval15AllRobot
    Automates the Eval15All process, managing the integration of diffraction images.
EvalViewRobot
    Automates tasks related to the EvalView component of the Eval program,
    focusing on the preparation and processing of diffraction data.
EvalAnyRobot
    Manages the automation of generating CIF files from Evals integrated data
    using the 'any' command. This class also can be used to write .sad files
    that can be read into SADABS.

Example
-------
A workflow using this module to automate Eval program tasks for
crystallographic data processing. This does not include the modification of
parameters at the moment. Therefore the presented workflow is more instructional
than of actual use:

    from qcrboxtools.robots.eval import (
        SettingsVicFile, TextFile, RmatFile, EvalViewRobot, PicFile, Eval15AllRobot
    )
    import pathlib
    import shutil

    # Setting up working and source directories
    work_dir = pathlib.Path('/new/folder')
    source_dir = pathlib.Path('/original/integration/folder')
    work_dir.mkdir(exist_ok=True)

    # Copying .oxf files (diffraction images) to the working directory
    frame_ending = '.oxf'
    for filename in source_dir.glob(f'*{frame_ending}'):
        shutil.copy(filename, work_dir / filename.relative_to(source_dir))

    # Loading RMAT, beamstop, and detalign files
    rmat_file = RmatFile.from_file(next(source_dir.glob('ic.rmat')))
    beamstop_file = SettingsVicFile.from_file(next(source_dir.glob('beamstop.vic')))
    detalign_file = SettingsVicFile.from_file(next(source_dir.glob('detalign.vic')))

    # Loading datcol files
    datcol_files = [TextFile.from_file(path) for path in sorted(source_dir.glob('datcol*.vic'))]

    # Creating and executing EvalViewRobot for data preparation
    evalview = EvalViewRobot(work_folder=work_dir, file_list=[
        rmat_file, beamstop_file, detalign_file, *datcol_files
    ])
    evalview.create_shoes()

    # Loading PIC files
    pic_files = [PicFile.from_file(path) for path in source_dir.glob('ic/*.pic')]

    # you would probably modify the pic files here before integration
    # ...

    #execute Eval15AllRobot for image integrat
    eval15 = Eval15AllRobot(work_folder=work_dir / 'ic', file_list=pic_files)
    eval15.integrate_shoes()
"""

from itertools import product
import re
from typing import Union, List, Optional, Tuple, Dict
import os
from pathlib import Path
import subprocess

import numpy as np

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
    if s.lower() in ['true', 'false']:
        return s.lower() == 'true'

    # Try to cast to numpy array (if contains space-separated values)
    if ' ' in s:
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
    def from_file(cls, file_path: str) -> 'RelativePathFile':
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
        with path.open('r', encoding='UTF-8') as fobj:
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
        with file_path.open('w', encoding='UTF-8') as fobj:
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


class PicFile(RelativePathFile, dict):
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
        'abort', 'adcnoise', 'addconstant', 'ambiguity', 'animo', 'anivec', 'autofirst',
        'autorestore', 'badresponsthreshold', 'batch', 'beamgridsize', 'beampixelsize',
        'beampos', 'boxclose', 'boxfixed', 'boximpactdiff', 'boxnext', 'boxopen', 'boxscan',
        'boxsize', 'cell', 'centraloverlap', 'changecoef', 'check', 'clear', 'cntclose',
        'cntfile', 'cntwrite', 'collimatordiameter', 'collimatortheight', 'collimatortype',
        'collimatorwidth', 'colour', 'colourscheme', 'contour', 'correct', 'cutoff', 'cvectorrot',
        'debugimpact', 'delay', 'dict', 'dictname', 'diffarg', 'diffracpos', 'diffdividesigma',
        'diffscale', 'difftype', 'dirax', 'diraxclose', 'diraxopen', 'dist', 'distribution',
        'divmax', 'draw', 'dump', 'dur', 'edgefraction', 'epsilon', 'eval14', 'eval14pointspread',
        'eval14threshold', 'eval14type', 'exhor', 'exit', 'exrot', 'exver', 'facetest',
        'faceconvert', 'fbg', 'fbragg', 'fibre', 'fibreaxis', 'file', 'fix', 'flex', 'focfile',
        'focus', 'focuschange', 'focuslambda', 'focustype', 'follow', 'fomtype', 'force',
        'forever', 'framefraction', 'free', 'gain', 'go', 'goniostatchange', 'goniostatprint',
        'goniostattype', 'graphics', 'gravity', 'gravitycycles', 'gravitytype', 'help', 'hkl',
        'ignore', 'imagefilename', 'impact', 'impactcolourtype', 'impactdotsize', 'impactlabel',
        'impactnb', 'impactrow', 'incidencecoefficient', 'incidencecoefficient2',
        'incidencecoefficient3', 'inner', 'intermediate', 'isigrefine', 'kappasupport',
        'lambdaadd', 'lambdaaddfull', 'lambdacalcmean', 'lambdahigh', 'lambdainit', 'lambdalow',
        'lambdameanmid', 'lambdaprint', 'lambdarange', 'lambdasigma', 'lambdatype', 'lambdavalue',
        'lambdaweight', 'largelegenda', 'latt', 'lattfraction', 'latthigh', 'lattiso', 'lattlow',
        'latttype', 'lattvec', 'legenda', 'limits', 'link', 'listoff', 'liston', 'loadexppic',
        'loadrefine', 'loadxtalevc', 'lock', 'longdelay', 'low', 'lpfinal', 'lsqtype', 'manual',
        'markdefault', 'maxtry', 'menu', 'mica', 'micahigh', 'micalow', 'micascale', 'micatype',
        'micavec', 'model', 'mosaic', 'mosaicadd', 'mosaicinit', 'mosaicrotangle',
        'mosaicrotaxis', 'mosaictype', 'mosaicweight', 'msa1', 'msa2', 'mu', 'nbcommonfraction',
        'nbcovariance', 'nbdeltaindex', 'nbeffective', 'nbexpandhv', 'nbexpandr',
        'nboverlapfraction', 'nbsimfraction', 'nbstillfraction', 'nbtype', 'nbvolume', 'ncell',
        'ndivinner', 'ndivouter', 'netto', 'newworld', 'next', 'nmosaic', 'nrefineouter', 'nop',
        'noprintcontour', 'normalize', 'nrmat', 'obsminfrac', 'oneframe', 'onerot', 'onescan',
        'outer', 'output', 'overflowthreshold', 'penalty', 'pick', 'play', 'plot', 'plot0',
        'plot000', 'plotrot', 'pointspreadgammma', 'pointspreadmoment', 'pointspreadthreshold',
        'pointspreadtype', 'polarisation', 'pq', 'printcontour', 'printextreme', 'printframes',
        'printprojection', 'printposmin', 'printsliceframe', 'printslicelimit', 'printslices',
        'printxtalsample', 'projection', 'ps', 'qhor', 'qrefine', 'qrot', 'qshift', 'qvec',
        'qver', 'randomcache', 'randominit', 'randomseed', 'read', 'readcorners', 'readdetalign',
        'readfaces', 'readfcalc', 'readicalc', 'readins', 'refine', 'refinebox', 'refinefile',
        'refineouter', 'refineshift', 'refinestatus', 'refinetol', 'refinetol2', 'refinetypeinner',
        'refinetypeouter', 'refinewrite', 'reflnr', 'reject', 'reset', 'responsfraction',
        'restore', 'rmat', 'rock', 'rotatermat', 'rotaxnr', 'sampleplot', 'sampleplotcolourtype',
        'sampleplotdotsize', 'sampleplotfloor', 'sampleplotscale', 'save', 'savemeanshift',
        'savemeanshiftfloor', 'scaletype', 'scanfomtype', 'searchanivec', 'sensordepth',
        'sensoroffset', 'shextra', 'shhor', 'shift', 'shiftgrid', 'shiftrestfracmin', 'shiftrmat',
        'shinit', 'shinitsector', 'show', 'showfaces', 'shrot', 'shver', 'simulate', 'sliceclose',
        'slicefactor', 'slicefile', 'slicethreshold', 'slicewrite', 'slicewritefilter',
        'splitshift', 'square', 'still', 'stillframes', 'stillinc', 'stilltype', 'store',
        'storestatus', 'sumhkl', 'svdtol', 'swing', 'sync', 'target', 'targetname', 'test',
        'trace', 'unlink', 'vecscan', 'vecscanfile', 'vector', 'view', 'viewrot', 'volcortype',
        'wait', 'weightscheme', 'window', 'xa', 'xb', 'xc', 'xrot', 'xrotx', 'xroty', 'xrotz',
        'xtalax', 'xtalcfac', 'xtalplot', 'xtalsamplesize', 'xtalscale', 'xtalshift', 'xtalvec',
        'xyztohkl', 'zero', 'zoom', 'zoomfraction', 'zoompick', 'zoomtitle'
    ]

    _commands_with_X = [
        'Sample{}', 'dur{}', 'exhor{}', 'exrot{}', 'exver{}', 'lambdahigh{}', 'lambdalow{}',
        'lambdasigma{}', 'lambdatype{}', 'lambdavalue{}', 'lambdaweight{}', 'lambda{}', 'lamsig{}',
        'lamwght{}', 'lattice{}', 'mosaicrotangle{}', 'mosaicrotaxis{}', 'mosaicweight{}',
        'mosaic{}', 'mosrot{}', 'moswght{}', 'rmat{}', 'shextra{}', 'shhor{}', 'shiftrmat{}',
        'shinitsector{}', 'shinit{}', 'shrot{}', 'shver{}'
    ]

    _eval15_commands += [
        cmd.format(number) for cmd, number in product(_commands_with_X, range(2, 10))
    ]

    def __init__(self, filename, content_str: str,):
        """
        Initializes a new instance of the PicFile class.

        Parameters
        ----------
        content_str : str
            A string representation of the contents of a PIC file.
        """
        super().__init__(filename=filename, content=content_str)

        line_contents = [
            line.strip().split() for line in content_str.split('\n') if not line.startswith('!')
        ]
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
        entry_strs = (self.formatted_entry(key) for key in self)
        return '\n'.join(entry_strs)

    def add_data(self, new_key: str, options: List[Union[int, float, bool, str]]):
        """
        Adds a new command to the PIC file data or updates an existing command.

        Parameters
        ----------
        new_key : str
            The command key to add or update.
        options : List[Union[int, float, bool, str]]
            The list of values associated with the command.
        """
        if len(options) == 0:
            self[new_key] = None
        elif len(options) == 1:
            if new_key in self and isinstance(self[new_key], list):
                self[new_key].append(options[0])
            elif new_key in self:
                self[new_key] = [self[new_key], options[0]]
            else:
                self[new_key] = options[0]
        else:
            if new_key in self and isinstance(self[new_key], list):
                self[new_key] += options
            elif new_key in self:
                self[new_key] = [self[new_key]] + options
            else:
                self[new_key] = options

    def formatted_entry(self, key: str) -> str:
        """
        Formats a single data entry for output as a string.

        Parameters
        ----------
        key : str
            The command key to format.

        Returns
        -------
        str
            The formatted string for the specified command key.
        """
        entry = self[key]
        if entry is None:
            return key
        elif isinstance(entry, list):
            entry_str = ' '.join(str(val) for val in entry)
        else:
            entry_str = str(entry)
        return f'{key} {entry_str}'


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
        for line in content.split('\n'):
            if not line.startswith('!'):
                parts = line.split()
                if len(parts) > 1:
                    self[parts[0]] = infer_and_cast(' '.join(parts[1:]))

    def __str__(self) -> str:
        """
        Returns a formatted string representation of the VIC file content.

        Returns
        -------
        str
            The formatted content of the VIC file.
        """
        return '\n'.join(
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
        lines = array_str.strip().split('\n')
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
            'RMAT': r'RMAT ([ABCFIPR])\n(.*?\n.*?\n.*?)\n',
            'TMAT': r'TMAT .*? (.*?)\n(.*?\n.*?\n.*?)\n',
            'CELL': r'\nCELL ([\d\s\.\-]+)',
            'SIGMACELL': r'SIGMACELL ([\d\s\.\-]+)',
            'QVECType': r'QVECType (.*?)\n',
            'QVEC': r'QVEC (.*?)\n',
            'QVECRADIUS': r'QVECRADIUS (.*?)\n',
            'QVC': r'QVC (.*?)\n',
        }

        second_entry = {'RMAT': 'CENTRING', 'TMAT': 'POINTGROUP'}
        extracted_data = {}

        for key, pattern in patterns.items():
            matches = re.findall(pattern, text, re.DOTALL)
            for i, match in enumerate(matches):
                formatted_key = f'{key}_{i+1}' if key == 'QVEC' and len(matches) > 1 else key
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
            if key == 'RMAT':
                key_string = f'RMAT {self["CENTRING"]}\n'
            elif key == 'TMAT':
                key_string = f'TMAT {self["CENTRING"]} {self["POINTGROUP"]}\n'
            else:
                key_string = key + ' '
            return key_string + '\n'.join(''.join(f"{num: 12.7f}" for num in row) for row in value)
        elif isinstance(value, np.ndarray) and value.ndim == 1:
            key = 'QVEC' if key.startswith('QVEC') else key
            return f'{key} {" ".join(f"{num:.5f}" for num in value)}'
        else:
            return key + ' ' + ' '.join(value)

    def __str__(self) -> str:
        """
        String representation of RmatFile.

        Returns
        -------
        str
            Formatted string of the RMAT data.
        """
        output = [
            self.value_as_string(key) for key in self
            if key not in ('CENTRING', 'POINTGROUP')
        ]
        return '\n'.join(output)

    def to_cif_dict(self) -> dict:
        """
        Converts the RMAT data to a dictionary in CIF format.

        Returns
        -------
        dict
            A dictionary containing the RMAT data in CIF format.
        """
        cif_dict = {
            '_diffrn_orient_matrix.UB_11': self['RMAT'][0][0],
            '_diffrn_orient_matrix.UB_12': self['RMAT'][0][1],
            '_diffrn_orient_matrix.UB_13': self['RMAT'][0][2],
            '_diffrn_orient_matrix.UB_21': self['RMAT'][1][0],
            '_diffrn_orient_matrix.UB_22': self['RMAT'][1][1],
            '_diffrn_orient_matrix.UB_23': self['RMAT'][1][2],
            '_diffrn_orient_matrix.UB_31': self['RMAT'][2][0],
            '_diffrn_orient_matrix.UB_32': self['RMAT'][2][1],
            '_diffrn_orient_matrix.UB_33': self['RMAT'][2][2],
            '_space_group.centring_type': self['CENTRING']
        }
        if 'TMAT' in self:
            cif_dict.update({
                '_diffrn_reflns_transf_matrix.11': self['TMAT'][0][0],
                '_diffrn_reflns_transf_matrix.12': self['TMAT'][0][1],
                '_diffrn_reflns_transf_matrix.13': self['TMAT'][0][2],
                '_diffrn_reflns_transf_matrix.21': self['TMAT'][1][0],
                '_diffrn_reflns_transf_matrix.22': self['TMAT'][1][1],
                '_diffrn_reflns_transf_matrix.23': self['TMAT'][1][2],
                '_diffrn_reflns_transf_matrix.31': self['TMAT'][2][0],
                '_diffrn_reflns_transf_matrix.32': self['TMAT'][2][1],
                '_diffrn_reflns_transf_matrix.33': self['TMAT'][2][2]
            })

        if 'POINTGROUP' in self:
            cif_dict['_space_group.point_group_h-m'] = self['POINTGROUP']

        if 'CELL' in self:
            cif_dict.update({
                '_cell.length_a': self['CELL'][0],
                '_cell.length_b': self['CELL'][1],
                '_cell.length_c': self['CELL'][2],
                '_cell.angle_alpha': self['CELL'][3],
                '_cell.angle_beta': self['CELL'][4],
                '_cell.angle_gamma': self['CELL'][5],
                '_cell.volume': self['CELL'][6]
            })

        if 'SIGMACELL' in self:
            cif_dict.update({
                '_cell.length_a_su': self['SIGMACELL'][0],
                '_cell.length_b_su': self['SIGMACELL'][1],
                '_cell.length_c_su': self['SIGMACELL'][2],
                '_cell.angle_alpha_su': self['SIGMACELL'][3],
                '_cell.angle_beta_su': self['SIGMACELL'][4],
                '_cell.angle_gamma_su': self['SIGMACELL'][5],
                '_cell.volume_su': self['SIGMACELL'][6]
            })
        return cif_dict

    @classmethod
    def from_cif_dict(cls, filename: str, cif_dict: dict) -> 'RmatFile':
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
        new = cls(filename)
        new['RMAT'] = np.array(
            [
                [cif_dict[f'_diffrn_orient_matrix.UB_{i}{j}'] for j in range(1,4)]
                for i in range(1,4)
            ]
        )
        new['CENTRING'] = cif_dict['_space_group.centring_type']

        tmat_entries = [
            f'_diffrn_reflns_transf_matrix.{i}{j}' for i, j in product(range(1,4), repeat=2)
        ]
        tmat_entries.append('_space_group.point_group_h-m')
        if all(entry in cif_dict for entry in tmat_entries):
            new['TMAT'] = np.array(
                [
                    [cif_dict[f'_diffrn_reflns_transf_matrix.{i}{j}'] for j in range(1,4)]
                    for i in range(1,4)
                ]
            )
            new['POINTGROUP'] = cif_dict['_space_group.point_group_h-m']

        cell_entries = (
            '_cell.length_a', '_cell.length_b', '_cell.length_c',
            '_cell.angle_alpha', '_cell.angle_beta', '_cell.angle_gamma',
            '_cell.volume'
        )

        if all(entry in cif_dict for entry in cell_entries):
            new['CELL'] = np.array([cif_dict[entry] for entry in cell_entries])

        sigmacell_entries = tuple(entry + '_su' for entry in cell_entries)
        if all(entry in cif_dict for entry in sigmacell_entries):
            new['SIGMACELL'] = np.array([cif_dict[entry] for entry in sigmacell_entries])
        return new


class EvalBaseRobot:
    """
    Base class for automating execution of various programs in crystallography data processing.

    This class provides foundational functionalities for derived classes to automate
    the execution of crystallography-related programs with specified command sequences.

    Attributes
    ----------
    work_folder : Path
        The directory where the automated processes are executed.

    Methods
    -------
    __init__(self, work_folder: Union[str, Path])
        Initializes EvalBaseRobot with a specified work folder.
    _run_program_with_commands(self, program_name: str, command_list: List[str])
        Executes a specified program with a list of commands in the work folder.
    """
    def __init__(self, work_folder: Union[str, Path]):
        """
        Initializes the EvalBaseRobot with a specified work folder.

        Parameters
        ----------
        work_folder : Union[str, Path]
            The directory where the automated processes are to be executed.
        """
        self.work_folder = Path(work_folder)

    def _run_program_with_commands(self, program_name: str, command_list: List[str]):
        """
        Executes a specified program with a list of commands in the work folder.

        This method creates an initialization file with the provided commands and
        executes the specified program, capturing its output in a log file.

        Parameters
        ----------
        program_name : str
            The name of the program to be executed.
        command_list : List[str]
            A list of commands to be written to the program's initialization file.
        """
        init_file = self.work_folder / f'{program_name}.init'
        with init_file.open('w', encoding='UTF-8') as fobj:
            fobj.write('\n'.join(command_list))
            fobj.write('\n')

        with open(self.work_folder / f'{program_name}_output.log', 'w', encoding='UTF-8') as fobj:
            subprocess.call(program_name, cwd=self.work_folder, stdout=fobj, stderr=fobj)


class Eval15AllRobot(EvalBaseRobot):
    """
    A class to automate the integration process using the 'eval15all' command.

    This class is designed to handle the automation of data integration tasks
    in a specified work folder using a list of input files.

    Attributes
    ----------
    work_folder : Path
        The directory where the integration process is to be executed.
    file_list : List[PicFile]
        A list of file objects to be processed.

    Methods
    -------
    __init__(work_folder: Union[str, Path], file_list: List[PicFile])
        Initializes an Eval15AllRobot with a work folder and a list of files.
    integrate_shoes()
        Executes the integration process using 'eval15all'.
    """
    def __init__(
            self,
            work_folder: Union[str, Path],
            file_list: List[PicFile]
        ):
        """
        Initializes the Eval15AllRobot with a specified work folder and a list of files.

        Parameters
        ----------
        work_folder : Union[str, Path]
            The directory where the integration process will be executed. The subdirectory
            usually has the name of the used .rmat file, e.g. ic
        file_list : List[PicFile]
            A list of file objects that are to be included in the integration process.
        """
        super().__init__(work_folder)
        self.file_list = file_list

    def integrate_shoes(self):
        """
        Executes the integration process using the 'eval15all' command.

        This method writes the files in the file list to the work directory,
        creates an initialization file for 'eval15all', and runs the integration process.
        """
        for file in self.file_list:
            file.to_file(self.work_folder)

        command_list = [f'@{file.filename}' for file in self.file_list]
        command_list.append('markdefault')
        self._run_program_with_commands('eval15all', command_list)

class EvalViewRobot(EvalBaseRobot):
    """
    A class to automate using the 'view' command.

    This class is responsible for automating the process of preparing data
    and executing the 'view' command in a specified work folder.

    Attributes
    ----------
    work_folder : Path
        The directory where the visualization process is to be executed.
    file_list : List[Union[TextFile, PicFile, SettingsVicFile, RmatFile]]
        A list of file objects to be processed.

    Methods
    -------
    __init__(work_folder: Union[str, Path], file_list: List[Union[TextFile,
        SettingsVicFile, RmatFile]])
        Initializes an EvalViewRobot with a work folder and a list of files.
    create_shoes()
        Prepares and executes the visualization process using 'view'.
    """
    def __init__(
            self,
            work_folder: Union[str, Path],
            file_list: List[Union[TextFile, PicFile, SettingsVicFile, RmatFile]]
        ):
        """
        Initializes the EvalViewRobot with a specified work folder and a list of files.

        Parameters
        ----------
        work_folder : Union[str, Path]
            The directory where the process will be executed.
        file_list : List[Union[TextFile, SettingsVicFile, RmatFile]]
            A list of file objects that are to be included in the process.
        """
        super().__init__(work_folder)
        self.file_list = file_list

    def create_shoes(self):
        """
        Prepares and executes the creation of shoe files process using the 'view' command.

        This method writes the files in the file list to the work directory,
        creates an initialization file for 'view', and runs the visualization process.
        """
        command_list = ['@datcol', 'exit']

        for file in self.file_list:
            file.to_file(self.work_folder)

        self._run_program_with_commands('view', command_list)

class EvalAnyRobot(EvalBaseRobot):
    """
    A class designed to automate the creation of CIF (Crystallographic Information File)
    files and dictionaries from data using Eval's 'any' command within a specified work folder.

    This class manages the processes involved in creating the 'sad' file, reading data,
    and generating CIF files and dictionaries.

    Attributes
    ----------
    work_folder : Path
        The directory where the 'any' command and related file operations will be executed.

    Methods
    -------
    __init__(self, work_folder: Union[str, Path])
        Initializes the EvalAnyRobot with a specified work folder.
    create_abs(self)
        Executes Eval's 'any' command to generate necessary data for CIF file creation.
    create_cif_dict(self) -> Dict[str, List[Union[int, float]]]
        Creates a CIF format dictionary by calling any and transforming
    create_cif_file(self, file_path: Union[str, Path])
        Generates a CIF file at the specified file path via any export
    """


    _abs_columns = (
        # name, type, n_chars
        ('_diffrn_refln.index_h', int, 4),
        ('_diffrn_refln.index_k', int, 4),
        ('_diffrn_refln.index_l', int, 4),
        ('_diffrn_refln.intensity_net', float, 8),
        ('_diffrn_refln.intensity_net_su', float, 8),
        ('_diffrn_refln.class_code', int, 4),
        ('_qcrbox.diffrn_refln.direction_cosine_incid_x', float, 8),
        ('_qcrbox.diffrn_refln.direction_cosine_incid_y', float, 8),
        ('_qcrbox.diffrn_refln.direction_cosine_incid_z', float, 8),
        ('_qcrbox.diffrn_refln.direction_cosine_diffrn_x', float, 8),
        ('_qcrbox.diffrn_refln.direction_cosine_diffrn_y', float, 8),
        ('_qcrbox.diffrn_refln.direction_cosine_diffrn_z', float, 8),
        ('_qcrbox.diffrn_refln.detector_px_x_obs', float, 7),
        ('_qcrbox.diffrn_refln.detector_px_y_obs', float, 7),
        ('_qcrbox.diffrn_refln.detector_frame_obs', float, 8),
        ('_qcrbox.diffrn_refln.evalsad_mystery_val1', float, 7),
        ('_qcrbox.diffrn_refln.evalsad_mystery_val2', int, 5)
    )

    def create_abs(self):
        """
        Executes the 'any' command to generate data necessary for subsequent processing with
        SADABS.
        """
        command_list = ['read final', 'sadabs', 'exit']
        self._run_program_with_commands('any', command_list)

    def create_cif_dict(self):
        """
        Creates a CIF format dictionary by calling 'any' and reading the output.

        Returns
        -------
        Dict[str, List[Union[int, float]]]
            A dictionary with CIF format data where keys are column names and values are
            lists of data.
        """
        self.create_abs()

        sad_path = self.work_folder / 'shelx.sad'
        with open(sad_path, encoding='UTF-8') as fobj:
            content = fobj.read()

        column_names, types, widthes = zip(*self._abs_columns)

        ends = [sum(widthes[:index]) for index in range(len(widthes) + 1)]

        cast_values = [
            [
                type(line[start:stop]) for start, stop, type in zip(ends[:-1], ends[1:], types)
            ] for line in content.split('\n') if len(line) >= ends[-1]
        ]

        cif_dict = {
            key: values for key, values in zip(column_names, zip(*cast_values))
        }
        return cif_dict

    def create_cif_file(self, file_path: Union[str, Path]):
        """
        Generates a CIF file at the specified file path that contains the any output.

        Parameters
        ----------
        file_path : Union[str, Path]
            The file path where the CIF file will be written.
        """
        cif_dict = self.create_cif_dict()

        # there are only int and float type entries so if not int -> float
        format_strings = (
            f'{{:{entry[2]}d}}' if entry[1] == int else f'{{:.{entry[2]}f}}'
            for entry in self._abs_columns
        )

        line_format_string = ' '.join(format_strings)

        file_lines = [
            r'#\#CIF_2.0', '', 'data_eval_output', '', 'loop_'
        ]

        file_lines += list(cif_dict.keys())

        file_lines += [
            line_format_string.format(*line_entries) for line_entries in zip(*cif_dict.values())
        ]

        file_lines.append('')

        with open(file_path, 'w', encoding='UTF-8') as fobj:
            fobj.write('\n'.join(file_lines))

    def create_pk(self):
        """
        Creates a final.pk file containing the peak information from the integration.
        Needed for final cell refinement in peakref
        """
        command_list = ['read final', 'pkrestfrac 0.2', 'pk', 'exit']
        self._run_program_with_commands('any', command_list)

class EvalPeakrefRobot(EvalBaseRobot):
    """
    A class for automating refinement of cell and diffractometer parameters using the
    'peakref' command.

    This class extends EvalBaseRobot to facilitate the refinement of crystallographic
    parameters, specifically focusing on peak refinement and cell parameter extraction
    using various strategies in the context of the 'peakref' program.

    Attributes
    ----------
    rmat_file : RmatFile
        An RmatFile object containing RMAT file information for refinement.

    Methods
    -------
    __init__(self, work_folder: Union[str, Path], rmat_file: Union[RmatFile, str])
        Initializes EvalPeakrefRobot with a work folder and an RmatFile.
    refine_parameters(
        self,
        peakfile_path: str,
        refinement_strategy: Union[Tuple[Tuple[str]], str] = 'default',
        point_group_tolerance: Optional[Tuple[float, ...]] = None,
        end_with_cell: bool = False,
        new_rmat_filename: Optional[str] = None,
        rewrite_detalign: bool = True,
        rewrite_goniostat: bool = True
    )
        Executes refinement processes based on specified parameters and strategies.
    cell_cif_from_log(self) -> Dict[str, float]
        Extracts cell parameters from the 'peakref' output log in CIF format.
    folder_to_cif(self) -> Dict[str, float]
        Consolidates RMAT and cell parameter data into a CIF format dictionary.
    """

    def __init__(
        self,
        work_folder: Union[str, Path],
        rmat_file: Union[RmatFile, str]
    ):
        """
        Initializes the EvalPeakrefRobot with a specified work folder and RmatFile.

        Parameters
        ----------
        work_folder : Union[str, Path]
            The directory where the refinement process will be executed.
        rmat_file : Union[RmatFile, str]
            An RmatFile object or the path to an RMAT file used for refinement.
        """
        super().__init__(work_folder)
        if isinstance(rmat_file, RmatFile):
            self.rmat_file = rmat_file
        else:
            self.rmat_file = RmatFile.from_file(rmat_file)


    def refine_parameters(
        self,
        peakfile_path: str,
        refinement_strategy: Union[Tuple[Tuple[str]], str] = 'default',
        point_group_tolerance: Optional[Tuple[float, ...]] = None,
        end_with_cell: bool = True,
        new_rmat_filename: Optional[str] = None,
        rewrite_detalign: bool = True,
        rewrite_goniostat: bool = True
    ):
        """
        Executes the refinement process using the 'peakref' command with specified parameters.

        Parameters
        ----------
        peakfile_path : str
            Path to the peak file used for refinement.
        refinement_strategy : Union[Tuple[Tuple[str]], str], optional
            The strategy for refinement, if 'default' use a default ordering of the refinement.
            Otherwise use a tuple of steps where each step is represented by a tuple of strings
            containing the parameters that are to be refined in that step..
        point_group_tolerance : Optional[Tuple[float, ...]], optional
            Tolerance values for point group refinement, if applicable. Expects two floats for
            the tolerance for point group determination for 1. the cell length parameters in
            angstrom and 2. the cell angle parameters in degree
        end_with_cell : bool, default False
            Whether to end refinement with cell parameter with everything else fixed
        new_rmat_filename : Optional[str], default None
            The filename for the new RMAT file to be saved. If None, uses the original RMAT file's
            name.
        rewrite_detalign : bool, default True
            Whether to save the updated detector alignment to 'detalign.vic'.
        rewrite_goniostat : bool, default True
            Whether to save the updated goniostat settings to 'goniostat.vic'.
        """
        if refinement_strategy == 'default':
            refinement_strategy = (
                ('zerohor', 'zerover'),
                ('rmat',),
                ('detrot',),
                ('zerodist',)
            )
        command_list = []

        if new_rmat_filename is None:
            new_rmat_filename = self.rmat_file.filename
        self.rmat_file.filename = 'transfer.rmat'
        self.rmat_file.to_file(self.work_folder)

        command_list.append(f'rmat {self.rmat_file.filename}')
        command_list.append(f'pk {peakfile_path}')

        # fix everything
        command_list.append('fix all')

        # for each entry in refinement state
        for step in refinement_strategy:
            for variable in step:
                command_list.append(f'free {variable}')
            command_list.append('gox')

        if point_group_tolerance is not None:
            number_string = ' '.join(str(val) for val in point_group_tolerance)
            command_list.append(f'pgzero {number_string}')
            command_list.append('gox')
            command_list.append('reind')
            command_list.append('gox')

        if end_with_cell:
            command_list.append('fix all')
            command_list.append('free cell')
            command_list.append('sigrnd 0.1 50')

        if rewrite_detalign:
            command_list.append('save detalign.vic')

        if rewrite_goniostat:
            command_list.append('savegonio goniostat.vic')

        command_list.append(f'savermat {new_rmat_filename}')

        command_list.append('exit')

        self._run_program_with_commands('peakref', command_list)

        os.remove(self.work_folder / 'transfer.rmat')

        self.rmat_file = RmatFile.from_file(self.work_folder / new_rmat_filename)

    def cell_cif_from_log(self) -> Dict[str, float]:
        """
        Extracts cell parameters from the peakref output log and returns them in CIF format.

        This method reads the 'peakref_output.log' file, parses the refined cell parameters,
        and converts them into a dictionary formatted for CIF (Crystallographic Information File).

        Returns
        -------
        Dict[str, Tuple[float, float]]
            A dictionary where keys are CIF parameter names and values are tuples containing
            the parameter value and its standard uncertainty.

        Raises
        ------
        FileNotFoundError
            If the 'peakref_output.log' file does not exist in the working directory.
        """
        log_file = self.work_folder / 'peakref_output.log'

        if not log_file.exists():
            raise FileNotFoundError('You need to run refine_parameters with end_with_cell first')

        with log_file.open('r', encoding='UTF-8') as fobj:
            content = fobj.read()

        refined_pattern = (
            r'\n\s+(a(?:\s+[A-Za-z]+){1,6})\n'
            + r'refined((?:\s+\d+\.\d+){1,7})\n'
            + r'sigma((?:\s+\d+\.\d+){1,7})'
        )
        par_string, val_string, su_string = re.search(refined_pattern, content).groups()

        ref_zip = zip(
            par_string.strip().split(),
            val_string.strip().split(),
            su_string.strip().split()
        )

        cell_dict = {name: (float(val), float(su)) for name, val, su in ref_zip}

        cell_parameters = ('a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Volume')

        for par in cell_parameters:
            pattern = rf'{par} constrained to \[([A-Za-z]+)\]\.'
            search_result = re.search(pattern, content)
            if search_result is not None:
                lookup = search_result.group(1)
                cell_dict[par] = cell_dict[lookup]

        fixed_parameters = (par for par in cell_parameters if par not in cell_dict)
        for par in fixed_parameters:
            pattern = rf'{par}\s+Fix\s+(\d+\.\d+)'
            cell_dict[par] = (float(re.findall(pattern, content)[-1]), 0.0)

        cif_names = {
            '_cell.length_a': 'a',
            '_cell.length_b': 'b',
            '_cell.length_c': 'c',
            '_cell.angle_alpha': 'alpha',
            '_cell.angle_beta': 'beta',
            '_cell.angle_gamma': 'gamma',
            '_cell.volume': 'Volume'
        }

        cif_dict = {}
        for cif_name, eval_name in cif_names.items():
            eval_val, eval_su = cell_dict[eval_name]
            cif_dict[cif_name] = eval_val
            cif_dict[cif_name + '_su'] = eval_su

        return cif_dict

    def folder_to_cif(self) -> Dict[str, float]:
        """
        Consolidates RMAT and cell parameter data into a CIF format dictionary.

        This method combines the data from the RMAT file and the refined cell parameters
        extracted from the 'peakref_output.log' file into a single dictionary formatted
        for CIF (Crystallographic Information File).

        Returns
        -------
        Dict[str, float]
            A dictionary with CIF-formatted data, including both RMAT file data and
            refined cell parameters.
        """
        cif_dict = self.rmat_file.to_cif_dict()
        cif_dict.update(self.cell_cif_from_log())
        return cif_dict
