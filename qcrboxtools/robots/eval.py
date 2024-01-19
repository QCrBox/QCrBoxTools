from itertools import product
import re
from typing import Union, List

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

class TextFile:
    pass

class CommandFile:
    pass

class PicFile(dict):
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

    def __init__(self, content_str: str):
        """
        Initializes a new instance of the PicFile class.

        Parameters
        ----------
        content_str : str
            A string representation of the contents of a PIC file.
        """
        super().__init__()

        line_contents = [
            line.strip().split() for line in content_str.split('\n') if not line.startswith('!')
        ]
        options = []
        for line in line_contents:
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

    def to_file(self, file_path: str):
        """
        Writes the current state of the PIC file content to a specified file.

        Parameters
        ----------
        file_path : str
            The path to the file where the content should be written.
        """
        with open(file_path, 'w', encoding='UTF-8') as fobj:
            fobj.write(str(self))

    @classmethod
    def from_file(cls, file_path: str) -> 'PicFile':
        """
        Creates a PicFile instance from the contents of a specified file.

        This method reads the content of a PIC file from the given file path and
        initializes a PicFile object with this content.

        Parameters
        ----------
        file_path : str
            The path to the PIC file to be read.

        Returns
        -------
        PicFile
            An instance of PicFile initialized with the contents of the file.
        """
        with open(file_path, 'r', encoding='UTF-8') as fobj:
            content = fobj.read()
        return cls(content)


class VicFile(dict):
    """
    A class representing the content of a VIC file as a dictionary.

    This class allows for reading, modifying, and writing VIC file data.
    It can be initialized from a string representation of a VIC file or directly from
    a file.

    Methods
    -------
    __init__(content_str)
        Constructs a VicFile object from a string.
    __str__()
        Returns a formatted string representation of the VIC file content.
    from_file(file_path)
        Creates a VicFile instance from the contents of a specified file.
    to_file(file_path)
        Writes the VIC file content to a file.
    """

    def __init__(self, content_str: str):
        """
        Initializes a new instance of the VicFile class.

        Parameters
        ----------
        content_str : str
            A string representation of the contents of a VIC file.
        """
        super().__init__()
        for line in content_str.split('\n'):
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

    @classmethod
    def from_file(cls, file_path: str) -> 'VicFile':
        """
        Creates a VicFile instance from the contents of a specified file.

        Parameters
        ----------
        file_path : str
            The path to the VIC file to be read.

        Returns
        -------
        VicFile
            An instance of VicFile initialized with the contents of the file.
        """
        with open(file_path, 'r', encoding='UTF-8') as fobj:
            content = fobj.read()
        return cls(content)

    def to_file(self, file_path: str):
        """
        Writes the current state of the VIC file content to a specified file.

        Parameters
        ----------
        file_path : str
            The path to the file where the content should be written.
        """
        with open(file_path, 'w', encoding='UTF-8') as fobj:
            fobj.write(str(self))


class RmatFile(dict):
    """
    A class to handle RMAT file data extraction and manipulation.

    Attributes
    ----------
    data : dict
        Dictionary containing extracted data from RMAT file.

    Methods
    -------
    convert_to_numpy(array_str: str) -> np.ndarray
        Converts a string representation of a matrix to a numpy array.
    extract_data(text: str) -> dict
        Extracts and processes data from RMAT file text.
    value_as_string(key: str) -> str
        Converts data values to a formatted string.
    from_file(file_path: str) -> 'Rmat'
        Class method to create an instance from a file.
    to_file(file_path: str)
        Writes the RMAT data to a file.
    to_cif_dict() -> dict
        Converts RMAT data to a dictionary in CIF format.
    from_cif_dict(cif_dict: dict) -> 'Rmat'
        Class method to create an instance from a CIF dictionary.
    """

    def __init__(self, text: str = None):
        """Initialize RmatData with text from an RMAT file."""
        if text is not None:
            self.extract_data(text)

    @staticmethod
    def convert_to_numpy(array_str: str) -> np.ndarray:
        """
        Convert a string matrix to a numpy array.

        Parameters
        ----------
        array_str : str
            String containing matrix data.
        dimensions : int, optional
            Dimension of the numpy array (default is 2).

        Returns
        -------
        np.ndarray
            Numpy array representation of the string matrix.
        """
        lines = array_str.strip().split('\n')
        return np.squeeze(np.array([list(map(float, row.split())) for row in lines]))

    def extract_data(self, text: str) -> dict:
        """
        Extract and process data from the RMAT text.

        Parameters
        ----------
        text : str
            String containing the RMAT file content.

        Returns
        -------
        dict
            Dictionary with extracted data.
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
            # Add patterns for other entries here
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
        Convert data value to a formatted string.

        Parameters
        ----------
        key : str
            Key for the data item.

        Returns
        -------
        str
            Formatted string representation of the data item.
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
        String representation of Rmat.

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

    @classmethod
    def from_file(cls, file_path: str) -> 'RmatFile':
        """
        Create an Rmat instance from a file.

        Parameters
        ----------
        file_path : str
            Path to the RMAT file.

        Returns
        -------
        RmatData
            An instance of RmatData.
        """
        with open(file_path, encoding='UTF-8') as fobj:
            content = fobj.read()
        return cls(text=content)

    def to_file(self, file_path: str):
        """
        Write the RMAT data to a file.

        Parameters
        ----------
        file_path : str
            Path to the output file.
        """
        with open(file_path, 'w', encoding='UTF-8') as fobj:
            fobj.write(str(self))

    def to_cif_dict(self) -> dict:
        """
        Convert RMAT data to CIF format dictionary.

        Returns
        -------
        dict
            Dictionary with data in CIF format.
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
    def from_cif_dict(cls, cif_dict: dict) -> 'RmatFile':
        """
        Create an Rmat instance from a CIF dictionary.

        Parameters
        ----------
        cif_dict : dict
            Dictionary in CIF format.

        Returns
        -------
        Rmat
            An instance of Rmat created from CIF dictionary data.
        """
        new = cls()
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


class EvalAppRobot:
    def __init__(self, command_list, file_list):
        self.command_list = command_list
        self.file_list = file_list

    def integrate_shoes(self):
        pass

class EvalViewRobot:
    def __init__(self, command_list, file_list):
        self.command_list = command_list
        self.file_list = file_list

    def create_shoes(self):
        command_list = ['@datcol', 'exit']
