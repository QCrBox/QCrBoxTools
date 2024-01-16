from itertools import product
import re

import numpy as np


class TextFile:
    pass

class CommandFile:
    pass

class PicFile:
    pass

class BeamStopFile:
    pass

class DetAlignFile:
    pass

class Rmat:
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
    from_rmat_file(file_path: str) -> 'Rmat'
        Class method to create an instance from a file.
    to_rmat_file(file_path: str)
        Writes the RMAT data to a file.
    to_cif_dict() -> dict
        Converts RMAT data to a dictionary in CIF format.
    from_cif_dict(cif_dict: dict) -> 'Rmat'
        Class method to create an instance from a CIF dictionary.
    """

    def __init__(self, text: str = None):
        """Initialize RmatData with text from an RMAT file."""
        self.data = self.extract_data(text) if text else {}

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

        return extracted_data

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
        value = self.data[key]
        if isinstance(value, np.ndarray) and value.ndim == 2:
            if key == 'RMAT':
                key_string = f'RMAT {self.data["CENTRING"]}\n'
            elif key == 'TMAT':
                key_string = f'TMAT {self.data["CENTRING"]} {self.data["POINTGROUP"]}\n'
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
            self.value_as_string(key) for key in self.data
            if key not in ('CENTRING', 'POINTGROUP')
        ]
        return '\n'.join(output)

    @classmethod
    def from_rmat_file(cls, file_path: str) -> 'Rmat':
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

    def to_rmat_file(self, file_path: str):
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
            '_diffrn_orient_matrix.UB_11': self.data['RMAT'][0][0],
            '_diffrn_orient_matrix.UB_12': self.data['RMAT'][0][1],
            '_diffrn_orient_matrix.UB_13': self.data['RMAT'][0][2],
            '_diffrn_orient_matrix.UB_21': self.data['RMAT'][1][0],
            '_diffrn_orient_matrix.UB_22': self.data['RMAT'][1][1],
            '_diffrn_orient_matrix.UB_23': self.data['RMAT'][1][2],
            '_diffrn_orient_matrix.UB_31': self.data['RMAT'][2][0],
            '_diffrn_orient_matrix.UB_32': self.data['RMAT'][2][1],
            '_diffrn_orient_matrix.UB_33': self.data['RMAT'][2][2],
            '_space_group.centring_type': self.data['CENTRING']
        }
        if 'TMAT' in self.data:
            cif_dict.update({
                '_diffrn_reflns_transf_matrix.11': self.data['TMAT'][0][0],
                '_diffrn_reflns_transf_matrix.12': self.data['TMAT'][0][1],
                '_diffrn_reflns_transf_matrix.13': self.data['TMAT'][0][2],
                '_diffrn_reflns_transf_matrix.21': self.data['TMAT'][1][0],
                '_diffrn_reflns_transf_matrix.22': self.data['TMAT'][1][1],
                '_diffrn_reflns_transf_matrix.23': self.data['TMAT'][1][2],
                '_diffrn_reflns_transf_matrix.31': self.data['TMAT'][2][0],
                '_diffrn_reflns_transf_matrix.32': self.data['TMAT'][2][1],
                '_diffrn_reflns_transf_matrix.33': self.data['TMAT'][2][2]
            })

        if 'POINTGROUP' in self.data:
            cif_dict['_space_group.point_group_h-m'] = self.data['POINTGROUP']

        if 'CELL' in self.data:
            cif_dict.update({
                '_cell.length_a': self.data['CELL'][0],
                '_cell.length_b': self.data['CELL'][1],
                '_cell.length_c': self.data['CELL'][2],
                '_cell.angle_alpha': self.data['CELL'][3],
                '_cell.angle_beta': self.data['CELL'][4],
                '_cell.angle_gamma': self.data['CELL'][5],
                '_cell.volume': self.data['CELL'][6]
            })

        if 'SIGMACELL' in self.data:
            cif_dict.update({
                '_cell.length_a_su': self.data['SIGMACELL'][0],
                '_cell.length_b_su': self.data['SIGMACELL'][1],
                '_cell.length_c_su': self.data['SIGMACELL'][2],
                '_cell.angle_alpha_su': self.data['SIGMACELL'][3],
                '_cell.angle_beta_su': self.data['SIGMACELL'][4],
                '_cell.angle_gamma_su': self.data['SIGMACELL'][5],
                '_cell.volume_su': self.data['SIGMACELL'][6]
            })
        return cif_dict

    @classmethod
    def from_cif_dict(cls, cif_dict: dict) -> 'Rmat':
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
        new.data['RMAT'] = np.array(
            [
                [cif_dict[f'_diffrn_orient_matrix.UB_{i}{j}'] for j in range(1,4)]
                for i in range(1,4)
            ]
        )
        new.data['CENTRING'] = cif_dict['_space_group.centring_type']

        tmat_entries = [
            f'_diffrn_reflns_transf_matrix.{i}{j}' for i, j in product(range(1,4), repeat=2)
        ]
        tmat_entries.append('_space_group.point_group_h-m')
        if all(entry in cif_dict for entry in tmat_entries):
            new.data['TMAT'] = np.array(
                [
                    [cif_dict[f'_diffrn_reflns_transf_matrix.{i}{j}'] for j in range(1,4)]
                    for i in range(1,4)
                ]
            )
            new.data['POINTGROUP'] = cif_dict['_space_group.point_group_h-m']

        cell_entries = (
            '_cell.length_a', '_cell.length_b', '_cell.length_c',
            '_cell.angle_alpha', '_cell.angle_beta', '_cell.angle_gamma',
            '_cell.volume'
        )

        if all(entry in cif_dict for entry in cell_entries):
            new.data['CELL'] = np.array([cif_dict[entry] for entry in cell_entries])

        sigmacell_entries = tuple(entry + '_su' for entry in cell_entries)
        if all(entry in cif_dict for entry in sigmacell_entries):
            new.data['SIGMACELL'] = np.array([cif_dict[entry] for entry in sigmacell_entries])
        return new


class EvalAppRobot:
    def __init__(self, command_list, file_list):
        self.command_list = command_list
        self.file_list = file_list
