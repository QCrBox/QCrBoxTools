# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

import re
from typing import Union, List, Optional, Tuple, Dict
import os
from pathlib import Path
import subprocess
import textwrap
import warnings

from .eval_files import PicFile, TextFile, SettingsVicFile, RmatFile

class EvalBaseRobot:
    """
    Base class for automating execution of various programs in crystallography data processing.

    This class provides foundational functionalities for derived classes to automate
    the execution of crystallography-related programs with specified command sequences.
    Assures that a work_folder will always be a pathlib.Path even when set as string

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
    _work_folder = None

    def __init__(self, work_folder: Union[str, Path]):
        """
        Initializes the EvalBaseRobot with a specified work folder.

        Parameters
        ----------
        work_folder : Union[str, Path]
            The directory where the automated processes are to be executed.
        """
        self.work_folder = work_folder

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

        init_existed = init_file.exists()
        if init_existed:
            old_init_file = init_file.read_text(encoding='UTF-8')
        init_file.write_text('\n'.join(command_list) + '\n', encoding='UTF-8')
        log_file = self.work_folder / f'{program_name}_output.log'
        try:
            with open(log_file, 'w', encoding='UTF-8') as fobj:
                subprocess.call(program_name, cwd=self.work_folder, stdout=fobj, stderr=fobj)
        except OSError:
            with open(log_file, 'w', encoding='UTF-8') as fobj:
                subprocess.call(
                    program_name,
                    cwd=self.work_folder,
                    stdout=fobj,
                    stderr=fobj,
                    shell=True
                )

        if init_existed:
            init_file.write_text(old_init_file, encoding='UTF-8')
        else:
            init_file.unlink()

        @property
        def work_folder(self):
            return self._work_folder

        @work_folder.setter
        def work_folder(self, path):
            self._work_folder = Path(path)

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

        eval15_init_command_list = [f'@{file.filename}' for file in self.file_list]
        eval15_init_command_list.append('markdefault')
        eval15_init_path = self.work_folder / 'eval15.init'
        previous_file_existed = eval15_init_path.exists()
        if previous_file_existed:
            eval15content = eval15_init_path.read_text(encoding='UTF-8')
        eval15_init_path.write_text('\n'.join(eval15_init_command_list) + '\n', encoding='UTF-8')
        self._run_program_with_commands('eval15all', [''] * 10)
        eval15_init_path.unlink()
        if previous_file_existed:
            eval15_init_path.write_text(eval15content, encoding='UTF-8')


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
        # name, type, n_chars, format_str
        ('_diffrn_refln.index_h', int, 4, ' 4d'),
        ('_diffrn_refln.index_k', int, 4, ' 4d'),
        ('_diffrn_refln.index_l', int, 4, ' 4d'),
        ('_diffrn_refln.intensity_net', float, 8, ' 11.2f'),
        ('_diffrn_refln.intensity_net_su', float, 8, ' 9.2f'),
        ('_diffrn_refln.class_code', int, 4, ' 4d'),
        ('_qcrbox.diffrn_refln.direction_cosine_incid_x', float, 8, ' 9.5f'),
        ('_qcrbox.diffrn_refln.direction_cosine_incid_y', float, 8, ' 9.5f'),
        ('_qcrbox.diffrn_refln.direction_cosine_incid_z', float, 8, ' 9.5f'),
        ('_qcrbox.diffrn_refln.direction_cosine_diffrn_x', float, 8, ' 9.5f'),
        ('_qcrbox.diffrn_refln.direction_cosine_diffrn_y', float, 8, ' 9.5f'),
        ('_qcrbox.diffrn_refln.direction_cosine_diffrn_z', float, 8, ' 9.5f'),
        ('_qcrbox.diffrn_refln.detector_px_x_obs', float, 7, ' 8.2f'),
        ('_qcrbox.diffrn_refln.detector_px_y_obs', float, 7, ' 8.2f'),
        ('_qcrbox.diffrn_refln.detector_frame_obs', float, 8, ' 9.2f'),
        ('_qcrbox.diffrn_refln.evalsad_mystery_val1', float, 7, ' 8.2f'),
        ('_qcrbox.diffrn_refln.evalsad_mystery_val2', int, 5, ' 5d')
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

        column_names, types, widthes, _ = zip(*self._abs_columns)

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

        line_format_string = ' '.join(f'{{:{entry[3]}}}' for entry in self._abs_columns)

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

    def folder_to_cif(self, cif_filename) -> Dict[str, float]:
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

        file_lines = [
            r'#\#CIF_2.0', '', 'data_eval_output', ''
        ]
        file_lines += [
            f'{key} {str(val)}' for key, val in cif_dict.items()
        ]
        cif_path = self.work_folder / cif_filename

        cif_path.write_text('\n'.join(file_lines))

class EvalScandbRobot(EvalBaseRobot):
    """
    A class designed to automate the operation of the Eval's 'scandb' program.

    Attributes
    ----------
    work_folder : Path
        The working directory where the 'scandb' command will be executed.
    """

    def run(self) -> None:
        """
        Executes the 'scandb' program in the specified working folder.

        This method checks for the existence of a 'view.init' file in the working directory.
        If such a file exists, it is temporarily removed before running 'scandb' to avoid
        interference, and then restored after 'scandb' execution completes.
        """
        view_init_path = self.work_folder / 'view.init'
        view_init_existed = view_init_path.exists()

        # as scandb itself is a wrapper around view, a view.init might interfere
        if view_init_existed:
            view_init_content = view_init_path.read_text(encoding='UTF-8')
            view_init_path.unlink()

        self._run_program_with_commands('scandb', [])

        if view_init_existed:
            view_init_path.write_text(view_init_content, encoding='UTF-8')

class EvalBuilddatcolRobot(EvalBaseRobot):
    """
    A class for automating the creation of data collection files using the 'builddatcol'
    for subsequent processing with Eval's 'view'

    Attributes
    ----------
    work_folder : Path
        The working directory where the 'builddatcol' command will be or has been executed.
    """
    def create_datcol_files(
        self,
        rmat_file: RmatFile,
        minimum_res: float,
        maximum_res: float,
        box_size: float,
        box_depth: float,
        maximum_duration: int,
        min_refln_in_box: int
    ) -> None:
        """
        Creates the configuration file for 'builddatcol' and executes the command.

        Parameters
        ----------
        rmat_file : Union[RmatFile, str]
            The RmatFile object or path to an RMAT file used for data collection setup.
        minimum_res : float
            The minimum resolution for data processing in angstrom.
        maximum_res : float
            The maximum resolution for data processing in angstrom.
        box_size : float
            The size of the box for data collection in millimeters.
        box_depth : float
            The depth of the box for data collection in number of frames.
        maximum_duration : int
            The maximum duration see durationmax on:
            http://www.crystal.chem.uu.nl/distr/eval/documentation/ccd/view/doc/view.html
        min_refln_in_box : int
            The minimum number of reflections in a box.

        This method first ensures that 'scaninfo.txt' is present in the working directory,
        creating it if necessary. It then generates a 'datcolsetup.vic' file with the
        specified parameters and runs the 'builddatcol' command.
        """

        if maximum_res > minimum_res:
            raise ValueError(
                'The value of the maximum resolution will always'
                + 'be smaller than the minimum resolution.'
            )
        # builddatcol will not run without scaninfo.txt
        scandb_path = self.work_folder / 'scaninfo.txt'
        if not scandb_path.exists():
            scandb = EvalScandbRobot(self.work_folder)
            scandb.run()

        datcolsetup = textwrap.dedent(f"""\
            ! Created by builddatcol at 16-Nov-2023 13:50:14
            ! Host DW-PF3E3G40 User niklas WD /home/niklas/messing_around/eval/Ylid_OD_Images/
            ! abort on warnings
            abort on
            badpixel on
            rmat {rmat_file.filename}
            ! load scan specific rmat if it exists
            \if file 'scan'.rmat rmat 'scan'.rmat
            ! scandependent beamstop
            &beamstop'scan'.vic
            ! scandependent goniostat
            &goniostat'scan'.vic
            resomin {minimum_res}
            resomax {maximum_res}
            boxsizemm {box_size}
            boxdepth {box_depth}
            durationmax {maximum_duration}
            boxrefl {min_refln_in_box}
            ! display only
            datcolboxes 1
            output none
        """)

        datcolsetup_file = self.work_folder / 'datcolsetup.vic'
        datcolsetup_file.write_text(datcolsetup)

        rmat_file.to_file(self.work_folder)

        self._run_program_with_commands('builddatcol', [''] * 10)

    def extract_vars(self) -> Dict[str, Union[float, int]]:
        """
        Extracts parameters from the 'datcolsetup.vic' file in the working directory.

        Parses the configuration file used by 'builddatcol' to retrieve parameters
        related to the data collection setup, such as resolution limits, box size,
        box depth, maximum duration, and minimum reflections in a box.

        Returns
        -------
        Dict[str, Union[float, int]]
            A dictionary containing the extracted parameters with their values.
            The keys correspond to parameter names such as 'minimum_res', 'maximum_res',
            'box_size', 'box_depth', 'maximum_duration', and 'min_refln_in_box',
            with their values cast to the appropriate types (float or int).

        Raises
        ------
        KeyError
            If any expected parameter is not found in the 'datcolsetup.vic' file,
            indicating a potential issue with file content or format.
        """

        datcolsetup = self.work_folder / 'datcolsetup.vic'
        content = datcolsetup.read_text()
        searches = (
            ('minimum_res', 'resomin', float),
            ('maximum_res', 'resomax', float),
            ('box_size', 'boxsizemm', float),
            ('box_depth', 'boxdepth', int),
            ('maximum_duration', 'durationmax', float),
            ('min_refln_in_box', 'boxrefl', float)
        )
        results = {}
        number_pattern = r'(\d+\.?\d*)'
        for name, internal_name, output_type in searches:
            search_pattern = rf'{internal_name}\s+{number_pattern}'
            search = re.search(search_pattern, content)
            if search is None:
                raise KeyError(f'Could not find {name}/{internal_name} in datcolsetup.vic')
            results[name] = output_type(search.group(1))

        return results


class EvalBuildeval15Robot(EvalBaseRobot):
    """
    A class for automating the configuration of the 'buildeval15' command.

    Attributes
    ----------
    p4p_file : Optional[str]
        The path to a '.p4p' file, if any, currently only skips entering a crystal
        size if not None.

    Methods
    -------
    __init__(self, work_folder: Union[str, Path], p4p_file: Optional[str] = None)
        Initializes EvalBuildeval15Robot with a work folder and an optional '.p4p' file.
    run(
        self,
        focus_type: Optional[str] = None,
        polarisation_type: Optional[str] = None,
        pointspread_gamma: Optional[float] = None,
        acdnoise: Optional[float] = None,
        crystal_dimension: Optional[Tuple[float, float, float]] = None,
        mosaic: Optional[float] = None
    )
        Executes the 'buildeval15' command with specified configuration parameters.
    """
    def __init__(
        self,
        work_folder: Union[str, Path],
        p4p_file: Optional[str] = None
    ):
        """
        Initializes the EvalBuildeval15Robot with a specified work folder and
        an optional '.p4p' file.

        Parameters
        ----------
        work_folder : Union[str, Path]
            The directory where the 'buildeval15' command configuration will be executed.
        p4p_file : Optional[str], default None
            The path to a '.p4p' file, if any, currently only skips entering a crystal
            size if not None.
        """
        super().__init__(work_folder)

        self.p4p_file = p4p_file


    def run(
        self,
        focus_type: Optional[str] = None,
        polarisation_type: Optional[str] = None,
        pointspread_gamma: Optional[float] = None,
        acdnoise: Optional[float] = None,
        crystal_dimension: Optional[Tuple[float, float, float]] = None,
        mosaic: Optional[float] = None
    ):
        """
        Executes the 'buildeval15' command with specified configuration parameters. If values
        are not specified the buildeval15 defaults are used. Will create the necessary .pic
        files to run eval15all.

        Parameters
        ----------
        focus_type : Optional[str], default None
            The type of focus, e.g., 'tube', 'mirror'. Must be one of the predefined focus types.
            if None will use 'synchrotron'
        polarisation_type : Optional[str], default None
            The type of polarisation. Must be one of the predefined polarisation types.
            Only tested with 'none' (the default) so far.
        pointspread_gamma : Optional[float], default None
            The gamma value for the point spread function, default is 0.8.
        acdnoise : Optional[float], default None
            The noise level for the automatic crystal detection, default is 2.0.
        crystal_dimension : Optional[Tuple[float, float, float]], default None
            The dimensions of the crystal in millimeters for a cube-shaped crystal
            eval's default value is 0.2
        mosaic : Optional[float], default None
            The mosaic spread in degrees. Eval's default is 0.3.

        Raises
        ------
        ValueError
            If an invalid focus type or polarisation type is specified.
        """
        possible_focusses = (
            'unknown', 'tube', 'rotating', 'mirror', 'synchrotron', 'file'
        )
        if focus_type not in possible_focusses:
            raise ValueError(
                f'Invalid focus type, choose one of: {", ".join(possible_focusses)}'
            )

        if polarisation_type is None:
            polarisation_type='none'

        possible_polarisations = (
            'perpendicular', 'parallel', 'antiparallel', 'none', 'synchrotron',
            'synchrotronz', 'osmic', 'pe', 'pa', 'ap', 'n', 's', 'sz', 'o'
        )
        if polarisation_type not in possible_polarisations:
            raise ValueError(
                f'Invalid polarisation, choose one of: {", ".join(possible_polarisations)}'
            )

        if self.p4p_file is None:
            command_base = (
                focus_type, polarisation_type, pointspread_gamma,
                acdnoise, crystal_dimension, mosaic
            )
        else:
            command_base = (
                focus_type, polarisation_type, pointspread_gamma, acdnoise, mosaic
            )
            #self.p4p_file.to_file(self.work_folder)
            warnings.warn(
                'Reading and writing p4p files is currently not implemented.'
                + 'You need to add the p4p file yourself'
                )
        command_list = ['' if val is None else str(val) for val in command_base]
        self._run_program_with_commands('buildeval15', command_list)
