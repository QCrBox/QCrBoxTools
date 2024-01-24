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


from .eval_files import *
from .eval_robots import *