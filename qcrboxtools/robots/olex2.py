# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

"""
Socket-based interface for Olex2 crystallographic refinement program.

This module provides the `Olex2Socket` class for interacting with the Olex2
crystallographic refinement program running in server mode. It enables automated
structure refinement, including support for aspherical atom form factors via
NoSpherA2 integration.

The module extends the base socket functionality to provide Olex2-specific
command generation, job management, and refinement workflows.

Classes
-------
Olex2Socket
    Socket client for automated Olex2 refinement operations.

Examples
--------
>>> from qcrboxtools.robots.olex2 import Olex2Socket
>>> from pathlib import Path
>>>
>>> # Connect to Olex2 server
>>> olex = Olex2Socket("localhost", 8899)
>>>
>>> # Run refinement
>>> cif_file = Path("structure.cif")
>>> result = olex.run_full_refinement(cif_file, n_cycles=20)
"""

import os
import shutil
import time
import warnings
from itertools import count
from pathlib import Path

from .basesocket import SocketRobot


class Olex2Socket(SocketRobot):
    """
    Socket client for automated Olex2 crystallographic refinement operations.

    This class provides a high-level interface for interacting with the Olex2
    crystallographic refinement program running in server mode. It supports
    structure loading, refinement execution, and integration with aspherical
    atom form factors via NoSpherA2.

    The class manages job queuing, command generation, and result retrieval
    to enable automated refinement workflows.

    Parameters
    ----------
    olex_server : str, optional
        The hostname or IP address of the Olex2 server. Defaults to "localhost"
        or the value of the OLEX2SERVER environment variable if set.
    port : int, optional
        The port number on which the Olex2 server is listening. Defaults to 8899
        or the value of the OLEX2PORT environment variable if set.

    Attributes
    ----------
    _task_id_counter : itertools.count
        Counter for generating unique task identifiers.

    Methods
    -------
    run_full_refinement(cif_path, tsc_path=None, n_cycles=20, refine_starts=5)
        Execute a complete refinement workflow.
    send_command(cif_path, tsc_path, input_str)
        Send arbitrary commands to the Olex2 server.
    check_connection()
        Verify the connection to the Olex2 server.
    wait_for_completion(timeout_counter, input_str)
        Wait for a submitted job to complete.

    Examples
    --------
    >>> olex = Olex2Socket("localhost", 8899)
    >>> result = olex.run_full_refinement(Path("structure.cif"))
    >>> print("Refinement completed")
    """

    def __init__(
        self,
        olex_server: str = "localhost",
        port: int = 8899,
    ):
        """
        Initialize the Olex2Socket client.

        Sets up connection parameters and initializes the task counter for job
        management. Environment variables OLEX2SERVER and OLEX2PORT are used
        if available and default parameters are not explicitly provided.

        Parameters
        ----------
        olex_server : str, optional
            The hostname or IP address of the Olex2 server. Defaults to "localhost"
            or the value of the OLEX2SERVER environment variable if set.
        port : int, optional
            The port number on which the Olex2 server is listening. Defaults to 8899
            or the value of the OLEX2PORT environment variable if set.

        Notes
        -----
        The connection is not established during initialization. Use the inherited
        socket methods to connect when needed.
        """
        if olex_server == "localhost" and "OLEX2SERVER" in os.environ:
            olex_server = os.environ["OLEX2SERVER"]
        if port == 8899 and "OLEX2PORT" in os.environ:
            try:
                port = int(os.environ["OLEX2PORT"])
            except ValueError:
                warnings.warn(
                    f"OLEX2PORT environment variable ('{os.environ['OLEX2PORT']}')"
                    + " is not a valid integer. Using default port 8899."
                )
                port = 8899

        super().__init__(olex_server, port)
        self._task_id_counter = count()

    def move_refined_files(self, cif_path: Path):
        """
        Move existing .ins and .hkl files to prevent conflicts during refinement.

        Before starting refinement, this method backs up any existing .ins and .hkl
        files by copying them with a "_moved" suffix and removing the originals.
        This prevents conflicts when Olex2 creates new files during refinement.

        Parameters
        ----------
        cif_path : Path
            The path to the CIF file. The .ins and .hkl files are expected to
            have the same base name but with their respective extensions.
        """
        if cif_path.absolute().with_suffix(".ins").exists():
            shutil.copy(
                cif_path.absolute().with_suffix(".ins"),
                str(cif_path.absolute().with_suffix("")) + "_moved.ins",
            )
            cif_path.absolute().with_suffix(".ins").unlink()

        if cif_path.absolute().with_suffix(".hkl").exists():
            shutil.copy(
                cif_path.absolute().with_suffix(".hkl"),
                str(cif_path.absolute().with_suffix("")) + "_moved.hkl",
            )
            cif_path.absolute().with_suffix(".hkl").unlink()

    def create_structure_load_cmds(self, cif_path: Path) -> list[str]:
        """
        Generate Olex2 commands for loading a crystallographic structure.

        Creates a sequence of Olex2 commands to load a CIF file, set up the
        working directory, and prepare associated files (.ins and .hkl) for
        refinement operations.

        Parameters
        ----------
        cif_path : Path
            The path to the CIF file to be loaded.

        Returns
        -------
        list[str]
            A list of Olex2 commands for structure loading and setup.
        """
        return [
            f"if strcmp('{cif_path}','none') then none else spy.saveHistory()",
            "spy.SaveStructureParams()",
            f"if strcmp('{cif_path}','none') then none else 'spy.SaveCifInfo()'",
            "if IsFileLoaded() then clear",
            f"user {str(cif_path.parent)}",
            f"@reap {cif_path}",
            "spy.OnStructureLoaded(filename())",
            f"file {cif_path.with_suffix('.ins').name}",
            f"@export {cif_path.with_suffix('.hkl').name}",
            f"@reap {cif_path.with_suffix('.ins')}",
        ]

    def create_startup_cmds(self, cif_path: Path, job_id: int) -> list[str]:
        """
        Generate Olex2 startup commands for job initialization.

        Creates commands to set up the Olex2 environment for a specific job,
        including logging, working directory, job identification, and client
        mode configuration.

        Parameters
        ----------
        cif_path : Path
            The path to the CIF file being processed.
        job_id : int
            Unique identifier for the current job.

        Returns
        -------
        list[str]
            A list of Olex2 startup commands.
        """
        return [
            f"run:task_{job_id}",
            f"xlog:{str(cif_path.parent / f'task_{job_id}.log')}",
            f"user {str(cif_path.parent)}",
            "SetVar server.job_id " + f"task_{job_id}",
            "SetVar olex2.remote_mode true",
            "spy.LoadParams 'user,olex2'",
            "spy.SetParam user.refinement.client_mode False",
            "SetOlex2RefinementListener(True)",
        ]

    def create_nosphera2_cmds(self, tsc_path: Path | None) -> list[str]:
        """
        Generate NoSpherA2 configuration commands for aspherical refinement.

        Creates Olex2 commands to configure NoSpherA2 settings for either
        aspherical or spherical atom form factors based on whether a TSC
        file is provided.

        Parameters
        ----------
        tsc_path : Path or None
            The path to the TSC (or TSCB) file containing aspherical form factors.
            If None, aspherical refinement is disabled.

        Returns
        -------
        list[str]
            A list of Olex2 commands for NoSpherA2 configuration.
        """
        if tsc_path is None:
            return [
                "spy.SetParam('snum.NoSpherA2.use_aspherical', False)",
                "spy.SetParam('snum.NoSpherA2.Calculate', False)",
                "spy.SetParam('snum.NoSpherA2.file', None)",
            ]
        return [
            "spy.SetParam('snum.NoSpherA2.use_aspherical', True)",
            "spy.SetParam('snum.NoSpherA2.Calculate', False)",
            "spy.SetParam('snum.NoSpherA2.source', 'tsc_file')",
            f"spy.SetParam('snum.NoSpherA2.file', '{tsc_path.name}')",
        ]

    def create_refine_cmds(self, n_cycles: int, refine_starts: int) -> list[str]:
        """
        Generate refinement commands for crystallographic optimization.

        Creates a sequence of Olex2 commands to set up the refinement program,
        configure instruction file entries, and execute refinement cycles.

        Parameters
        ----------
        n_cycles : int
            The number of least-squares cycles to perform in each refinement run.
        refine_starts : int
            The number of times to restart the refinement process to adapt a
            weighting scheme iteratively.

        Returns
        -------
        list[str]
            A list of Olex2 refinement commands.
        """
        return [
            "spy.set_refinement_program(olex2.refine, Gauss-Newton)",
            "DelIns ACTA",
            "AddIns ACTA",
        ] + [f"refine {n_cycles}"] * refine_starts

    def reserve_new_job(self, task_id):
        """
        Reserve a job slot on the Olex2 server for execution.

        Attempts to reserve a job slot with the given task ID, waiting if the
        server is currently busy with other jobs.

        Parameters
        ----------
        task_id : str
            The unique identifier for the job to be reserved.

        Notes
        -----
        Polls the server every second until a slot becomes available.
        """
        data = self._send_input(f"reserve:{task_id}")
        while data == "busy":
            time.sleep(1)
            data = self._send_input(f"reserve:{task_id}")

    def run_full_refinement(
        self, cif_path: Path, tsc_path: Path | None = None, n_cycles: int = 20, refine_starts: int = 5
    ) -> str:
        """
        Execute a complete crystallographic refinement workflow.

        Performs a full refinement process including structure loading, optional
        aspherical atom form factor configuration via NoSpherA2, and iterative
        least-squares refinement.

        Parameters
        ----------
        cif_path : Path
            The path to the CIF file containing the crystal structure to be refined.
        tsc_path : Path or None, optional
            The path to the TSC (or TSCB) file for aspherical refinement using
            NoSpherA2. If None, standard spherical atom form factors are used.
            Default is None.
        n_cycles : int, optional
            The number of least-squares refinement cycles to perform in each
            refinement iteration. Default is 20.
        refine_starts : int, optional
            The number of times to restart the refinement process to improve
            convergence. Default is 5.

        Returns
        -------
        str
            The complete log output from the refinement process, containing
            detailed information about the refinement progress and results.

        Notes
        -----
        Uses helper methods to generate command sequences and delegates to
        `send_command` for execution.
        """
        cmd_string = "\n".join(self.create_refine_cmds(n_cycles, refine_starts))
        return self.send_command(cif_path, tsc_path, cmd_string)

    def check_connection(self):
        """
        Verify the connection status with the Olex2 server.

        Sends a status query to the Olex2 server to determine if it is ready
        to accept new jobs and commands.

        Returns
        -------
        bool
            True if the server responds with "ready" status, False otherwise.
        """
        answer = self._send_input("status")
        return answer.strip() == "ready"

    def send_command(self, cif_path: Path, tsc_path: Path | None, input_str: str) -> str:
        """
        Send arbitrary commands to the Olex2 server for execution.

        This method provides a general interface for sending custom command
        strings to Olex2. It handles job setup, command execution, and result
        retrieval, making it suitable for both standard and custom workflows.

        Parameters
        ----------
        cif_path : Path
            The path to the CIF file to be used in the command execution.
        tsc_path : Path or None
            The path to the TSC (or TSCB) file for aspherical refinement.
            If None, aspherical refinement is disabled for this command.
        input_str : str
            The command string to be executed on the Olex2 server. This can
            contain multiple Olex2 commands separated by newlines.

        Returns
        -------
        str
            The complete log output from the command execution, or "No log file found."
            if the log file could not be accessed.

        Notes
        -----
        This method:
        1. Backs up existing refinement files
        2. Sets up a new job with unique ID
        3. Configures the Olex2 environment
        4. Loads the structure and sets up NoSpherA2
        5. Executes the provided commands
        6. Waits for completion and retrieves the log

        The method is used internally by `run_full_refinement` but can also
        be called directly for custom command sequences.
        """

        cif_path_abs = cif_path.absolute()
        self.move_refined_files(cif_path_abs)

        job_id = next(self._task_id_counter)
        task_id = f"task_{job_id}"
        cmds = (
            self.create_startup_cmds(cif_path_abs, job_id)
            + self.create_structure_load_cmds(cif_path_abs)
            + self.create_nosphera2_cmds(tsc_path)
            + [input_str, "@close"]
        )

        self.reserve_new_job(task_id)

        full_cmd = "\n".join(cmds)
        _ = self._send_input(full_cmd + "\n")
        timeout_counter = 2000
        self.wait_for_completion(timeout_counter, full_cmd)
        try:
            with open(cif_path_abs.parent / f"task_{job_id}.log", "r", encoding="UTF-8") as fo:
                output = fo.read()
            return output
        except FileNotFoundError:
            return "No log file found."

    def wait_for_completion(self, timeout_counter: int, input_str: str):
        """
        Wait for a submitted job to complete on the Olex2 server.

        Continuously polls the Olex2 server status until the job completes,
        with timeout protection and error handling for failed jobs.

        Parameters
        ----------
        timeout_counter : int
            The maximum number of status checks before timing out. Each check
            occurs approximately every 500ms.
        input_str : str
            The original command string sent to the server. Used in error
            messages if the job fails during execution.

        Raises
        ------
        RuntimeError
            If the server returns an error or failure status during job execution.

        Warns
        -----
        UserWarning
            If the timeout limit is reached before job completion.

        Notes
        -----
        Returns immediately when server reports "ready" status or raises
        an exception if "error" or "failed" appears in the response.
        """
        return_msg = " "
        while "ready" not in return_msg:
            return_msg = self._send_input("status\n")
            time.sleep(0.5)
            timeout_counter -= 1
            if timeout_counter < 0:
                warnings.warn("TimeOut limit for job reached. Continuing")
                break
            if "error" in return_msg or "failed" in return_msg:
                raise RuntimeError(f"The command {input_str} raised an error during running in olex.")

    def _shutdown_server(self):
        """
        Shut down the Olex2 server.

        Sends a 'stop' command to the Olex2 server to initiate a clean shutdown.
        This is typically used when the server needs to be terminated after
        completing all refinement tasks.

        Notes
        -----
        This method sends the shutdown command but does not wait for confirmation.
        The server should stop accepting new jobs and terminate after completing
        any currently running tasks.
        """
        self._send_input("stop")
