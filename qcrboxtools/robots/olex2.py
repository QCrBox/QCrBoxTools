"""
This module provides functionality for interacting with the Olex2 refinement program
in server mode via sockets.
"""

import os
import pathlib
import time
from itertools import count
from typing import Any
import warnings

from .basesocket import SocketRobot

class Olex2Socket(SocketRobot):
    """
    A specialized socket client for Olex2, which provides functionalities like
    loading and refining a structure, as well as sending general commands to the
    Olex2 server.

    Attributes:
    - structure_path (pathlib.Path): The path to the structure file.
    """
    _structure_path = None
    _tsc_path = None
    _task_id_counter = count()

    def __init__(
        self,
        olex_server: str = 'localhost',
        port: int = 8899,
        structure_path: str = None
    ):
        """
        Initializes the Olex2Socket with server details and an optional structure path.

        Args:
        - olex_server (str): The Olex2 server's address. Defaults to the
          environment variable $OLEX2SERVER or if that is unavailable: 'localhost'.
        - port (int): The port number on which the Olex2 server is listening. Defaults to
          the environment variable $OLEX2PORT or if that is unavailable: 8899.
        - structure_path (str): The path to the structure file. If provided, will be set
          for the instance, otherwise needs to be set later for refinement.
        """
        if olex_server == 'localhost' and 'OLEX2SERVER' in os.environ:
            olex_server = os.environ['OLEX2SERVER']
        if port == 8899 and 'OLEX2PORT' in os.environ:
            port = os.environ['OLEX2PORT']

        super().__init__(olex_server, port)
        if structure_path is not None:
            self.structure_path = structure_path

    @property
    def structure_path(self):
        """Returns the path of the structure file."""
        return self._structure_path

    @structure_path.setter
    def structure_path(self, path: str):
        """
        Sets the path of the structure file loads the structure with the path into Olex2.

        Args:
        - path (str): The path to the structure file.
        """
        self._structure_path = pathlib.Path(path)
        startup_commands = [
            f'user {self._structure_path.parents[0]}',
            f'reap {self._structure_path}'
        ]

        cmd_list = '\n'.join(startup_commands)
        cmd = f'run:startup\n{cmd_list}'
        self._send_input(cmd)

        self.wait_for_completion(2000, 'startup', cmd)

        load_cmds = [
            f'file {self._structure_path.parents[0] / "olex2socket.ins"}',
            f'export {self._structure_path.parents[0] / "olex2socket.hkl"}',
            f'reap {self._structure_path.parents[0] / "olex2socket.ins"}'
        ]
        self.send_command('\n'.join(load_cmds))

    @property
    def tsc_path(self):
        """Returns the path of the currently selected tsc(b) file."""
        return self._tsc_path

    @tsc_path.setter
    def tsc_path(self, path):
        """
        Sets the path of the tsc(b), if not None, activate aspherical refinement.

        Args:
        - path (str): The path to the tsc(b) file.
        """
        if path is not None:
            path = pathlib.Path(path)
            cmds = [
                "spy.SetParam('snum.NoSpherA2.use_aspherical', True)",
                "spy.SetParam('snum.NoSpherA2.Calculate', False)",
                "spy.SetParam('snum.NoSpherA2.source', 'tsc_file')",
                f"spy.SetParam('snum.NoSpherA2.file', {path})"
            ]
            self.send_command('\n'.join(cmds))
        else:
            cmds = [
                "spy.SetParam('snum.NoSpherA2.use_aspherical', False)",
                "spy.SetParam('snum.NoSpherA2.Calculate', False)",
                "spy.SetParam('snum.NoSpherA2.file', None)"
            ]
            self.send_command('\n'.join(cmds))
        self._tsc_path = path

    def check_connection(self):
        """
        Checks the connection status with the Olex2 server.

        Returns:
        - bool: True if the server is ready, False otherwise.
        """
        answer = self._send_input('status')
        return answer.strip() == 'ready'

    def send_command(self, input_str: str) -> str:
        """
        Sends a command string to the Olex2 server and waits for its completion.

        Args:
        - input_str (str): The command string to send.

        Returns:
        - str: The output log of the command process, or None if the log file is not found.
        """

        task_id = next(self._task_id_counter)
        _ = self._send_input(f'run:{task_id}\nlog:task_{task_id}.log\n{input_str}')
        timeout_counter = 10000
        self.wait_for_completion(timeout_counter, task_id, input_str)

        log_path = self.structure_path.parents[0] / f'task_{task_id}.log'
        try:
            with open(log_path, 'r', encoding='UTF-8') as fo:
                output = fo.read()
            return output
        except FileNotFoundError:
            return None

    def wait_for_completion(self, timeout_counter: int, task_id: Any, input_str: str):
        """
        Waits for a specific task to complete on the Olex2 server.

        This method will repeatedly check the status of a task on the Olex2 server every 100 ms
        until it is finished. It provides timeout functionality to prevent infinite waiting
        and will also raise an error if the task fails on the server.

        Args:
        - timeout_counter (int): The maximum number of times to check the status before timing out.
        - task_id (Any): The unique identifier of the task to check.
        - input_str (str): The original command sent to the server. This is used in the error
          message if the task fails.

        Raises:
        - RuntimeError: If the task fails during execution on the Olex2 server.
        """
        return_msg = ' '
        while 'finished' not in return_msg:
            return_msg = self._send_input(f'status:{task_id}')
            time.sleep(0.1)
            timeout_counter -= 1
            if timeout_counter < 0:
                warnings.warn('TimeOut limit for job reached. Continuing')
                break
            if 'failed' in return_msg:
                raise RuntimeError(
                    f'The command {input_str} raised an error during running in olex.'
                )

    def refine(self, n_cycles=20, refine_starts=5):
        """
        Refines a loaded structure and writes a cif file with the refined structure.

        Returns:
        - str: The output log of the refinement process.
        """
        if self.structure_path is None:
            raise ValueError('No structure loaded to refine. The structure_path attribute needs to be set before refinement')
        cmds = [
            'DelIns ACTA',
            'AddIns ACTA'
        ] + [f'refine {n_cycles}'] * refine_starts
        return self.send_command('\n'.join(cmds))

    def _shutdown_server(self):
        """Sends a 'stop' command to shut down the Olex2 server."""
        self._send_input('stop')
