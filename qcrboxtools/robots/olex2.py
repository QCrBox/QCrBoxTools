"""
This module provides functionality for interacting with the Olex2 refinement program
in server mode via sockets.
"""

import os
import pathlib
import time
from itertools import count
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
        return_value = self._send_input(f'run:startup\nuser {self._structure_path.parents[0]}')
        time.sleep(0.5)
        return_value2 = self._send_input(f'run:startup\nreap {self._structure_path}')
        time.sleep(0.5)

        load_cmds = [
            f'file {self._structure_path.parents[0] / "olex2socket.ins"}',
            f'export {self._structure_path.parents[0] / "olex2socket.hkl"}',
            f'reap {self._structure_path.parents[0] / "olex2socket.ins"}'
        ]
        out = self.send_command('\n'.join(load_cmds))


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
        return_msg = ' '
        while 'finished' not in return_msg:
            return_msg = self._send_input(f'status:{task_id}')
            time.sleep(0.1)
            timeout_counter -= 1
            if timeout_counter < 0:
                warnings.warn('TimeOut limit for job reached. Continuing')
                break
            if 'failed' in return_msg:
                raise RuntimeError(f'The command {input_str} raised an error during running in olex.')

        log_path = self.structure_path.parents[0] / f'task_{task_id}.log'
        try:
            with open(log_path, 'r', encoding='UTF-8') as fo:
                output = fo.read()

            return output
        except FileNotFoundError:
            return None

    def refine(self, n_cycles=20, refine_starts=5):
        """
        Refines a loaded structure and writes a cif file with the refined structure.

        Returns:
        - str: The output log of the refinement process.
        """
        cmds = [
            'DelIns ACTA',
            'AddIns ACTA'
        ] + [f'refine {n_cycles}'] * refine_starts
        return self.send_command('\n'.join(cmds))

    def _shutdown_server(self):
        """Sends a 'stop' command to shut down the Olex2 server."""
        self._send_input('stop')
