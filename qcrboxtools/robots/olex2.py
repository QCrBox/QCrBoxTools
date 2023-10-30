import os
import pathlib
import time
from itertools import count
import warnings

from .basesocket import SocketRobot

class Olex2Socket(SocketRobot):
    _structure_path = None
    task_id_counter = count(1)

    def __init__(self, olex_server='localhost', port=8899, structure_path=None):
        if olex_server == 'localhost' and 'OLEX2SERVER' in os.environ:
            olex_server = os.environ['OLEX2SERVER']
        if port == 8899 and 'OLEX2PORT' in os.environ:
            port = os.environ['OLEX2PORT']

        super().__init__(olex_server, port)
        if structure_path is not None:
            self.structure_path = structure_path

    @property
    def structure_path(self):
        return self._structure_path

    @structure_path.setter
    def structure_path(self, path):
        self._structure_path = pathlib.Path(path)
        cmd = f'reap {path}'
        output = self.send_command(cmd)


    def check_connection(self):
        answer = self._send_input('status')
        return answer.strip() == 'ready'

    def refine(self):
        cmds = [
            'export',
            'refine',
            'DelIns ACTA',
            'AddIns ACTA',
            'refine'
        ]
        return self.send_command('\n'.join(cmds))

    def send_command(self, input_str: str):
        task_id = next(self.task_id_counter)
        return_val = self._send_input(f'run:{task_id}\nlog:task_{task_id}.log\n{input_str}')
        timeout_counter = 10000
        while 'finished' not in self._send_input(f'status:{task_id}'):
            time.sleep(0.1)
            timeout_counter -= 1
            if timeout_counter < 0:
                warnings.warn('TimeOut limit for job reached. Continuing')
                break

        log_path = self.structure_path.parents[0] / f'task_{task_id}.log'
        try:
            with open(log_path, 'r', encoding='UTF-8') as fo:
                output = fo.read()

            return output
        except FileNotFoundError:
            return None

    def _shutdown_server(self):
        self._send_input('stop')
