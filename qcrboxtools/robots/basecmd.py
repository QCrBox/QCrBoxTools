import subprocess
from pathlib import Path
from typing import List
import time
class CmdAppRobot:
    _popen = None
    def __init__(self, call_args:List[str], cwd: Path = '.', env=None):
        if env is None:
            env = {}
        self._popen = subprocess.Popen(
            call_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            #text=True,
            universal_newlines=True,
            env=env,
            cwd=cwd
        )

    def __del__(self):
        if self._popen is not None and self._popen.poll() is None:
            self._popen.kill()

    def _send_input(self, input_str:str):
        return self._popen.communicate(input_str + '\n')


