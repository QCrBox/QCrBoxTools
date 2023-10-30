import platform

from qcrboxtools.robots.basecmd import CmdAppRobot


def test_cmd_with_python():
    if platform.system() == 'Windows':
        executable = 'python.exe'
    else:
        executable = 'python'
    cmd_rob = CmdAppRobot(call_args=[executable])
    stdout, _ = cmd_rob._send_input('print("mytest123")')
    assert stdout == 'mytest123\n'



