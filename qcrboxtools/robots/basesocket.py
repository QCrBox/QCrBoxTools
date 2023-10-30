import socket

class SocketRobot:
    server = None
    port = None

    def __init__(self, server: str, port: int):
        self.server = server
        self.port = port

    def _send_input(self, input_str: str):
        input_str += '\n'
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.server, self.port))
            s.sendall(input_str.encode('UTF-8'))
            # TODO receive all the data until delimiter
            data = s.recv(1024)
        return data.decode()