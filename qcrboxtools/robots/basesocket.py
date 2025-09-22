# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

"""Base classes for program interactions via socket"""

import socket


class SocketRobot:
    """
    Represents a simple socket robot that can send input to a server and can be
    extended to specialise the interaction to the specific program

    Attributes:
    - server (str): The server's address.
    - port (int): The port number on which the server is listening.

    Methods:
    - _send_input(input_str: str) -> str: Sends a string input to the server and
        returns the server's response.
    """

    server = None
    port = None

    def __init__(self, server: str, port: int):
        """
        Initializes a new instance of the SocketRobot class.

        Args:
        - server (str): The server's address.
        - port (int): The port number on which the server is listening.
        """
        self.server = server
        self.port = port

    def _send_input(self, input_str: str) -> str:
        """
        Sends a string input to the server and returns the server's response.

        Args:
        - input_str (str): The input string to send to the server.

        Returns:
        - str: The server's response.
        """
        input_str += "\n"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.server, self.port))
                s.sendall(input_str.encode("UTF-8"))
                data = s.recv(1024)
            except (ConnectionRefusedError, OSError) as ex:
                raise ConnectionError(f"Could not connect to Socket server, {self.server}:{self.port}") from ex

        return data.decode("UTF-8").rstrip("\n")
