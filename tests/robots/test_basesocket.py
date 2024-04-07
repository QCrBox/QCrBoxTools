# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

import socket
from unittest.mock import patch

from qcrboxtools.robots.basesocket import SocketRobot


def test_socket_robot_init():
    server = "test_server"
    port = 1234
    socket_robot = SocketRobot(server, port)
    assert socket_robot.server == server
    assert socket_robot.port == port


def test_socket_robot_send_input():
    server = "test_server"
    port = 1234
    socket_robot = SocketRobot(server, port)
    with patch("socket.socket") as mock_socket:
        mock_socket.return_value.__enter__.return_value.recv.return_value = b"test_response"
        response = socket_robot._send_input("test_input")
        assert response == "test_response"
        mock_socket.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
        mock_socket.return_value.__enter__.return_value.connect.assert_called_once_with((server, port))
        mock_socket.return_value.__enter__.return_value.sendall.assert_called_once_with(b"test_input\n")
        mock_socket.return_value.__enter__.return_value.recv.assert_called_once_with(1024)
