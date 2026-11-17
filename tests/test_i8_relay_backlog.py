import os
import socket

import pytest

from scripts.study_operator.network_relay import server


def test_relays_listen_with_a_deep_backlog():
    # F-020: the socketserver default of 5 made the kernel refuse (EAGAIN) the extra Unix connects that
    # Caddy opens for a burst of concurrent requests, so lazy Streamlit chunks came back as 502.
    with server(('127.0.0.1', 0), ('127.0.0.1', 1)) as relay:
        assert relay.request_queue_size >= 128


@pytest.mark.skipif(os.name != 'posix', reason='POSIX Unix sockets')
def test_unix_relay_absorbs_a_cold_burst_without_eagain(tmp_path):
    path = str(tmp_path / 'streamlit.sock')
    clients = []
    # The relay is deliberately not serving: only the listen queue is measured.
    with server(path, ('127.0.0.1', 1)):
        try:
            for _ in range(40):
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                clients.append(client)
                client.setblocking(False)
                client.connect(path)  # raises BlockingIOError (EAGAIN) once the queue is full
        finally:
            for client in clients:
                client.close()