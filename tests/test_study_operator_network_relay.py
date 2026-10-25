import socket
import socketserver
import threading
import os
import stat
from types import SimpleNamespace

import pytest

from scripts.study_operator.network_relay import server, unlink_owned_socket


@pytest.mark.parametrize('mode,inode,uid,removed', [
    (stat.S_IFSOCK,7,10001,True), (stat.S_IFSOCK,8,10001,False),
    (stat.S_IFREG,7,10001,False), (stat.S_IFLNK,7,10001,False), (stat.S_IFSOCK,7,0,False)])
def test_socket_cleanup_preserves_replacement_files_and_other_owners(mode,inode,uid,removed):
    current=SimpleNamespace(st_mode=mode,st_dev=2,st_ino=inode,st_uid=uid)
    deleted=[]
    unlink_owned_socket('/owned/sock',(2,7,10001),lstat=lambda path:current,unlink=deleted.append)
    assert deleted == (['/owned/sock'] if removed else [])


@pytest.mark.skipif(os.name != 'posix', reason='POSIX filesystem Unix socket lifecycle')
def test_closed_unix_relay_can_rebind_after_freeze(tmp_path):
    path=str(tmp_path/'streamlit.sock')
    with server(path,('127.0.0.1',8501)):
        assert os.path.exists(path)
    assert not os.path.exists(path)
    with server(path,('127.0.0.1',8501)):
        assert os.path.exists(path)
    assert not os.path.exists(path)


def test_byte_preservation_and_half_close():
    class Echo(socketserver.BaseRequestHandler):
        def handle(self):
            data = bytearray()
            while True:
                chunk = self.request.recv(65536)
                if not chunk:
                    break
                data.extend(chunk)
            self.request.sendall(data)

    with socketserver.TCPServer(('127.0.0.1', 0), Echo) as backend:
        target = backend.server_address
        with server(('127.0.0.1', 0), target) as proxy:
            for service in [backend, proxy]:
                threading.Thread(target=service.serve_forever, daemon=True).start()
            expected = b'GET /synthetic HTTP/1.1\r\nUser-Agent: PRIVATE_SENTINEL\r\n\r\n' + bytes(range(256)) * 4096
            with socket.create_connection(proxy.server_address, timeout=5) as client:
                client.sendall(expected)
                client.shutdown(socket.SHUT_WR)
                received = bytearray()
                while True:
                    chunk = client.recv(65536)
                    if not chunk:
                        break
                    received.extend(chunk)
                assert bytes(received) == expected
            proxy.shutdown()
        backend.shutdown()
