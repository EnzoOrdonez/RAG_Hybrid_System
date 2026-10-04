import socket
import socketserver
import threading

from scripts.study_operator.network_relay import server


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
