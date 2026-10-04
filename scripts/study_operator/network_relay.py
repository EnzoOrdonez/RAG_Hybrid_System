"""Byte-preserving Unix/loopback relays; the app needs no external network."""
import argparse
import selectors
import socket
import socketserver
import threading


def transfer(left, right):
    """Bounded buffers and half-closes preserve HTTP and WebSocket bytes."""
    selector = selectors.DefaultSelector()
    for stream in (left, right):
        stream.setblocking(False)
        selector.register(stream, selectors.EVENT_READ)
    destinations = {left: right, right: left}
    pending = {left: bytearray(), right: bytearray()}
    read_open = {left: True, right: True}
    max_buffer = 256 * 1024
    try:
        while selector.get_map():
            for key, events in selector.select(timeout=60):
                stream = key.fileobj
                other = destinations[stream]
                if events & selectors.EVENT_READ:
                    try:
                        data = stream.recv(65536)
                    except BlockingIOError:
                        data = None
                    if data == b'':
                        read_open[stream] = False
                    elif data:
                        pending[other].extend(data)
                if events & selectors.EVENT_WRITE and pending[stream]:
                    try:
                        count = stream.send(pending[stream])
                        del pending[stream][:count]
                    except BlockingIOError:
                        pass
                for candidate in (stream, other):
                    peer = destinations[candidate]
                    if not read_open[peer] and not pending[candidate]:
                        try:
                            candidate.shutdown(socket.SHUT_WR)
                        except OSError:
                            pass
                    flags = 0
                    if read_open[candidate] and len(pending[peer]) < max_buffer:
                        flags |= selectors.EVENT_READ
                    if pending[candidate]:
                        flags |= selectors.EVENT_WRITE
                    registered = candidate in selector.get_map()
                    if flags:
                        if registered:
                            selector.modify(candidate, flags)
                        else:
                            selector.register(candidate, flags)
                    elif registered:
                        selector.unregister(candidate)
    finally:
        selector.close()


def server(listen, target):
    base = socketserver.UnixStreamServer if isinstance(listen, str) else socketserver.TCPServer

    class Relay(socketserver.ThreadingMixIn, base):
        daemon_threads = True
        allow_reuse_address = True

        def handle_error(self, request, client_address):
            # Never log client addresses, bytes or exception payloads.
            pass

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            outgoing = socket.socket(socket.AF_UNIX if isinstance(target, str) else socket.AF_INET, socket.SOCK_STREAM)
            with outgoing:
                outgoing.settimeout(10)
                outgoing.connect(target)
                transfer(self.request, outgoing)

    return Relay(listen, Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ollama-socket', required=True)
    parser.add_argument('--streamlit-socket', required=True)
    args = parser.parse_args()
    ollama = server(('127.0.0.1', 11434), args.ollama_socket)
    web = server(args.streamlit_socket, ('127.0.0.1', 8501))
    threading.Thread(target=ollama.serve_forever, daemon=True).start()
    web.serve_forever()


if __name__ == '__main__':
    main()
