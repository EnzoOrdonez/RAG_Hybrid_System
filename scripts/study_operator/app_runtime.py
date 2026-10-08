"""Run the unchanged app behind private relays in a network=none container."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import signal
import subprocess
import threading

from scripts.study_operator.network_relay import server


@contextmanager
def relays(ollama_socket, streamlit_socket, *, factory=server):
    instances, threads = [], []
    try:
        for listen, target in [(('127.0.0.1', 11434), ollama_socket), (streamlit_socket, ('127.0.0.1', 8501))]:
            instance = factory(listen, target)
            instances.append(instance)
            thread = threading.Thread(target=instance.serve_forever, daemon=True)
            thread.start()
            threads.append(thread)
        yield
    finally:
        for instance in instances:
            instance.shutdown()
            instance.server_close()
        for thread in threads:
            thread.join(timeout=5)


def run_child(arguments, *, launch=subprocess.Popen):
    """Keep relay threads alive; exec would destroy them before Streamlit serves."""
    child = launch(arguments, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        return child.wait()
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('freeze', 'verify', 'serve', 'gate', 'backup', 'stimulus'))
    parser.add_argument('--deployment', required=True)
    parser.add_argument('--ollama-socket', required=True)
    parser.add_argument('--streamlit-socket', required=True)
    parser.add_argument('--request')
    parser.add_argument('--output')
    parser.add_argument('--boot-index', type=int, choices=range(1, 13))
    args = parser.parse_args(argv)
    from scripts import cloud_entrypoint

    def shutdown(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    with relays(args.ollama_socket, args.streamlit_socket):
        if args.operation == 'stimulus':
            from scripts.study_operator.stimulus_daemon import serve

            if args.boot_index is None:
                raise ValueError('Registered cold boot index required')
            deployment = json.loads(Path(args.deployment).read_text(encoding='utf-8'))
            return serve(deployment, args.boot_index)
        if args.operation == 'serve':
            from src.ui.components.study_sessions import StudyStore

            deployment = json.loads(Path(args.deployment).read_text(encoding='utf-8'))
            cloud_entrypoint.configure(deployment)
            protocol = cloud_entrypoint.verify(deployment)
            store = StudyStore(deployment['session_root'], protocol, deployment['purpose'])
            store.freeze()
            return run_child(cloud_entrypoint.streamlit_command(deployment))
        forwarded = [args.operation, '--deployment', args.deployment]
        if args.request:
            forwarded.extend(['--request', args.request])
        if args.output:
            forwarded.extend(['--output', args.output])
        return cloud_entrypoint.main(forwarded)


if __name__ == '__main__':
    raise SystemExit(main())
