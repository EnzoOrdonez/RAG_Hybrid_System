"""Offline operator tool; never loads models or modifies historical interviews."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ui.components.session_manager import SESSIONS_DIR
from src.ui.components.session_storage import InvitationStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    issue = commands.add_parser("invite")
    issue.add_argument("participant_id")
    abandon = commands.add_parser("abandon", help="Revoke and release an active interview; preserve all files")
    abandon.add_argument("session_id")
    args = parser.parse_args()
    store = InvitationStore(SESSIONS_DIR)
    if args.command == "invite":
        print(store.issue(args.participant_id))  # display once; deliver through an authorized channel
    else:
        store.abandon(args.session_id)
        print("Invitation revoked; interview files preserved.")


if __name__ == "__main__":
    main()
