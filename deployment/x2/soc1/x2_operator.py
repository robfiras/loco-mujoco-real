"""On the laptop: the operator's end of the link to a controller running on the X2's SoC1.

::

    source ~/x2_aimdk/env.sh
    python3 deployment/x2/soc1/x2_operator.py [--gamepad /dev/input/js0]

Keys from this terminal and a USB gamepad go into the controller's key pipe (``run/keys``,
see start.sh) over one SSH connection through the robot's relay, with a heartbeat byte
``HEARTBEAT_HZ`` times a second. A controller started with ``--heartbeat`` (start.sh does)
falls back to the default pose when the heartbeat stops, Wi-Fi lost or this program
closed, and ignores every key but damp until the link has been back for a second. The
connection is reopened on its own.

=============  =========  ==================================================
keyboard       gamepad
=============  =========  ==================================================
Enter, space   A          start the policy / play the clip / replay
s              B          the default pose (safety layer)
n / p          RB / LB    next / previous clip
Ctrl-C         LT + B     damp and stop
=============  =========  ==================================================

Ctrl-C damps and ends the session; no letter key damps, so none can be hit by accident.
Closing the terminal, or losing the link, leaves the robot in the default pose (see above)
until damp (``x2ctl.sh stop``) or a new operator.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import termios
import threading
import time
import tty
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]
from lmj_aimdk import Gamepad  # noqa: E402

HEARTBEAT_HZ = 10
#: Runs on SoC1: copies this program's output into the key pipe. The pipe is opened without
#: waiting, which fails unless a controller is reading it, so nothing pressed while no
#: controller runs is held back for the next one; and a write after the controller has quit
#: ends it, which shows here as the link going down.
PIPE_WRITER = """import os, sys
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_NONBLOCK)
os.set_blocking(fd, True)
while chunk := os.read(0, 64):
    os.write(fd, chunk)
"""
KEYS = {"\n": b"\n", "\r": b"\n", " ": b"\n", "s": b"s", "n": b"n", "p": b"p"}
NAMES = {b"\n": "start / play", b"s": "default pose", b"n": "next clip", b"p": "previous clip",
         b"d": "DAMP"}


class Link:
    """One SSH process feeding the controller's key pipe, reopened whenever it dies."""

    def __init__(self, relay: str, soc1: str):
        # Only a pipe the controller made: never create a plain file in its place.
        remote = ("p=~/lmj_deploy/run/keys; test -p $p && exec python3 -c "
                  + shlex.quote(PIPE_WRITER) + " $p")
        self.cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                    "-o", "ServerAliveInterval=1", "-o", "ServerAliveCountMax=2",
                    "-J", relay, soc1, remote]
        self.proc: subprocess.Popen | None = None
        self.up = False
        self._closed = False
        self._lock = threading.Lock()
        threading.Thread(target=self._keep_up, daemon=True).start()

    def send(self, data: bytes) -> bool:
        with self._lock:
            proc = self.proc
            if proc is None or proc.poll() is not None:
                return False
            try:
                proc.stdin.write(data)
                proc.stdin.flush()
                return True
            except OSError:
                return False

    def close(self) -> None:
        """End the connection for good: it is not reopened."""
        with self._lock:
            self._closed = True
            if self.proc is not None:
                self.proc.terminate()

    def _keep_up(self) -> None:
        while not self._closed:
            proc = self.proc
            if proc is None or proc.poll() is not None:
                if self._closed:
                    return
                if self.up:
                    say("link DOWN: the robot falls back to the default pose; reconnecting")
                self.up = False
                with self._lock:
                    if self._closed:
                        return
                    # Its own session: the terminal's Ctrl-C must not reach it, or it
                    # is gone before the damp that Ctrl-C sends. It ends when its input
                    # closes, so it never outlives this program.
                    self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE,
                                                 stdout=subprocess.DEVNULL,
                                                 stderr=subprocess.DEVNULL,
                                                 start_new_session=True)
                time.sleep(1.5)
                if self.proc.poll() is None:
                    self.up = True
                    say("link up (the controller accepts keys a second after its heartbeat "
                        "returns)")
            time.sleep(0.2)


class Sender:
    """What :class:`lmj_aimdk.Gamepad` expects of a keyboard, sending each press instead."""

    class _Key:
        def __init__(self, link: Link, data: bytes):
            self.link, self.data = link, data

        def set(self) -> None:
            press(self.link, self.data)

    def __init__(self, link: Link):
        self.start = Sender._Key(link, b"\n")
        self.safe = Sender._Key(link, b"s")
        self.damp = Sender._Key(link, b"d")
        self._link = link

    def press_step(self, step: int) -> None:
        press(self._link, b"n" if step > 0 else b"p")


def say(text: str) -> None:
    print(f"\r[operator] {text}", flush=True)


def press(link: Link, data: bytes) -> None:
    sent = link.send(data)
    say(f"{NAMES[data]}: " + ("sent" if sent else "NOT sent, link down"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relay", default=os.environ.get("X2_RELAY", "agi@192.168.88.88"))
    ap.add_argument("--soc1", default=os.environ.get("X2_SOC1", "agi@10.0.1.41"))
    ap.add_argument("--gamepad", default="/dev/input/js0", help="'none' to leave it out")
    args = ap.parse_args()

    link = Link(args.relay, args.soc1)

    def heartbeat() -> None:
        while True:
            link.send(b"h")
            time.sleep(1.0 / HEARTBEAT_HZ)

    threading.Thread(target=heartbeat, daemon=True).start()
    if args.gamepad != "none" and os.path.exists(args.gamepad):
        try:
            Gamepad(Sender(link), args.gamepad)
            say(f"{Gamepad.HELP} ({args.gamepad})")
        except OSError as e:
            say(f"no gamepad: {e}")
    say("keys: Enter start / play | s default pose | n p clip | Ctrl-C damp and stop")

    fd = sys.stdin.fileno()
    damped = False
    saved = termios.tcgetattr(fd) if os.isatty(fd) else None
    if saved:
        tty.setcbreak(fd)
    try:
        while chunk := os.read(fd, 1):
            key = chunk.decode(errors="ignore").lower()
            if key in KEYS:
                press(link, KEYS[key])
    except KeyboardInterrupt:
        damped = link.send(b"d")
        say("DAMP: sent; the controller damps and stops" if damped
            else "DAMP NOT sent, link down")
        time.sleep(0.5)  # out through SSH before the link closes
    finally:
        if saved:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        link.close()
        if not damped:
            say("LEFT without damping: the controller keeps the robot in the default pose, "
                "and only damp reaches it.")
            say("  reconnect:  deployment/x2/soc1/x2ctl.sh")
            say("  stop (damp): deployment/x2/soc1/x2ctl.sh stop")


if __name__ == "__main__":
    main()
