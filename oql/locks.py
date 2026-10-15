"""Locks that make the rules of a test hold in code, not in good intentions.

freeze(files)   SHA-256 fingerprints of the pre-registration and experiment at the moment they are fixed.
verify(frozen)  refuses if any frozen file changed since.
claim(path)     a one-shot lock: written BEFORE the result is computed, refuses forever after.
"""
from __future__ import annotations

import hashlib
import json
import os
import time


class LockError(RuntimeError):
    pass


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def freeze(files: list[str]) -> dict:
    return {"frozen_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
            "files": {os.path.basename(f): sha256(f) for f in files}}


def verify(frozen: dict, folder: str) -> None:
    for name, h in frozen["files"].items():
        p = os.path.join(folder, name)
        if not os.path.exists(p) or sha256(p) != h:
            raise LockError(f"{name} changed after it was frozen on {frozen['frozen_utc']} UTC. "
                            f"A changed test is a new test: start a new card (oql card revise).")


def claim(path: str, what: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            rec = json.load(f)
        raise LockError(f"{what} was already run on {rec['claimed_utc']} UTC. It is one-shot.")
    rec = {"what": what, "claimed_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime())}
    with open(path, "x") as f:                        # 'x' fails if another process got there first
        json.dump(rec, f)
    return rec
