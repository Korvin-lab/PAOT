#!/usr/bin/env python3
"""Checkpointed scan of a very large final CSV, safe under command time limits."""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"offset": 0, "rows": 0, "calculated": set(), "finished": False}
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_state(path: Path, state: dict) -> None:
    temp = path.with_suffix(path.suffix + ".next")
    with temp.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-csv", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=18.0)
    args = parser.parse_args()

    state = load_state(args.state)
    if state["finished"]:
        print(f"DONE rows={state['rows']} pairs={len(state['calculated'])}")
        return
    deadline = time.monotonic() + args.seconds
    with args.final_csv.open("rb", buffering=16 * 1024 * 1024) as source:
        if state["offset"] == 0:
            header = source.readline().decode("utf-8-sig").rstrip("\r\n").split(",")
            if header[:3] != ["date", "id", "Qv"] or len(header) != 51:
                raise ValueError("Unexpected final CSV schema")
        else:
            source.seek(state["offset"])

        # Segment rows belonging to one pipe-date are contiguous.  Decode and
        # retain a key only when the profile changes, avoiding millions of
        # temporary Python strings during a 92-GB scan.
        last_key = None
        while time.monotonic() < deadline:
            line = source.readline()
            if not line:
                state["finished"] = True
                break
            state["rows"] += 1
            if line.count(b",") != 50:
                raise ValueError(f"Malformed row {state['rows'] + 1}")
            first = line.find(b",")
            second = line.find(b",", first + 1)
            key = line[:second]
            if key != last_key:
                date_value, pipe_id = key.split(b",", 1)
                state["calculated"].add((pipe_id.decode("ascii"), date_value.decode("ascii")))
                last_key = key
        state["offset"] = source.tell()
    save_state(args.state, state)
    print(f"CHECKPOINT rows={state['rows']} pairs={len(state['calculated'])} offset={state['offset']} finished={state['finished']}")


if __name__ == "__main__":
    main()
