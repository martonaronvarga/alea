#!/usr/bin/env python3
"""Run a CI command and expose a bounded failure excerpt in check annotations.

Full logs stay in the normal job output and the selected file. Annotations make
the actual diagnostic available to contributors without log-download privileges.
"""
from __future__ import annotations

import argparse
from collections import deque
import os
from pathlib import Path
import subprocess
import sys


def annotation(text: str) -> str:
    """Escape workflow-command data so log text cannot create extra commands."""
    return text.replace("\0", "").replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def run(command: list[str], log: Path) -> int:
    tail: deque[str] = deque(maxlen=80)
    with log.open("w", encoding="utf-8") as output:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace") as process:
            assert process.stdout is not None
            for line in process.stdout:
                output.write(line)
                output.flush()
                print(line, end="", flush=True)
                tail.append(line[-4000:])
            status = process.wait()
    if status:
        excerpt = "".join(tail)[-14000:]
        print("::error title=CI command failed::" + annotation(excerpt), flush=True)
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with Path(summary).open("a", encoding="utf-8") as output:
                output.write("\nCommand failed; final diagnostic output:\n\n")
                output.write("<pre>" + excerpt.replace("&", "&amp;").replace("<", "&lt;")
                             .replace(">", "&gt;") + "</pre>\n")
    return status if status >= 0 else 128 - status


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    sys.exit(run(command, args.log))


if __name__ == "__main__":
    main()
