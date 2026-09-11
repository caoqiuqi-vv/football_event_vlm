#!/usr/bin/env python3
"""List and launch the project's UI tools without relocating legacy entrypoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
CATALOG = Path(__file__).with_name("catalog.json")


def main() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="Show tools, ports and entrypoints")
    run = commands.add_parser("run", help="Launch one tool using this Python interpreter")
    run.add_argument("tool", choices=catalog)
    run.add_argument("args", nargs=argparse.REMAINDER, help="Arguments after -- go to the tool")
    args = parser.parse_args()
    if args.command == "list":
        for name, item in catalog.items():
            print(f"{name:20} {item['default_port']:5}  {item['description']}")
            print(f"  {item['entrypoint']}")
        return
    entrypoint = (ROOT / catalog[args.tool]["entrypoint"]).resolve()
    if not entrypoint.is_relative_to(ROOT / "tools") or not entrypoint.is_file():
        parser.error(f"Invalid tool entrypoint: {entrypoint}")
    forwarded = args.args[1:] if args.args[:1] == ["--"] else args.args
    # Preserve the caller's cwd so relative input/output paths keep their meaning.
    # exec also preserves Ctrl+C and the original service's exit status.
    os.execv(sys.executable, [sys.executable, str(entrypoint), *forwarded])


if __name__ == "__main__":
    main()
