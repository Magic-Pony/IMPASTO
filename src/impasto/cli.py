"""Small command-line entry point for reproducible local runs."""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy


def main(argv=None):
    parser = argparse.ArgumentParser(prog="impasto")
    parser.add_argument("command", choices=("stage1",), help="run a reproduction stage")
    args = parser.parse_args(argv)
    if args.command == "stage1":
        root = Path(__file__).resolve().parents[2]
        runpy.run_path(str(root / "scripts/run_stage1.py"), run_name="__main__")


if __name__ == "__main__":
    main()
