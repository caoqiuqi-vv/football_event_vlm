#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
def main() -> None:
 p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('inputs',nargs='+'); a=p.parse_args()
 output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
 output.write_text(''.join(Path(x).read_text() for x in a.inputs))
if __name__=='__main__': main()
