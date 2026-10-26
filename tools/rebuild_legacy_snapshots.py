#!/usr/bin/env python3
"""
rebuild_legacy_snapshots.py
===========================
Regenerate COMPLETE forecast (b<bin>f<NN>.csv) and matching (b<bin>m<N>.csv)
files from the per-particle b<bin>p<N>.csv files.

Why: the Java program never closed its f/m writers, so in the real result
folders those files contain only what happened to be flushed (e.g. 63 of 78
particles in every f-file, and nothing at all in m2/m14). The p-files are
complete, and every f/m row is just a copy of one p-file row (minus the
"step" field), so the f/m files can be rebuilt exactly.

Works for old Java folders and for legacy-layout output of this package
(run_scenario.py --output-format legacy|both).

Usage (from the project root):
    python tools/rebuild_legacy_snapshots.py /path/to/orekit-results ./rebuilt
    python tools/rebuild_legacy_snapshots.py /path/to/results ./rebuilt --set 2007
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # project root

from holmes_dust.legacy_java import (  # noqa: E402
    JAVA_1892_SNAPSHOTS,
    JAVA_2007_REFACTOR_SNAPSHOTS,
    rebuild_snapshot_files,
)

SETS = {
    "1892": JAVA_1892_SNAPSHOTS,            # SlaveMode_Holmes1892_Refactor_1892.java / _Refactor.java
    "2007": JAVA_2007_REFACTOR_SNAPSHOTS,   # SlaveMode_Holmes1892_Refactor_2007.java
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("p_folder", help="folder with b<bin>p<N>.csv files")
    ap.add_argument("out_folder", help="where to write the rebuilt f/m files (created if missing)")
    ap.add_argument("--set", choices=sorted(SETS), default="1892", help="which Java program's f/m definitions")
    args = ap.parse_args(argv)
    if os.path.abspath(args.p_folder) == os.path.abspath(args.out_folder):
        print("refusing to overwrite the original f/m files: choose a different out_folder", file=sys.stderr)
        return 1
    counts = rebuild_snapshot_files(args.p_folder, args.out_folder, SETS[args.set])
    for name, n in counts.items():
        print(f"{name:12s} {n:6d} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
