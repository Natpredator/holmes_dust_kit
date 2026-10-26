#!/usr/bin/env python3
"""
inspect_java_results.py
=======================
Summarise ONE folder of old Java/Orekit output (b<bin>p<N>.csv, b<bin>f<NN>.csv,
b<bin>m<N>.csv) without loading it all into memory: which epoch and which
Java program it most likely came from, how many particles finished, the
output cadence, the covered dates, and whether the f/m files are complete
or were cut off because the Java program never closed them.

Usage (from the project root):
    python tools/inspect_java_results.py /path/to/orekit-results
    python tools/inspect_java_results.py /path/to/orekit-results --markdown report.md

Everything printed is read from the files themselves; the "most likely Java
program" line is an INFERENCE from the run length and epoch index, and is
labelled as such.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # project root

from holmes_dust.legacy_java import (  # noqa: E402
    JAVA_1892_SNAPSHOTS,
    JAVA_2007_REFACTOR_SNAPSHOTS,
    iter_legacy_rows,
    list_legacy_files,
)
from holmes_dust.particles import beta_from_radius_mm  # noqa: E402

# Run lengths hard-coded in the Java programs (`duration` in seconds / 86400).
KNOWN_DURATIONS_DAYS = {
    6205: "SlaveMode_Holmes2007*.java / *_original.java (duration = 536112000 s)",
    9131: "SlaveMode_Holmes1892_Refactor_2007.java (duration = 788918400 s)",
    48910: "SlaveMode_Holmes1892_Refactor.java (duration = 4225824000 s)",
    50140: "SlaveMode_Holmes1892_Refactor_1892.java (duration = 4332096000 s)",
    13: "t1testing precision test (duration = 1123200 s)",
}
JAVA_EPOCHS = {9: "1892-11-04", 10: "1893-01-16", 11: "1899-07-04", 12: "2015-01-26",
               13: "1873-06-25", 14: "1894-04-23", 15: "1904-11-10"}
JAVA_EPOCHS.update({i: f"2007-10-{20 + i}" for i in range(9)})


ALL_SNAPSHOTS = {s.stem: s for ss in (JAVA_1892_SNAPSHOTS, JAVA_2007_REFACTOR_SNAPSHOTS) for s in ss}


def scan_p_file(path: str) -> dict:
    """One pass over a p-file: row count, first/last row, JD step check,
    and which known f/m snapshot conditions its rows satisfy."""
    matches = collections.Counter()
    n = 0
    first = last = None
    bad_steps = 0
    prev_jd = None
    for row in iter_legacy_rows(path):
        if first is None:
            first = row
        if prev_jd is not None and row["jd"] - prev_jd != 1:
            bad_steps += 1
        prev_jd = row["jd"]
        last = row
        n += 1
        for stem, snap in ALL_SNAPSHOTS.items():
            if snap.matches(row["jd"], row["ra_deg"], row["dec_deg"]):
                matches[stem] += 1
    truncated_tail = False                               # does the file end mid-line?
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            truncated_tail = f.read(1) != b"\n"
    return dict(rows=n, first=first, last=last, non_daily_steps=bad_steps, truncated_tail=truncated_tail,
                matches=matches)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--markdown", help="also write the report to this .md file")
    args = ap.parse_args(argv)

    files = list_legacy_files(args.folder)
    if not files:
        print(f"no b<bin>[pfm]<N>.csv files in {args.folder}")
        return 1
    out: list[str] = []
    say = out.append

    say(f"# Java results folder: {os.path.abspath(args.folder)}\n")
    by_kind = collections.defaultdict(list)
    for bin_no, kind, number, path in files:
        by_kind[(bin_no, kind)].append((number, path))

    for bin_no in sorted({b for b, _ in by_kind}):
        p_files = sorted(by_kind.get((bin_no, "p"), []))
        if not p_files:
            others = by_kind.get((bin_no, "f"), []) + by_kind.get((bin_no, "m"), [])
            empty = all(os.path.getsize(p) == 0 for _, p in others)
            say(f"## Bin b{bin_no}: no particle files; {len(others)} f/m files "
                f"{'all empty (this bin was not run)' if empty else 'present'}\n")
            continue
        numbers = [n for n, _ in p_files]
        missing = sorted(set(range(1, max(numbers) + 1)) - set(numbers))
        say(f"## Bin b{bin_no}: {len(p_files)} particle files (b{bin_no}p{min(numbers)}..b{bin_no}p{max(numbers)})")
        if missing:
            say(f"- p-files MISSING from the folder: {len(missing)} "
                f"(b{bin_no}p{missing[0]}..b{bin_no}p{missing[-1]}{', with gaps' if len(missing) != missing[-1] - missing[0] + 1 else ''}); "
                f"run_scenario.py --reproduce-java rebuilds them from the forecast files")
        scans = {num: scan_p_file(path) for num, path in p_files}
        full = max(s["rows"] for s in scans.values())
        complete = [n for n, s in scans.items() if s["rows"] == full]
        partial = [n for n, s in scans.items() if 0 < s["rows"] < full]
        empty = [n for n, s in scans.items() if s["rows"] == 0]
        any_full = scans[complete[0]]
        first, last = any_full["first"], any_full["last"]
        days = last["jd"] - first["jd"]
        say(f"- rows per complete particle: {full} (daily cadence: "
            f"{'yes' if all(s['non_daily_steps'] == 0 for s in scans.values()) else 'NO - irregular JD steps'})")
        say(f"- complete particles: {len(complete)}; partial: {partial or 'none'}; empty files: {empty or 'none'}")
        for n in partial:
            s = scans[n]
            if s["first"]["jd"] != first["jd"]:
                # the end is there but the beginning is not: damaged file (e.g. NUL bytes)
                say(f"  - b{bin_no}p{n}: {s['rows']} rows, but it STARTS at {s['first']['date']} -- the "
                    f"beginning of the file is damaged/unreadable (for example overwritten with zero bytes)")
            else:
                say(f"  - b{bin_no}p{n}: {s['rows']} rows, stops at {s['last']['date']} "
                    f"(the Java program was stopped while computing this particle)")
        say(f"- dates: {first['date']} -> {last['date']}  ({days} days)")
        epoch_idx = first["epoch_index"]
        say(f"- Java epoch index (OutburstAndInitialDateIndex): {epoch_idx} = outburst "
            f"{JAVA_EPOCHS.get(epoch_idx, 'unknown')}")
        guess = KNOWN_DURATIONS_DAYS.get(days)
        say(f"- INFERENCE, most likely program: {guess or 'no Java program with this exact length is known'}")
        radii = [s["first"]["radius_mm"] for s in scans.values() if s["first"]]
        say(f"- radius range: {min(radii):.5f} .. {max(radii):.5f} mm; beta range: "
            f"{beta_from_radius_mm(max(radii)):.4f} .. {beta_from_radius_mm(min(radii)):.4f}")
        say(f"- 'step' column of the last row: {last['step']} vs {full} rows "
            f"({'corrupted counter, as in the Java code' if last['step'] != full else 'clean'})")
        label_drift = last["date"][11:19]
        if label_drift != first["date"][11:19]:
            say(f"- time-of-day label drifts {first['date'][11:19]} -> {label_drift} "
                f"(leap seconds; Java added 86400 s of atomic time per step)")

        # f / m files: compare with what the p-files imply
        for kind in ("f", "m"):
            for num, path in sorted(by_kind.get((bin_no, kind), [])):
                stem = f"{kind}{num}"
                n_rows = sum(1 for _ in iter_legacy_rows(path))
                snap = ALL_SNAPSHOTS.get(stem)
                if snap is None:
                    say(f"- b{bin_no}{stem}.csv: {n_rows} rows (definition not known to this tool)")
                    continue
                # Rows the p-files say this file SHOULD contain (same condition as the Java code).
                expected = sum(sc["matches"][stem] for sc in scans.values())
                if n_rows == expected:
                    status = "OK"
                elif n_rows > expected:
                    status = ("more rows than the p-files present: the file has rows of particles whose "
                              "p-files are not in this folder")
                else:
                    status = "INCOMPLETE (file was never flushed/closed)"
                say(f"- b{bin_no}{stem}.csv ({snap.date}): {n_rows} rows, the p-files imply {expected} -> {status}")
        if any(1 for (b, k) in by_kind if b == bin_no and k in "fm"):
            say(f"- complete f/m files can be regenerated with: "
                f"python tools/rebuild_legacy_snapshots.py {args.folder} <out_dir>")
        say("")

    text = "\n".join(out)
    print(text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
