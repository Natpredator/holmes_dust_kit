#!/usr/bin/env python3
"""
compare_with_java.py
====================
Compare a run of this package against an old Java/Orekit result folder.

Two modes, chosen automatically:

1. PAIRED (the new folder has legacy-layout b<bin>p<N>.csv files with the
   same particle numbers -- i.e. it was made with
   `run_scenario.py --reproduce-java JAVA_DIR`): for every particle and every
   common day, the angular separation of the RA/DEC positions and the 3-D
   distance of the barycentric positions. This is a direct, particle-by-
   particle check of the new integrator against the old one.

2. STATISTICAL (the new folder only has trajectory_full.csv, i.e. different
   random particles): per day, the centre (mean direction) of the dust cloud
   on the sky and its spread (median / 90th-percentile distance of the
   particles from that centre), for both runs.

Usage (from the project root; pandas required):
    python tools/compare_with_java.py JAVA_DIR NEW_DIR
    python tools/compare_with_java.py JAVA_DIR NEW_DIR --days 0 1 10 100 1000 10000 50140 --csv cmp.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # project root

from holmes_dust.legacy_java import iter_legacy_rows, list_legacy_files  # noqa: E402

DEFAULT_DAYS = [0, 1, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 50140]


def unit_vectors(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


def separation_deg(ra1, dec1, ra2, dec2):
    """Great-circle distance, degrees (vectorised). atan2(|a x b|, a.b) stays
    accurate for tiny angles, where arccos(a.b) loses ~1e-6 deg to rounding."""
    a, b = unit_vectors(ra1, dec1), unit_vectors(ra2, dec2)
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    dot = np.sum(a * b, axis=-1)
    return np.degrees(np.arctan2(cross, dot))


def load_p(path, cols=("jd", "ra_deg", "dec_deg", "x_au", "y_au", "z_au")):
    import pandas as pd
    return pd.DataFrame([{c: r[c] for c in cols} for r in iter_legacy_rows(path)])


def paired(java_dir, new_dir, days, csv_out):
    import pandas as pd
    java = {(b, n): p for b, k, n, p in list_legacy_files(java_dir) if k == "p"}
    new = {(b, n): p for b, k, n, p in list_legacy_files(new_dir) if k == "p"}
    common = sorted(set(java) & set(new))
    print(f"paired comparison: {len(common)} particles present in both folders")
    per_particle, per_day = [], []
    for key in common:
        a, b = load_p(java[key]), load_p(new[key])
        m = a.merge(b, on="jd", suffixes=("_java", "_new"))
        if m.empty:
            continue
        sep = separation_deg(m.ra_deg_java.values, m.dec_deg_java.values, m.ra_deg_new.values, m.dec_deg_new.values)
        dxyz = np.sqrt(sum((m[f"{c}_au_java"] - m[f"{c}_au_new"]) ** 2 for c in "xyz")).values
        day = (m.jd - b.jd.iloc[0]).values          # days since the outburst (first row of the new run)
        per_particle.append(dict(bin=key[0], particle=key[1], days_compared=len(m),
                                 max_sep_deg=sep.max(), final_sep_deg=sep[-1], final_dist_au=dxyz[-1]))
        for d in days:
            i = np.searchsorted(day, d)
            if i < len(day) and day[i] == d:
                per_day.append(dict(day=d, sep_deg=sep[i], dist_au=dxyz[i]))
    pp = pd.DataFrame(per_particle)
    pd_ = pd.DataFrame(per_day)
    if pp.empty:
        print("no common days found")
        return
    summary = pd_.groupby("day").agg(n=("sep_deg", "size"), median_sep_deg=("sep_deg", "median"),
                                     max_sep_deg=("sep_deg", "max"), median_dist_au=("dist_au", "median"),
                                     max_dist_au=("dist_au", "max"))
    print("\nRA/DEC separation and barycentric distance, new vs Java, over all particles:")
    print(summary.to_string(float_format=lambda v: f"{v:.6g}"))
    print("\nworst particles (by final separation):")
    print(pp.sort_values("final_sep_deg", ascending=False).head(10).to_string(index=False))
    if csv_out:
        pp.to_csv(csv_out, index=False)
        print(f"\nper-particle table written to {csv_out}")


def cloud_stats(ra, dec):
    v = unit_vectors(ra, dec)
    c = v.mean(axis=0)
    c /= np.linalg.norm(c)
    sep = np.degrees(np.arccos(np.clip(v @ c, -1, 1)))
    return c, np.median(sep), np.percentile(sep, 90)


def statistical(java_dir, new_dir, days, csv_out):
    import pandas as pd
    traj = os.path.join(new_dir, "trajectory_full.csv")
    if not os.path.exists(traj):
        print(f"{new_dir} has neither b<bin>p<N>.csv files nor trajectory_full.csv")
        return
    java_files = [p for b, k, n, p in list_legacy_files(java_dir) if k == "p"]
    first = next(iter_legacy_rows(java_files[0]))
    wanted = {first["jd"] + d for d in days}
    jrows = []
    for p in java_files:
        for r in iter_legacy_rows(p):
            if r["jd"] in wanted:
                jrows.append((r["jd"], r["ra_deg"], r["dec_deg"]))
    java = pd.DataFrame(jrows, columns=["jd", "ra_deg", "dec_deg"])
    parts = []
    for chunk in pd.read_csv(traj, usecols=["julian_day", "ra_deg", "dec_deg"], chunksize=2_000_000,
                             on_bad_lines="skip"):
        parts.append(chunk[chunk.julian_day.isin(wanted)])
    new = pd.concat(parts).dropna()
    rows = []
    for d in days:
        j, n = java[java.jd == first["jd"] + d], new[new.julian_day == first["jd"] + d]
        if len(j) == 0 or len(n) == 0:
            continue
        cj, mj, pj = cloud_stats(j.ra_deg.values, j.dec_deg.values)
        cn, mn, pn = cloud_stats(n.ra_deg.values, n.dec_deg.values)
        rows.append(dict(day=d, n_java=len(j), n_new=len(n),
                         centre_sep_deg=float(np.degrees(np.arccos(np.clip(cj @ cn, -1, 1)))),
                         spread_median_java=mj, spread_median_new=mn, spread_p90_java=pj, spread_p90_new=pn))
    out = pd.DataFrame(rows)
    print("statistical comparison (different random particles): cloud centre and spread on the sky, degrees")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if csv_out:
        out.to_csv(csv_out, index=False)


def snapshots(java_dir, new_dir):
    """Forecast/matching files (b1f10.csv, ...): the same particle is found
    in both files by its theta value (identical in a --reproduce-java run),
    so this works even when the Java file is cut short or the Java p-files
    of some particles were deleted."""
    java = {(b, k, n): p for b, k, n, p in list_legacy_files(java_dir) if k in "fm"}
    new = {(b, k, n): p for b, k, n, p in list_legacy_files(new_dir) if k in "fm"}
    lines = []
    for key in sorted(set(java) & set(new)):
        a = {repr(r["theta"]): r for r in iter_legacy_rows(java[key])}
        b = {repr(r["theta"]): r for r in iter_legacy_rows(new[key])}
        common = sorted(set(a) & set(b))
        if not common:
            continue
        sep = separation_deg(np.array([a[t]["ra_deg"] for t in common]), np.array([a[t]["dec_deg"] for t in common]),
                             np.array([b[t]["ra_deg"] for t in common]), np.array([b[t]["dec_deg"] for t in common]))
        date = next(iter(a.values()))["date"][:10]
        lines.append(f"b{key[0]}{key[1]}{key[2]:<3d} {date}  particles {len(common):4d}  "
                     f"median {np.median(sep):9.5f}  90% {np.percentile(sep, 90):9.5f}  max {sep.max():9.4f}  "
                     f"<0.1 deg: {np.mean(sep < 0.1) * 100:5.1f} %")
    if lines:
        print("\nforecast/matching files, same particles, RA/DEC difference new vs Java (deg):")
        print("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("java_dir")
    ap.add_argument("new_dir")
    ap.add_argument("--days", type=int, nargs="+", default=DEFAULT_DAYS, help="days since outburst to report")
    ap.add_argument("--csv", help="write the detailed table to this CSV file")
    args = ap.parse_args(argv)
    has_p = any(k == "p" for _, k, _, _ in list_legacy_files(args.new_dir))
    (paired if has_p else statistical)(args.java_dir, args.new_dir, args.days, args.csv)
    snapshots(args.java_dir, args.new_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
