#!/usr/bin/env python3
"""
plan_observations.py
====================
Hour-by-hour topocentric sky position (RA/DEC, altitude, azimuth) of a dust-
trail target for a list of observatories -- a generalisation of the
project's Skyfield scripts (holmes_<year>_<date>_<site>.py, e.g.
holmes_1892_2027_2_kouvola.py), writing the SAME text line format.

Target modes
------------
--point X Y Z
    A FIXED point in space, barycentric ICRF, AU -- exactly what the old
    scripts do (class Holmes_Dust_Trail_Stationary, velocity [0, 0, 0]).
    Appropriate for a fixed location of the trail tube (e.g. its node).
--track FOLDER_OR_CSV --particles N [N ...]
    The MEAN position of the given particles, MOVING with them, taken from
    the daily barycentric positions of a run (b1p<N>.csv files of the Java
    program or of run_scenario.py --output-format legacy/both, or a modern
    trajectory_full.csv with barycentric_* columns) and interpolated
    linearly in time (error < 1 arcsec for daily rows).
--track FOLDER_OR_CSV --near X Y Z --on YYYY-MM-DD --radius R
    Same, but the particles are selected as those within R AU of the point
    (X, Y, Z) on that date -- i.e. "the clump of dust that is at this point
    on the forecast date", followed as it moves. For comparison: a Java
    particle sitting at the 2027-02 filament-1 point of the old scripts
    moves ~0.01 AU per day, ~3.5 AU over the scripts' 8000-hour window, so a
    moving clump and a fixed point give very different tracks.

Requires skyfield (pip install skyfield) and downloads/uses de421.bsp, like
the original scripts.

Examples (from the project root):
    python tools/plan_observations.py --point -1.790431897831839 -2.1448241371064594 -2.273927653630378 \\
        --label Kouvola_1892_2027_2_FILAMENT1 --site Kouvola --start 2027-01-01T00 --hours 8000
    python tools/plan_observations.py --track output/repro_1892 --near -1.79 -2.14 -2.27 \\
        --on 2027-02-01 --radius 0.1 --site Kouvola Utah --start 2027-01-15T00 --hours 720 --out plans/
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # project root

from holmes_dust.legacy_java import iter_legacy_rows, julian_day_number, list_legacy_files  # noqa: E402

# Observatories used in the project's Skyfield scripts: (lat deg, lon deg, elevation m); S/W negative.
SITES = {
    "Kouvola": (60.86667, 26.7, 81.0),
    "Utah": (37.7378, -113.6975, 1570.0),
    "SidingSpring": (-31.273333, 149.064444, 1122.0),
    "RioHurtado": (-30.526391, -70.8536101, 1731.6),
    "Feregnal": (38.215662, 353.37251, 582.8),      # written as 353.37251 E in the original scripts
}
AU_PER_M = 6.68458712e-12      # same factor as the Java/CSV AU columns


# ---------------------------------------------------------------------------
# Loading daily barycentric positions
# ---------------------------------------------------------------------------

def load_positions(source: str, jd_min: int, jd_max: int) -> dict[int, dict[int, np.ndarray]]:
    """{particle number: {JDN: xyz_au}} for JDN in [jd_min, jd_max]."""
    out: dict[int, dict[int, np.ndarray]] = {}
    if os.path.isdir(source):
        for bin_no, kind, number, path in list_legacy_files(source):
            if kind != "p":
                continue
            rows = out.setdefault(number, {})
            for r in iter_legacy_rows(path):
                if jd_min <= r["jd"] <= jd_max:
                    rows[r["jd"]] = np.array([r["x_au"], r["y_au"], r["z_au"]])
                elif r["jd"] > jd_max:
                    break
    else:
        import pandas as pd
        cols = ["julian_day", "particle_id", "barycentric_x_au", "barycentric_y_au", "barycentric_z_au"]
        for chunk in pd.read_csv(source, usecols=cols, chunksize=2_000_000):
            c = chunk[(chunk.julian_day >= jd_min) & (chunk.julian_day <= jd_max)].dropna()
            for rec in c.itertuples(index=False):
                out.setdefault(int(rec.particle_id) + 1, {})[int(rec.julian_day)] = np.array(
                    [rec.barycentric_x_au, rec.barycentric_y_au, rec.barycentric_z_au])
    return out


def select_near(positions, jdn: int, point, radius_au: float) -> list[int]:
    point = np.asarray(point, dtype=float)
    return sorted(n for n, rows in positions.items() if jdn in rows and np.linalg.norm(rows[jdn] - point) <= radius_au)


def mean_track(positions, numbers) -> tuple[np.ndarray, np.ndarray]:
    """(JD at noon, mean xyz) over the days where ALL chosen particles have a row."""
    common = sorted(set.intersection(*(set(positions[n]) for n in numbers)))
    jd = np.array(common, dtype=float)                       # JDN == JD at 12:00 UTC
    xyz = np.array([np.mean([positions[n][d] for n in numbers], axis=0) for d in common])
    return jd, xyz


# ---------------------------------------------------------------------------
# Skyfield targets
# ---------------------------------------------------------------------------

def make_targets():
    from skyfield.vectorlib import VectorFunction

    class StationaryPoint(VectorFunction):
        """Same as Holmes_Dust_Trail_Stationary in the old scripts."""
        center = 0                                            # Solar System Barycentre

        def __init__(self, xyz_au):
            self.xyz = np.asarray(xyz_au, dtype=float)
            self.target = self

        def _at(self, t):
            shape = np.shape(t.tdb)
            p = self.xyz if shape == () else np.repeat(self.xyz[:, None], shape[0], axis=1)
            return p, np.zeros_like(p), None, None

    class MovingTrack(VectorFunction):
        """Linear interpolation of daily barycentric positions (AU)."""
        center = 0

        def __init__(self, jd, xyz):
            self.jd, self.xyz, self.target = jd, xyz, self
            self.vel = np.gradient(xyz, jd, axis=0)          # AU/day, only used for aberration bookkeeping

        def _at(self, t):
            # Table rows are at 12:00 UTC; t.tdb differs from UTC by ~69 s --
            # negligible here (< 0.01 arcsec for dust moving ~0.01 AU/day).
            tj = np.atleast_1d(t.tdb)
            if tj.min() < self.jd[0] or tj.max() > self.jd[-1]:
                raise ValueError("requested time outside the loaded daily positions; widen the window")
            p = np.array([np.interp(tj, self.jd, self.xyz[:, k]) for k in range(3)])
            v = np.array([np.interp(tj, self.jd, self.vel[:, k]) for k in range(3)])
            if np.ndim(t.tdb) == 0:
                p, v = p[:, 0], v[:, 0]
            return p, v, None, None

    return StationaryPoint, MovingTrack


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--point", type=float, nargs=3, metavar=("X", "Y", "Z"), help="fixed barycentric point, AU")
    g.add_argument("--track", metavar="FOLDER_OR_CSV", help="run output with daily barycentric positions")
    ap.add_argument("--particles", type=int, nargs="+", help="with --track: particle numbers (b1p<N>)")
    ap.add_argument("--near", type=float, nargs=3, metavar=("X", "Y", "Z"), help="with --track: select near this point")
    ap.add_argument("--on", help="with --near: date YYYY-MM-DD of the selection")
    ap.add_argument("--radius", type=float, default=0.1, help="with --near: selection radius, AU")
    ap.add_argument("--site", nargs="+", default=list(SITES), choices=list(SITES))
    ap.add_argument("--start", required=True, help="UTC start, e.g. 2027-01-01T00 (first line is start + step)")
    ap.add_argument("--hours", type=int, default=8000, help="number of lines (the old scripts used 4000/8000)")
    ap.add_argument("--step-hours", type=float, default=1.0)
    ap.add_argument("--label", default="Holmes", help="text at the start of every line")
    ap.add_argument("--temperature", type=float, default=22.0, help="deg C, for refraction (old scripts: 22)")
    ap.add_argument("--pressure", type=float, default=1005.0, help="mbar, for refraction (old scripts: 1005)")
    ap.add_argument("--out", help="folder for one <label>_<site>.txt per site (default: print)")
    args = ap.parse_args(argv)

    try:
        from skyfield.api import load, wgs84
    except ImportError:
        print("this tool needs skyfield:  pip install skyfield", file=sys.stderr)
        return 1
    StationaryPoint, MovingTrack = make_targets()

    start = dt.datetime.fromisoformat(args.start)
    end = start + dt.timedelta(hours=args.step_hours * (args.hours + 1))
    if args.point is not None:
        target = StationaryPoint(args.point)
        print(f"# target: fixed point {args.point} AU (barycentric ICRF)", file=sys.stderr)
    else:
        jd_min, jd_max = julian_day_number(start) - 2, julian_day_number(end) + 2
        if args.on:
            jd_on = julian_day_number(dt.date.fromisoformat(args.on))
            jd_min, jd_max = min(jd_min, jd_on), max(jd_max, jd_on)
        positions = load_positions(args.track, jd_min, jd_max)
        if args.particles:
            numbers = args.particles
        elif args.near and args.on:
            numbers = select_near(positions, jd_on, args.near, args.radius)
        else:
            print("--track needs --particles, or --near X Y Z --on DATE", file=sys.stderr)
            return 1
        if not numbers:
            print("no particles selected (increase --radius?)", file=sys.stderr)
            return 1
        jd, xyz = mean_track(positions, numbers)
        target = MovingTrack(jd, xyz)
        print(f"# target: mean of {len(numbers)} moving particles {numbers[:20]}{' ...' if len(numbers) > 20 else ''}",
              file=sys.stderr)

    ts = load.timescale()
    planets = load("de421.bsp")
    earth = planets["earth"]
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    for site_name in args.site:
        lat, lon, elev = SITES[site_name]
        site = earth + wgs84.latlon(lat, lon, elevation_m=elev)
        label = f"{args.label}" if len(args.site) == 1 else f"{site_name}_{args.label}"
        sink = open(os.path.join(args.out, f"{label}.txt"), "w", encoding="utf-8") if args.out else sys.stdout
        t = ts.utc(start.year, start.month, start.day, start.hour, start.minute)
        for _ in range(args.hours):
            t = t + args.step_hours / 24.0                    # like the old scripts: step first, then print
            app = site.at(t).observe(target).apparent()
            alt, az, _distance = app.altaz(temperature_C=args.temperature, pressure_mbar=args.pressure)
            ra, dec, _d = app.radec()
            print(label, "", t.utc_jpl(), "JD: {:.10f}".format(t.ut1), "RADEC: {:.8f}".format(ra.hours * 15),
                  "DeclDEC: {:.8f}".format(dec.degrees), "RA:", ra, "Decl:", dec, "Alt:", alt.dstr(),
                  "Az:", az.dstr(), file=sink)
        if sink is not sys.stdout:
            sink.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
