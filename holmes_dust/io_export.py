"""
io_export.py
============
Streamed CSV output.

This module replaces ALL of the old Java program's individually-named
``PrintWriter`` fields (``b1outm2``, ``b1outf10``, ... -- dozens of them,
opened/closed by hand for every scenario) with:

* one streamed "full trajectory" CSV (every particle, every output epoch),
  written incrementally so memory use stays bounded even for scenarios with
  thousands of particles over a century-plus of daily-cadence output, and
* zero or more "snapshot" CSVs, one per `config.SnapshotSpec`, populated by
  filtering the very same rows as they are produced -- adding a new
  snapshot date is now a config change, not a code change.

Field names in OBSERVATION_FIELDS intentionally mirror the old CSV columns
(Julian day, date, geocentric position scaled by the old code's constant
6.68458712E-12 [metres -> AU], RA/DEC in both sexagesimal and decimal-degree
form, particle radius, bin, beta, ejection speed, phase angle) so a person
already familiar with the old output can read the new one directly, with a
few honestly-new columns appended (light_time_s; JPL Horizons runs also
carry a `comet_designation` column, useful once multiple target comets are
being explored side by side, as the newest Java file already started to do).
"""

from __future__ import annotations  # allows modern type-hint syntax on older Python

import csv            # Python's built-in CSV writer -- used instead of pandas for streaming
import dataclasses    # for the @dataclasses.dataclass decorator (see ObservationRecord)
import datetime as dt  # for the `datetime` type used by date_utc
import os              # for creating folders and joining file paths
from typing import IO, Optional   # IO[str] = "an open text file handle"; Optional = "or None"

from .config import SnapshotSpec               # the "which dates to also export" spec
import numpy as np

from .legacy_java import julian_day_number   # calendar date -> integer Julian Day Number
from .observations import Observation, to_dms, to_hms  # the RA/DEC result type + formatters

# Old Java code's hard-coded metres -> AU conversion factor
# (`pv2.getPosition().getX()*6.68458712E-12`). Kept as a named constant
# rather than recomputed from a modern AU-in-metres value, specifically so
# a byte-for-byte comparison against historical Java CSVs is possible.
METERS_TO_AU_LEGACY = 6.68458712e-12

# The exact column order every output CSV (main file AND every snapshot
# file) will use. Defined once, here, so the main writer and every snapshot
# writer always agree on the same columns.
OBSERVATION_FIELDS = [
    "julian_day",           # integer Julian Day number for this row's date
    "date_utc",              # the same date, as a human-readable ISO string
    "particle_id",           # which dust particle this row describes
    "bin_name",               # which size class that particle belongs to
    "radius_mm",              # that particle's own radius (millimetres)
    "beta",                    # that particle's radiation-pressure ratio
    "ejection_speed_m_s",      # that particle's ejection speed (m/s)
    "geocentric_x_au",         # position relative to Earth, x component (AU)
    "geocentric_y_au",         # position relative to Earth, y component (AU)
    "geocentric_z_au",         # position relative to Earth, z component (AU)
    "barycentric_x_au",        # position relative to the Solar-System Barycentre, x (AU) --
    "barycentric_y_au",        #   the frame of the old Java b1p*.csv x/y/z columns,
    "barycentric_z_au",        #   geometric (no light-time), same 6.68458712E-12 factor
    "ra_deg",                   # Right Ascension, decimal degrees
    "dec_deg",                  # Declination, decimal degrees
    "ra_hms",                    # Right Ascension, sexagesimal (hours/min/sec) string
    "dec_dms",                   # Declination, sexagesimal (deg/min/sec) string
    "phase_angle_deg",           # Sun-target-Earth phase angle, degrees
    "light_time_s",               # light-travel time used for the correction, seconds
    "theta",                      # ejection azimuth random number (old Java "theta" column)
    "u",                          # ejection z-component random number (old Java "u" column)
]


# A plain container: everything needed to write ONE output row (one
# particle, at one date). `as_csv_row()` turns it into the dict the csv
# module actually writes.
@dataclasses.dataclass
class ObservationRecord:
    """One output row: everything needed to describe one particle's
    synthetic observation at one epoch.

    Note: geocentric position for the CSV's AU columns is read straight off
    `observation.geocentric_position_m` (rather than being passed in
    separately) so it is always guaranteed consistent with the reported
    RA/DEC -- both reflect the light-time-corrected apparent position when
    that correction was applied.
    """
    date_utc: dt.datetime          # calendar date/time this row describes
    particle_id: int                # which particle
    bin_name: str                    # which size class
    radius_mm: float                  # that particle's radius
    beta: float                        # that particle's radiation-pressure ratio
    ejection_speed_m_s: float          # that particle's ejection speed
    observation: Observation           # the RA/DEC/phase-angle result from observations.py
    # Geometric SSB-frame position of the particle at date_utc (metres);
    # None -> the barycentric_* columns are left empty.
    barycentric_position_m: Optional[np.ndarray] = None
    theta: float = float("nan")         # the particle's ejection-direction random numbers
    u: float = float("nan")

    def as_csv_row(self) -> dict:
        """Build the dict of {column_name: value} that csv.DictWriter
        expects, in exactly the order/names OBSERVATION_FIELDS defines."""
        # Convert the geocentric position from metres (SI) to the old
        # Java program's AU convention, for the 3 geocentric_*_au columns.
        au = self.observation.geocentric_position_m * METERS_TO_AU_LEGACY
        bary = (self.barycentric_position_m * METERS_TO_AU_LEGACY
                if self.barycentric_position_m is not None else ("", "", ""))
        return {
            # Julian Day NUMBER of the calendar date (same convention as the
            # Java `JulianFields.JULIAN_DAY` column); equals the JD at noon.
            "julian_day": julian_day_number(self.date_utc),
            "date_utc": self.date_utc.isoformat(),                # human-readable date string
            "particle_id": self.particle_id,
            "bin_name": self.bin_name,
            "radius_mm": self.radius_mm,
            "beta": self.beta,
            "ejection_speed_m_s": self.ejection_speed_m_s,
            "geocentric_x_au": au[0],       # x component of the AU-scaled vector
            "geocentric_y_au": au[1],       # y component
            "geocentric_z_au": au[2],       # z component
            "barycentric_x_au": bary[0],
            "barycentric_y_au": bary[1],
            "barycentric_z_au": bary[2],
            "ra_deg": self.observation.ra_deg,
            "dec_deg": self.observation.dec_deg,
            "ra_hms": to_hms(self.observation.ra_deg),     # human-readable RA string
            "dec_dms": to_dms(self.observation.dec_deg),   # human-readable DEC string
            "phase_angle_deg": self.observation.phase_angle_deg,
            "light_time_s": self.observation.light_time_s,
            "theta": self.theta,
            "u": self.u,
        }


def _snapshot_matches(spec: SnapshotSpec, record: ObservationRecord, date_tolerance_s: float) -> bool:
    """Does this record belong in this snapshot's output file?

    Two conditions, mirroring the old code's two output styles:
      1. the record's date must be within `date_tolerance_s` of the
         snapshot's target date (old code compared full ISO date STRINGS
         for equality, which is equivalent to an exact-match tolerance at
         whatever cadence the output loop ran at);
      2. if the snapshot carries an observed RA/DEC (i.e. it is a
         "matching" snapshot, not a pure "forecast" one), the record's
         computed RA/DEC must additionally fall inside the configured
         tolerance box, exactly like the old
         `abs(alphaDeg - 242) < 40.0 and abs(deltaDeg - (-40.4)) < 40.0`.
    """
    # Condition 1: is this row's date close enough to the snapshot's target date?
    if abs((record.date_utc - spec.date_utc).total_seconds()) > date_tolerance_s:
        return False   # too far away in time -- definitely not a match

    # Condition 2 (only for "matching" snapshots -- ones with a real
    # historical observation attached): is the computed RA/DEC close
    # enough to the observed RA/DEC?
    if spec.observed_ra_deg is not None and spec.observed_dec_deg is not None:
        ra_ok = abs(record.observation.ra_deg - spec.observed_ra_deg) < spec.tolerance_deg
        dec_ok = abs(record.observation.dec_deg - spec.observed_dec_deg) < spec.tolerance_deg
        return ra_ok and dec_ok   # both RA and DEC must be within tolerance

    # No observed RA/DEC configured -> this is a plain "forecast" snapshot:
    # the date match above (condition 1) is the only requirement.
    return True


class OutputManager:
    """Owns every open output file for one scenario run.

    Usage:
        with OutputManager(output_dir, snapshots) as out:
            for record in ...:
                out.write(record)
    All files are flushed and closed automatically on exit, even if the run
    raises partway through -- avoiding the old code's pattern of one
    `.close()` call per writer at the very end of `main()`, which would
    leak file handles on any exception.
    """

    def __init__(self, output_dir: str, snapshots: list[SnapshotSpec], main_filename: str = "trajectory_full.csv",
                 date_tolerance_s: float = 43200.0):
        self.output_dir = output_dir            # folder to write CSV files into
        self.snapshots = list(snapshots)          # the list of SnapshotSpec objects to also export
        self.date_tolerance_s = date_tolerance_s  # how close (in seconds) counts as "the same date"

        os.makedirs(output_dir, exist_ok=True)    # create the output folder if it doesn't exist yet

        self._main_file: Optional[IO[str]] = None       # will hold the open main-CSV file handle
        self._main_writer: Optional[csv.DictWriter] = None  # will hold the csv writer for that file
        self._main_filename = main_filename               # name of the main trajectory CSV

        # snapshot name -> (file handle, DictWriter); created lazily on the
        # first matching row so a snapshot that never actually matches
        # anything does not leave behind an empty, confusing CSV file.
        self._snapshot_writers: dict[str, tuple[IO[str], csv.DictWriter]] = {}

    def __enter__(self) -> "OutputManager":
        # Runs automatically when entering a `with OutputManager(...) as out:` block.
        path = os.path.join(self.output_dir, self._main_filename)   # full path to the main CSV
        self._main_file = open(path, "w", newline="", encoding="utf-8")   # open it for writing
        self._main_writer = csv.DictWriter(self._main_file, fieldnames=OBSERVATION_FIELDS)
        self._main_writer.writeheader()   # write the column-name header row first
        return self   # `out` inside the `with` block refers to this same object

    def __exit__(self, exc_type, exc, tb) -> None:
        # Runs automatically when leaving the `with` block, whether normally
        # or because of an exception -- guarantees files always get closed.
        self.close()

    def write(self, record: ObservationRecord) -> None:
        """Write one row to the main file, and to any snapshot files it
        also matches."""
        row = record.as_csv_row()          # turn the record into a plain dict
        self._main_writer.writerow(row)     # always write it to the main trajectory file

        for spec in self.snapshots:                                     # check every configured snapshot
            if _snapshot_matches(spec, record, self.date_tolerance_s):   # does this row belong there?
                writer = self._get_or_create_snapshot_writer(spec.name)  # get (or make) that file's writer
                writer.writerow(row)                                     # write the same row there too

    def _get_or_create_snapshot_writer(self, name: str) -> csv.DictWriter:
        """Return the DictWriter for snapshot `name`, opening (and writing
        the header for) its file the first time it's actually needed."""
        if name not in self._snapshot_writers:
            path = os.path.join(self.output_dir, f"snapshot_{name}.csv")   # e.g. snapshot_forecast_2012.csv
            f = open(path, "w", newline="", encoding="utf-8")               # open the new file
            writer = csv.DictWriter(f, fieldnames=OBSERVATION_FIELDS)
            writer.writeheader()                                            # write its header row
            self._snapshot_writers[name] = (f, writer)                      # remember it for next time
        return self._snapshot_writers[name][1]   # return just the writer half of the stored pair

    def close(self) -> None:
        """Flush and close every open file handle."""
        if self._main_file is not None:
            self._main_file.close()
            self._main_file = None    # mark as closed so close() is safe to call twice
        for f, _writer in self._snapshot_writers.values():   # close every snapshot file too
            f.close()
        self._snapshot_writers.clear()   # forget them all, now that they're closed
