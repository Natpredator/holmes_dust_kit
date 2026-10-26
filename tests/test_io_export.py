"""
Unit tests for holmes_dust.io_export -- checks that the SnapshotSpec
filtering logic (which replaces the old hand-written per-date `if` blocks
and named PrintWriter variables) behaves like the original.

Run with:  python3 -m unittest discover -s tests
"""

import csv
import datetime as dt
import os
import shutil
import tempfile
import unittest

import numpy as np

from holmes_dust.config import SnapshotSpec
from holmes_dust.io_export import ObservationRecord, OutputManager
from holmes_dust.observations import Observation


def _make_record(date_utc: dt.datetime, ra: float, dec: float, particle_id: int = 0) -> ObservationRecord:
    obs = Observation(
        ra_deg=ra,
        dec_deg=dec,
        phase_angle_deg=45.0,
        light_time_s=123.0,
        geocentric_position_m=np.array([1.0e11, 2.0e11, 3.0e10]),
    )
    return ObservationRecord(
        date_utc=date_utc,
        particle_id=particle_id,
        bin_name="b1_small",
        radius_mm=0.005,
        beta=0.02,
        ejection_speed_m_s=300.0,
        observation=obs,
    )


class TestOutputManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _read_csv(self, filename):
        path = os.path.join(self.tmp_dir, filename)
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_main_file_gets_every_row(self):
        target_date = dt.datetime(2026, 8, 1, 12, 0, 0)
        with OutputManager(self.tmp_dir, snapshots=[]) as out:
            out.write(_make_record(target_date, ra=10.0, dec=20.0))
            out.write(_make_record(target_date + dt.timedelta(days=1), ra=11.0, dec=21.0))

        rows = self._read_csv("trajectory_full.csv")
        self.assertEqual(len(rows), 2)

    def test_forecast_snapshot_matches_by_date_only(self):
        target_date = dt.datetime(2026, 8, 1, 11, 59, 23)
        spec = SnapshotSpec(name="forecast_2026_08_01", date_utc=target_date)

        with OutputManager(self.tmp_dir, snapshots=[spec]) as out:
            # Same day (within the default half-day tolerance) -> should match.
            out.write(_make_record(target_date, ra=999.0, dec=-999.0))
            # A day away -> should NOT match.
            out.write(_make_record(target_date + dt.timedelta(days=2), ra=1.0, dec=1.0))

        rows = self._read_csv("snapshot_forecast_2026_08_01.csv")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["ra_deg"]), 999.0)

    def test_matching_snapshot_filters_by_ra_dec_window(self):
        target_date = dt.datetime(2012, 11, 1, 12, 0, 0)
        spec = SnapshotSpec(
            name="obs_match", date_utc=target_date,
            observed_ra_deg=242.0, observed_dec_deg=-40.4, tolerance_deg=40.0,
        )

        with OutputManager(self.tmp_dir, snapshots=[spec]) as out:
            out.write(_make_record(target_date, ra=250.0, dec=-35.0, particle_id=1))   # inside window
            out.write(_make_record(target_date, ra=350.0, dec=10.0, particle_id=2))    # outside window
            out.write(_make_record(target_date + dt.timedelta(days=10), ra=242.0, dec=-40.4, particle_id=3))  # right date, wrong RA/DEC window because date mismatch

        rows = self._read_csv("snapshot_obs_match.csv")
        ids = {int(r["particle_id"]) for r in rows}
        self.assertEqual(ids, {1})

    def test_no_file_created_for_snapshot_with_no_matches(self):
        target_date = dt.datetime(2050, 1, 1, 12, 0, 0)
        spec = SnapshotSpec(name="never_matches", date_utc=target_date)

        with OutputManager(self.tmp_dir, snapshots=[spec]) as out:
            out.write(_make_record(target_date + dt.timedelta(days=365), ra=1.0, dec=1.0))

        path = os.path.join(self.tmp_dir, "snapshot_never_matches.csv")
        self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
