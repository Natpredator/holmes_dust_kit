"""
Tests for holmes_dust.legacy_java: reading, writing and rebuilding the old
Java/Orekit output layout (b<bin>p<N>.csv, b<bin>f<NN>.csv, b<bin>m<N>.csv).

tests/data/java_sample/ holds the first lines of three REAL Java output
files (1892-11-04 run), including a deliberately truncated last line, as
produced when the Java program was stopped mid-write.

Run with:  python3 -m unittest discover -s tests
"""

import datetime as dt
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

from tests.fakes import fake_rebound, fake_reboundx

sys.modules.setdefault("rebound", fake_rebound)
sys.modules.setdefault("reboundx", fake_reboundx)

import run_scenario  # noqa: E402
from holmes_dust import scenarios  # noqa: E402
from holmes_dust.config import Body, OutputFormat, RadiationModel  # noqa: E402
from holmes_dust.ephemeris import LEGACY_STATES, LegacyCometHybridProvider, StateVector  # noqa: E402
from holmes_dust.legacy_java import (  # noqa: E402
    JAVA_1892_SNAPSHOTS,
    LegacySnapshot,
    format_legacy_fields,
    iter_legacy_rows,
    java_deg_min_sec,
    java_double_str,
    java_hmin_sec,
    julian_day_number,
    list_legacy_files,
    particles_from_java_run,
    rebuild_snapshot_files,
)
from holmes_dust.particles import beta_from_radius_mm, ejection_speed_from_radius_mm  # noqa: E402

SAMPLE = os.path.join(os.path.dirname(__file__), "data", "java_sample")


class TestJavaFormatting(unittest.TestCase):
    def test_every_field_of_real_java_lines_is_reproduced_byte_for_byte(self):
        for _, _, _, path in list_legacy_files(SAMPLE):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parts = [p.strip() for p in line.rstrip("\n").split(",")]
                    if len(parts) != 15:
                        continue                                   # the truncated fixture line
                    for i in (3, 4, 5, 8, 9, 10, 11, 12, 14):      # all floating-point fields
                        self.assertEqual(java_double_str(float(parts[i])), parts[i])
                    self.assertEqual(java_hmin_sec(float(parts[8])), parts[6])
                    self.assertEqual(java_deg_min_sec(float(parts[9])), parts[7])
                    self.assertEqual(julian_day_number(dt.date.fromisoformat(parts[2][:10])), int(parts[0]))

    def test_java_double_toString_edge_cases(self):
        self.assertEqual(java_double_str(1e-5), "1.0E-5")
        self.assertEqual(java_double_str(9.99e-4), "9.99E-4")
        self.assertEqual(java_double_str(0.001), "0.001")
        self.assertEqual(java_double_str(123.0), "123.0")
        self.assertEqual(java_double_str(-2.5e7), "-2.5E7")
        self.assertEqual(java_double_str(0.0), "0.0")

    def test_julian_day_number(self):
        self.assertEqual(julian_day_number(dt.date(2000, 1, 1)), 2451545)
        self.assertEqual(julian_day_number(dt.datetime(1892, 11, 4, 12)), 2412407)
        self.assertEqual(julian_day_number(dt.date(2013, 2, 17)), 2456341)   # the Java m2 date

    def test_format_line_layout(self):
        line = format_legacy_fields(
            epoch_utc=dt.datetime(1892, 11, 4, 12), barycentric_position_m=np.array([1.0, 2.0, 3.0]) / 6.68458712e-12,
            ra_deg=13.78785386341796, dec_deg=39.382600285534025, radius_mm=0.007245399730486762,
            theta=1.7858415276095405, u=0.35038720354537745, epoch_index=9, phase_angle_deg=13.3, step=1)
        self.assertTrue(line.startswith("2412407, step  1, 1892-11-04T12:00:00.000Z, "))
        self.assertIn("00H 55min  9.085sec, 39° 22' 57.361''", line)
        self.assertTrue(line.endswith(", 9, 13.3 \n"))


class TestReadingAndRebuilding(unittest.TestCase):
    def test_truncated_last_line_is_skipped(self):
        rows = list(iter_legacy_rows(os.path.join(SAMPLE, "b1p10.csv")))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["jd"], 2412407)
        self.assertEqual(rows[1]["step"], 2)

    def test_particles_are_rebuilt_exactly(self):
        state = LEGACY_STATES["1892-11-04"]
        pos, vel = np.array(state["comet_pos"]), np.array(state["comet_vel"])
        parts = particles_from_java_run(SAMPLE, pos, vel)
        self.assertEqual([p.legacy_number for p in parts], [1, 2, 10])   # numeric order, not text order
        first = next(iter_legacy_rows(os.path.join(SAMPLE, "b1p1.csv")))
        p = parts[0]
        self.assertEqual(p.radius_mm, first["radius_mm"])
        self.assertEqual((p.theta, p.u), (first["theta"], first["u"]))
        self.assertAlmostEqual(p.beta, beta_from_radius_mm(first["radius_mm"]))
        dv = p.initial_velocity_m_s - vel
        self.assertAlmostEqual(np.linalg.norm(dv), ejection_speed_from_radius_mm(first["radius_mm"]), places=9)
        self.assertAlmostEqual(dv[2] / np.linalg.norm(dv), first["u"], places=12)   # z component = u
        # The start position printed by Java (row 0, AU) equals the legacy comet state.
        np.testing.assert_allclose(pos * 6.68458712e-12, [first["x_au"], first["y_au"], first["z_au"]], rtol=1e-15)

    def test_rebuild_snapshot_files(self):
        tmp = tempfile.mkdtemp()
        try:
            snap = LegacySnapshot("f99", 2412408)                              # 1892-11-05: row 2 of each file
            box = LegacySnapshot("m99", 2412408, ra_center_deg=13.6, dec_center_deg=39.3, tolerance_deg=0.001)
            counts = rebuild_snapshot_files(SAMPLE, tmp, (snap, box))
            self.assertEqual(counts["b1f99.csv"], 3)                          # all three particles
            self.assertLessEqual(counts["b1m99.csv"], 3)
            with open(os.path.join(tmp, "b1f99.csv"), encoding="utf-8") as f:
                line = f.readline()
            self.assertTrue(line.startswith("2412408, 1892-11-05T12:00:00.000Z, "))   # "step" dropped
            self.assertEqual(len(line.split(",")), 14)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class _FakeEphemeris:
    def get_state(self, body, epoch_utc, comet_designation=None):
        au = 1.495978707e11
        x = {Body.SUN: 0.0, Body.EARTH: 1.0, Body.COMET: 3.4}.get(body, 5.0)
        return StateVector(epoch_utc, np.array([x * au, 0.1 * au, 0.2 * au]), np.array([0.0, 1.0e4, 0.0]))

    def get_state_series(self, body, epochs_utc, comet_designation=None):
        return [self.get_state(body, e, comet_designation) for e in epochs_utc]


class TestLegacyOutputFromRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_1892_profile_matches_java_length_and_snapshots(self):
        sc = scenarios.build_scenario("1892-11-04", scenarios.PROFILE_1892_NO_COMET_GRAVITY)
        self.assertEqual(sc.duration_seconds(), 4332096000.0)            # Java `duration`
        epochs = run_scenario._build_output_epochs(sc)
        self.assertEqual(len(epochs), 50141)                              # rows per Java p-file
        self.assertEqual(epochs[-1], dt.datetime(2030, 2, 14, 12))        # Java last row: 2030-02-14
        self.assertEqual(sc.legacy_epoch_index, 9)
        self.assertEqual([s.name for s in sc.snapshots], [s.stem for s in JAVA_1892_SNAPSHOTS])

    def test_run_writes_both_layouts(self):
        # 1892 epoch, 3 particles, 2 days, both output layouts, with an
        # extra forecast snapshot on day 1 to check the f-file writer.
        extra = scenarios.snapshot_specs_from_legacy([LegacySnapshot("f01", 2412408)])
        sc = scenarios.build_scenario(
            "1892-11-04", scenarios.PROFILE_1892_NO_COMET_GRAVITY, n_particles_per_bin=3, duration_days=2,
            random_seed=1, output_dir=self.tmp, output_format=OutputFormat.BOTH,
            radiation_model=RadiationModel.LEGACY_SUN_GM_REDUCTION, apply_light_time_correction=False,
            snapshots=extra,
        )
        run_scenario.run(sc, ephemeris=_FakeEphemeris())
        files = sorted(os.listdir(self.tmp))
        for name in ("b1p1.csv", "b1p2.csv", "b1p3.csv", "b1f01.csv", "trajectory_full.csv", "snapshot_f01.csv"):
            self.assertIn(name, files)
        rows = list(iter_legacy_rows(os.path.join(self.tmp, "b1p2.csv")))
        self.assertEqual([r["step"] for r in rows], [1, 2, 3])            # clean step counter
        self.assertEqual([r["jd"] for r in rows], [2412407, 2412408, 2412409])
        self.assertEqual(rows[0]["epoch_index"], 9)
        self.assertTrue(all(r["date"].endswith("T12:00:00.000Z") for r in rows))
        f_rows = list(iter_legacy_rows(os.path.join(self.tmp, "b1f01.csv")))
        self.assertEqual(len(f_rows), 3)                                  # one row per particle
        # Barycentric start position in the p-file == comet start position.
        comet = _FakeEphemeris().get_state(Body.COMET, sc.outburst_epoch_utc)
        np.testing.assert_allclose([rows[0]["x_au"], rows[0]["y_au"], rows[0]["z_au"]],
                                   comet.position_m * 6.68458712e-12, rtol=1e-12)


class TestHybridProvider(unittest.TestCase):
    def test_comet_from_java_table_everything_else_from_base(self):
        hp = LegacyCometHybridProvider(base=_FakeEphemeris())
        t0 = dt.datetime(1892, 11, 4, 12)
        comet = hp.get_state(Body.COMET, t0, "17P")
        np.testing.assert_array_equal(comet.position_m, LEGACY_STATES["1892-11-04"]["comet_pos"])
        sun = hp.get_state(Body.SUN, t0)
        self.assertEqual(sun.velocity_m_s[1], 1.0e4)                      # came from the base provider
        later = hp.get_state(Body.COMET, dt.datetime(1893, 1, 1, 12), "17P")
        self.assertAlmostEqual(later.position_m[0], 3.4 * 1.495978707e11)


if __name__ == "__main__":
    unittest.main()


class TestPlanObservationsHelpers(unittest.TestCase):
    """The data-handling half of tools/plan_observations.py (the Skyfield
    half was checked against the project's original scripts: identical
    output, byte for byte, for holmes_1892_2027_2_filament1_kouvola.py)."""

    def test_load_select_and_track(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
        import plan_observations as po
        pos = po.load_positions(SAMPLE, 2412407, 2412408)
        self.assertEqual(sorted(pos), [1, 2, 10])
        first = next(iter_legacy_rows(os.path.join(SAMPLE, "b1p1.csv")))
        start = [first["x_au"], first["y_au"], first["z_au"]]
        self.assertEqual(po.select_near(pos, 2412407, start, 1e-9), [1, 2, 10])   # all start at the comet
        self.assertEqual(po.select_near(pos, 2412408, start, 1e-9), [])           # ... and separate after a day
        jd, xyz = po.mean_track(pos, [1, 2])
        self.assertEqual(list(jd), [2412407.0, 2412408.0])
        np.testing.assert_allclose(xyz[0], start)


class TestSpkProvider(unittest.TestCase):
    """Offline planetary ephemeris. Needs `pip install jplephem` and a JPL
    kernel; set HOLMES_TEST_SPK=/path/to/de440.bsp (or de405/de421)."""

    def test_earth_sun_distance(self):
        path = os.environ.get("HOLMES_TEST_SPK")
        try:
            import jplephem  # noqa: F401
        except ImportError:
            self.skipTest("jplephem not installed")
        if not path or not os.path.exists(path):
            self.skipTest("HOLMES_TEST_SPK not set")
        from holmes_dust.ephemeris import SpkEphemerisProvider
        e = SpkEphemerisProvider(path)
        t = dt.datetime(2000, 1, 1, 12)
        d = np.linalg.norm(e.get_state(Body.EARTH, t).position_m - e.get_state(Body.SUN, t).position_m)
        self.assertAlmostEqual(d / 1.495978707e11, 0.9833, places=3)             # perihelion season
        series = e.get_state_series(Body.EARTH, [t, t + dt.timedelta(days=1)])
        self.assertEqual(len(series), 2)
        comet = e.get_state(Body.COMET, dt.datetime(1892, 11, 4, 12), "17P")      # from the Java table
        np.testing.assert_array_equal(comet.position_m, LEGACY_STATES["1892-11-04"]["comet_pos"])


class TestRebuildFromForecastFiles(unittest.TestCase):
    """Particles whose p-file was deleted are rebuilt from line k of a
    forecast file (line k = particle k in the Java program)."""

    def test_missing_p_file_is_filled_from_forecast_lines(self):
        tmp = tempfile.mkdtemp()
        try:
            shutil.copy(os.path.join(SAMPLE, "b1p2.csv"), tmp)            # only particle 2 kept
            # forecast file: line 1 = particle 1, line 2 = particle 2 (copied from their p-files)
            rows = []
            for name in ("b1p1.csv", "b1p2.csv"):
                with open(os.path.join(SAMPLE, name), encoding="utf-8") as f:
                    parts = f.readline().split(",")
                rows.append(",".join([parts[0]] + parts[2:]))
            with open(os.path.join(tmp, "b1f99.csv"), "w", encoding="utf-8") as f:
                f.writelines(rows)
            state = LEGACY_STATES["1892-11-04"]
            parts = particles_from_java_run(tmp, np.array(state["comet_pos"]), np.array(state["comet_vel"]))
            self.assertEqual([p.legacy_number for p in parts], [1, 2])
            p1 = next(iter_legacy_rows(os.path.join(SAMPLE, "b1p1.csv")))
            self.assertEqual((parts[0].radius_mm, parts[0].theta, parts[0].u), (p1["radius_mm"], p1["theta"], p1["u"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
