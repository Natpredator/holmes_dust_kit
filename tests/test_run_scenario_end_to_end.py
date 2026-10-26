"""
End-to-end smoke test for run_scenario.py: runs a tiny scenario (few
particles, a handful of days) through the FULL pipeline -- particle
generation, simulation, observation, CSV export -- using the fake
rebound/reboundx modules and an in-memory fake ephemeris provider, so it
needs no network access and no real astrodynamics packages installed.

This does NOT validate physics or the real REBOUND/REBOUNDx API -- see the
fakes' own docstrings. It validates that run_scenario.run() wires all the
real holmes_dust modules (config, scenarios, particles, simulation,
observations, io_export) together without crashing, and that the expected
output files land on disk with the expected shape.

Run with:  python3 -m unittest discover -s tests
"""

import csv
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
from holmes_dust.config import (  # noqa: E402
    ALL_PLANETS,
    Body,
    EjectionDirectionMode,
    EphemerisSource,
    IntegrationConfig,
    PARTICLE_BINS,
    ParticleGenerationConfig,
    RadiationModel,
    ScenarioConfig,
    SnapshotSpec,
)
from holmes_dust.ephemeris import StateVector  # noqa: E402
from holmes_dust import constants  # noqa: E402


class _FakeEphemeris:
    """Same idea as tests/test_simulation_with_fakes.py's fake -- deterministic,
    offline, well-formed numbers; not astronomically meaningful."""

    def get_state(self, body: Body, epoch_utc: dt.datetime, comet_designation=None) -> StateVector:
        au = 1.495978707e11
        offsets = {
            Body.SUN: 0.0, Body.MERCURY: 0.4, Body.VENUS: 0.7,
            Body.EARTH_MOON_BARYCENTER: 1.0, Body.EARTH: 1.0, Body.MARS: 1.5,
            Body.JUPITER: 5.2, Body.SATURN: 9.5, Body.URANUS: 19.2,
            Body.NEPTUNE: 30.1, Body.COMET: 3.4,
        }
        return StateVector(
            epoch_utc=epoch_utc,
            position_m=np.array([offsets[body] * au, 0.0, 0.0]),
            velocity_m_s=np.array([0.0, 1.0e4, 0.0]),
        )

    def get_state_series(self, body, epochs_utc, comet_designation=None):
        return [self.get_state(body, e, comet_designation) for e in epochs_utc]


class TestRunScenarioEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _tiny_scenario(self, radiation_model: RadiationModel, snapshots=()) -> ScenarioConfig:
        return ScenarioConfig(
            name="tiny-e2e-test",
            comet_designation="17P",
            outburst_epoch_utc=dt.datetime(2007, 10, 24, 12, 0, 0),
            ephemeris_source=EphemerisSource.JPL_HORIZONS,  # unused: we inject a fake provider directly
            gravitating_planets=ALL_PLANETS,
            include_comet_self_gravity=False,
            comet_assumed_gm_m3_s2=None,
            radiation_model=radiation_model,
            particles=ParticleGenerationConfig(
                bins=[PARTICLE_BINS["b1_small"]],
                n_particles_per_bin=2,
                direction_mode=EjectionDirectionMode.ISOTROPIC_SPHERE,
                random_seed=42,
            ),
            integration=IntegrationConfig(duration_years=2.0 / 365.25, output_cadence_s=86400.0),
            snapshots=snapshots,
            output_dir=self.tmp_dir,
        )

    def test_batched_pipeline_produces_expected_rows(self):
        scenario = self._tiny_scenario(RadiationModel.REBOUNDX_BETA)
        run_scenario.run(scenario, ephemeris=_FakeEphemeris())

        main_csv = os.path.join(self.tmp_dir, "trajectory_full.csv")
        self.assertTrue(os.path.exists(main_csv))

        with open(main_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        # duration_years=2 days at 1-day cadence -> 3 output epochs (t=0,1,2 days)
        # times 2 particles (1 bin x 2 particles per bin) = 6 rows.
        self.assertEqual(len(rows), 3 * 2)
        # Every row must at least have a well-formed RA in [0, 360).
        for row in rows:
            ra = float(row["ra_deg"])
            self.assertGreaterEqual(ra, 0.0)
            self.assertLess(ra, 360.0)

    def test_legacy_pipeline_produces_expected_rows(self):
        scenario = self._tiny_scenario(RadiationModel.LEGACY_SUN_GM_REDUCTION)
        run_scenario.run(scenario, ephemeris=_FakeEphemeris())

        main_csv = os.path.join(self.tmp_dir, "trajectory_full.csv")
        with open(main_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3 * 2)

    def test_snapshot_file_is_created_when_it_matches(self):
        target_date = scenario_date = dt.datetime(2007, 10, 25, 12, 0, 0)  # t = +1 day
        spec = SnapshotSpec(name="plus_one_day", date_utc=target_date)
        scenario = self._tiny_scenario(RadiationModel.REBOUNDX_BETA, snapshots=[spec])

        run_scenario.run(scenario, ephemeris=_FakeEphemeris())

        snapshot_csv = os.path.join(self.tmp_dir, "snapshot_plus_one_day.csv")
        self.assertTrue(os.path.exists(snapshot_csv))
        with open(snapshot_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # Only the +1 day epoch should match, for both particles.
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
