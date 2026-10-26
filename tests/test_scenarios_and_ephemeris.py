"""
Unit tests for holmes_dust.scenarios and the legacy-table branch of
holmes_dust.ephemeris. The live-network JPL-Horizons request itself still
needs network + astroquery and is not exercised here; its offline-testable
RANGE-QUERY LOGIC (step formatting, fallback behaviour) is covered
separately in test_horizons_range_query.py, against a fake Horizons class.

Run with:  python3 -m unittest discover -s tests
"""

import datetime as dt
import unittest

import numpy as np

from holmes_dust import scenarios
from holmes_dust.config import Body, EphemerisSource, RadiationModel
from holmes_dust.ephemeris import LegacyTableEphemerisProvider, make_provider


class TestBuildScenario(unittest.TestCase):
    def test_1892_scenario_has_correct_epoch_and_comet(self):
        scenario = scenarios.build_scenario("1892-11-04", scenarios.PROFILE_1892_NO_COMET_GRAVITY)
        self.assertEqual(scenario.outburst_epoch_utc, dt.datetime(1892, 11, 4, 12, 0, 0))
        self.assertEqual(scenario.comet_designation, "17P")
        self.assertFalse(scenario.include_comet_self_gravity)
        self.assertIsNone(scenario.comet_assumed_gm_m3_s2)

    def test_tempel2_epochs_are_tagged_10p_not_17p(self):
        # Regression test for the correction made after the first
        # (analysis-only) pass: indices 13/14/15 are 10P/Tempel2, not Holmes.
        for key in ("1873-06-25", "1894-04-23", "1904-11-10"):
            scenario = scenarios.build_scenario(key, scenarios.PROFILE_HISTORICAL_CANDIDATE_DEFAULT)
            self.assertEqual(scenario.comet_designation, "10P", msg=f"epoch {key} should be tagged 10P")

    def test_unknown_epoch_key_raises(self):
        with self.assertRaises(KeyError):
            scenarios.build_scenario("not-a-real-date", scenarios.PROFILE_1892_NO_COMET_GRAVITY)

    def test_overrides_apply(self):
        scenario = scenarios.build_scenario(
            "2007-10-24", scenarios.PROFILE_2007_MAJOR_PLANETS_ONLY,
            n_particles_per_bin=7, duration_years=1.0,
        )
        self.assertEqual(scenario.particles.n_particles_per_bin, 7)
        self.assertEqual(scenario.integration.duration_years, 1.0)

    def test_duration_seconds_conversion(self):
        scenario = scenarios.build_scenario(
            "2007-10-24", scenarios.PROFILE_2007_MAJOR_PLANETS_ONLY, duration_years=1.0,
        )
        self.assertAlmostEqual(scenario.duration_seconds(), 365.25 * 86400.0)

    def test_all_profiles_are_internally_consistent(self):
        # include_comet_self_gravity=True must always come with a GM value,
        # and False must always come with None -- catches a copy/paste
        # mistake in one of the PROFILE_* dicts.
        for name, profile in scenarios.PROFILES.items():
            with self.subTest(profile=name):
                if profile["include_comet_self_gravity"]:
                    self.assertIsNotNone(profile["comet_assumed_gm_m3_s2"])
                else:
                    self.assertIsNone(profile["comet_assumed_gm_m3_s2"])


class TestLegacyEphemerisTable(unittest.TestCase):
    def test_comet_state_matches_java_1892_values(self):
        provider = LegacyTableEphemerisProvider()
        state = provider.get_state(Body.COMET, dt.datetime(1892, 11, 4, 12, 0, 0))
        np.testing.assert_allclose(
            state.position_m,
            [2.744228527724833e11, 1.354994421045742e11, 1.829999385978398e11],
        )
        np.testing.assert_allclose(
            state.velocity_m_s,
            [-9.275429660245276e3, 1.601729393482798e4, 1.248215654577809e4],
        )

    def test_unknown_epoch_raises_keyerror(self):
        provider = LegacyTableEphemerisProvider()
        with self.assertRaises(KeyError):
            provider.get_state(Body.COMET, dt.datetime(2000, 1, 1, 12, 0, 0))

    def test_planet_lookup_raises_not_implemented(self):
        provider = LegacyTableEphemerisProvider()
        with self.assertRaises(NotImplementedError):
            provider.get_state(Body.JUPITER, dt.datetime(1892, 11, 4, 12, 0, 0))

    def test_make_provider_factory(self):
        provider = make_provider(EphemerisSource.LEGACY_TABLE)
        self.assertIsInstance(provider, LegacyTableEphemerisProvider)


if __name__ == "__main__":
    unittest.main()
