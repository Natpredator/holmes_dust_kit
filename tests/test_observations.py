"""
Unit tests for holmes_dust.observations. Pure numpy -- no rebound needed.

Run with:  python3 -m unittest discover -s tests
"""

import unittest

import numpy as np

from holmes_dust.observations import (
    SPEED_OF_LIGHT_M_S,
    apply_light_time_correction,
    compute_observation,
    phase_angle_deg,
    ra_dec_from_geocentric_vector,
    to_dms,
    to_hms,
)


class TestRaDec(unittest.TestCase):
    def test_along_x_axis_is_ra0_dec0(self):
        ra, dec = ra_dec_from_geocentric_vector(np.array([1.0, 0.0, 0.0]))
        self.assertAlmostEqual(ra, 0.0, places=9)
        self.assertAlmostEqual(dec, 0.0, places=9)

    def test_along_y_axis_is_ra90(self):
        ra, dec = ra_dec_from_geocentric_vector(np.array([0.0, 1.0, 0.0]))
        self.assertAlmostEqual(ra, 90.0, places=9)
        self.assertAlmostEqual(dec, 0.0, places=9)

    def test_along_negative_x_axis_is_ra180(self):
        ra, dec = ra_dec_from_geocentric_vector(np.array([-1.0, 0.0, 0.0]))
        self.assertAlmostEqual(ra, 180.0, places=9)

    def test_along_negative_y_axis_wraps_to_270(self):
        # RA must be normalised into [0, 360), matching the old code's
        # normalizeAngle(...) + toDegrees(...) behaviour.
        ra, dec = ra_dec_from_geocentric_vector(np.array([0.0, -1.0, 0.0]))
        self.assertAlmostEqual(ra, 270.0, places=9)

    def test_along_z_axis_is_dec90(self):
        ra, dec = ra_dec_from_geocentric_vector(np.array([0.0, 0.0, 1.0]))
        self.assertAlmostEqual(dec, 90.0, places=9)


class TestPhaseAngle(unittest.TestCase):
    def test_target_between_sun_and_earth_is_180(self):
        # Sun ---- target ---- Earth, all colinear -> phase angle 180 deg.
        sun = np.array([-1.0, 0.0, 0.0])
        target = np.array([0.0, 0.0, 0.0])
        earth = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(phase_angle_deg(sun, target, earth), 180.0, places=6)

    def test_sun_and_earth_same_side_is_0(self):
        # Sun and Earth both far away in the SAME direction from target
        # -> phase angle 0 deg (full "opposition"-like geometry).
        sun = np.array([10.0, 0.0, 0.0])
        target = np.array([0.0, 0.0, 0.0])
        earth = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(phase_angle_deg(sun, target, earth), 0.0, places=6)

    def test_right_angle_geometry_is_90(self):
        sun = np.array([1.0, 0.0, 0.0])
        target = np.array([0.0, 0.0, 0.0])
        earth = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(phase_angle_deg(sun, target, earth), 90.0, places=6)


class TestLightTimeCorrection(unittest.TestCase):
    def test_stationary_target_position_unchanged(self):
        # Zero velocity -> position correction is exactly zero, only the
        # light-time itself is non-trivial.
        pos = np.array([1.5e11, 0.0, 0.0])
        vel = np.zeros(3)
        earth = np.array([1.496e11, 0.0, 0.0])
        corrected, tau = apply_light_time_correction(pos, vel, earth)
        np.testing.assert_allclose(corrected, pos)
        expected_tau = np.linalg.norm(pos - earth) / SPEED_OF_LIGHT_M_S
        self.assertAlmostEqual(tau, expected_tau, places=6)

    def test_moving_target_shifts_backward_along_velocity(self):
        pos = np.array([1.0e11, 0.0, 0.0])
        vel = np.array([0.0, 1.0e4, 0.0])  # 10 km/s, roughly comet-like
        earth = np.array([0.0, 0.0, 0.0])
        corrected, tau = apply_light_time_correction(pos, vel, earth)

        # tau should be roughly distance / c, and the correction should have
        # moved the apparent position backward along -vel by vel * tau.
        approx_tau = np.linalg.norm(pos - earth) / SPEED_OF_LIGHT_M_S
        self.assertAlmostEqual(tau, approx_tau, places=3)
        expected_shift = -vel * tau
        np.testing.assert_allclose(corrected - pos, expected_shift, rtol=1e-3)

    def test_light_time_is_small_fraction_of_a_day_at_au_scale(self):
        # Sanity check on scale: light-time across ~1 AU should be a few
        # hundred seconds, i.e. utterly negligible next to the 1-day output
        # cadence used throughout this project.
        pos = np.array([1.496e11, 0.0, 0.0])
        earth = np.zeros(3)
        _, tau = apply_light_time_correction(pos, np.zeros(3), earth)
        self.assertGreater(tau, 400.0)
        self.assertLess(tau, 600.0)


class TestComputeObservation(unittest.TestCase):
    def test_disabling_light_time_correction_gives_zero_light_time(self):
        obs = compute_observation(
            target_position_m=np.array([1.5e11, 0.0, 0.0]),
            target_velocity_m_s=np.array([0.0, 1.0e4, 0.0]),
            sun_position_m=np.zeros(3),
            earth_position_m=np.array([1.496e11, 0.0, 0.0]),
            apply_light_time=False,
        )
        self.assertEqual(obs.light_time_s, 0.0)

    def test_enabling_light_time_correction_gives_positive_light_time(self):
        obs = compute_observation(
            target_position_m=np.array([1.5e11, 0.0, 0.0]),
            target_velocity_m_s=np.array([0.0, 1.0e4, 0.0]),
            sun_position_m=np.zeros(3),
            earth_position_m=np.array([1.496e11, 0.0, 0.0]),
            apply_light_time=True,
        )
        self.assertGreater(obs.light_time_s, 0.0)


class TestFormatting(unittest.TestCase):
    def test_to_hms_zero(self):
        self.assertEqual(to_hms(0.0), "00H 00min 00.000sec")

    def test_to_dms_zero(self):
        self.assertEqual(to_dms(0.0), "0\u00b0 00' 00.000''")

    def test_to_hms_ninety_degrees_is_six_hours(self):
        self.assertTrue(to_hms(90.0).startswith("06H"))


if __name__ == "__main__":
    unittest.main()
