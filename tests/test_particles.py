"""
Unit tests for holmes_dust.particles.

These tests only exercise plain-Python/numpy code -- no rebound/reboundx
dependency -- so they can (and were) actually run in an environment without
those packages installed, to catch basic logic errors early.

Run with:  python3 -m unittest discover -s tests
"""

import math
import unittest

import numpy as np

from holmes_dust.config import EjectionDirectionMode, PARTICLE_BINS
from holmes_dust.particles import (
    beta_from_radius_mm,
    ejection_speed_from_radius_mm,
    generate_particles,
    sample_isotropic_direction,
    sample_particle_radius_mm,
    sample_toward_sun_direction,
)


class TestRadiusSampling(unittest.TestCase):
    def test_within_bounds_many_draws(self):
        rng = np.random.default_rng(0)
        size_bin = PARTICLE_BINS["b1_small"]
        for _ in range(1000):
            r = sample_particle_radius_mm(rng, size_bin)
            self.assertGreaterEqual(r, size_bin.radius_min_mm)
            self.assertLess(r, size_bin.radius_max_mm)


class TestBetaAndEjectionSpeedFormulas(unittest.TestCase):
    def test_beta_matches_hand_computation(self):
        # Direct port check against: Ybeta = 0.0002*Math.pow(radius, -1.0438)
        radius_mm = 0.005
        expected = 0.0002 * radius_mm ** (-1.0438)
        self.assertAlmostEqual(beta_from_radius_mm(radius_mm), expected, places=12)

    def test_beta_decreases_with_radius(self):
        # Bigger grains should be LESS affected by radiation pressure.
        self.assertGreater(beta_from_radius_mm(0.001), beta_from_radius_mm(0.5))

    def test_ejection_speed_matches_hand_computation(self):
        # Direct port check against: Yej = 600.99*Math.exp(-0.6086*radius)
        radius_mm = 0.02
        expected = 600.99 * math.exp(-0.6086 * radius_mm)
        self.assertAlmostEqual(ejection_speed_from_radius_mm(radius_mm), expected, places=6)

    def test_ejection_speed_decreases_with_radius(self):
        self.assertGreater(ejection_speed_from_radius_mm(0.001), ejection_speed_from_radius_mm(1.0))


class TestDirectionSampling(unittest.TestCase):
    def test_isotropic_direction_is_unit_vector(self):
        rng = np.random.default_rng(1)
        for _ in range(1000):
            d = sample_isotropic_direction(rng)
            self.assertAlmostEqual(np.linalg.norm(d), 1.0, places=9)

    def test_isotropic_direction_covers_full_sphere_on_average(self):
        # Weak statistical sanity check: mean of many isotropic samples
        # should be close to the origin (no preferred direction).
        rng = np.random.default_rng(2)
        samples = np.array([sample_isotropic_direction(rng) for _ in range(20000)])
        mean = samples.mean(axis=0)
        self.assertLess(np.linalg.norm(mean), 0.02)

    def test_toward_sun_direction_points_correctly(self):
        comet_pos = np.array([1.0, 0.0, 0.0])
        sun_pos = np.array([0.0, 0.0, 0.0])
        d = sample_toward_sun_direction(comet_pos, sun_pos)
        np.testing.assert_allclose(d, np.array([-1.0, 0.0, 0.0]))

    def test_toward_sun_direction_raises_on_coincident_positions(self):
        pos = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            sample_toward_sun_direction(pos, pos.copy())


class TestGenerateParticles(unittest.TestCase):
    def test_particle_count_and_ids(self):
        rng = np.random.default_rng(3)
        comet_pos = np.array([1.0e11, 0.0, 0.0])
        comet_vel = np.array([0.0, 3.0e4, 0.0])
        particles = generate_particles(
            bins=[PARTICLE_BINS["b1_small"], PARTICLE_BINS["b2_medium"]],
            n_particles_per_bin=10,
            direction_mode=EjectionDirectionMode.ISOTROPIC_SPHERE,
            comet_position_m=comet_pos,
            comet_velocity_m_s=comet_vel,
            rng=rng,
        )
        self.assertEqual(len(particles), 20)
        # ids should be unique and contiguous
        self.assertEqual([p.particle_id for p in particles], list(range(20)))
        # first 10 come from bin 1, next 10 from bin 2
        self.assertTrue(all(p.bin_name == "b1_small" for p in particles[:10]))
        self.assertTrue(all(p.bin_name == "b2_medium" for p in particles[10:]))

    def test_initial_velocity_is_comet_velocity_plus_ejection(self):
        rng = np.random.default_rng(4)
        comet_pos = np.array([1.0e11, 0.0, 0.0])
        comet_vel = np.array([0.0, 3.0e4, 0.0])
        particles = generate_particles(
            bins=[PARTICLE_BINS["b1_small"]],
            n_particles_per_bin=5,
            direction_mode=EjectionDirectionMode.ISOTROPIC_SPHERE,
            comet_position_m=comet_pos,
            comet_velocity_m_s=comet_vel,
            rng=rng,
        )
        for p in particles:
            expected_vel = comet_vel + p.ejection_speed_m_s * p.direction_unit_vector
            np.testing.assert_allclose(p.initial_velocity_m_s, expected_vel)
            np.testing.assert_allclose(p.initial_position_m, comet_pos)

    def test_toward_sun_mode_requires_sun_position(self):
        rng = np.random.default_rng(5)
        with self.assertRaises(ValueError):
            generate_particles(
                bins=[PARTICLE_BINS["b1_small"]],
                n_particles_per_bin=1,
                direction_mode=EjectionDirectionMode.TOWARD_SUN,
                comet_position_m=np.array([1.0, 0.0, 0.0]),
                comet_velocity_m_s=np.array([0.0, 0.0, 0.0]),
                sun_position_m=None,
                rng=rng,
            )


if __name__ == "__main__":
    unittest.main()
