"""
Smoke tests for holmes_dust.simulation, run against FAKE rebound/reboundx
modules (see tests/fakes/) instead of the real packages, which could not be
installed in the environment this project was developed in (no network
access). These tests catch Python-level orchestration bugs (wrong attribute
names, incorrect particle indexing, wrong call order) -- they do NOT
validate that the real REBOUND/REBOUNDx API is being used correctly, and do
NOT validate any physics. Re-run the equivalent scenario against the real
packages before trusting a production result.

Run with:  python3 -m unittest discover -s tests
"""

import datetime as dt
import sys
import unittest

import numpy as np

# Install the fakes under the real package names BEFORE holmes_dust.simulation
# ever runs `import rebound` / `import reboundx` (those imports are lazy,
# done inside functions, specifically so this substitution works cleanly).
from tests.fakes import fake_rebound, fake_reboundx

sys.modules.setdefault("rebound", fake_rebound)
sys.modules.setdefault("reboundx", fake_reboundx)

from holmes_dust.config import (  # noqa: E402
    ALL_PLANETS,
    Body,
    EjectionDirectionMode,
    EphemerisSource,
    IntegrationConfig,
    ParticleGenerationConfig,
    RadiationModel,
    ScenarioConfig,
)
from holmes_dust.ephemeris import StateVector  # noqa: E402
from holmes_dust.particles import generate_particles  # noqa: E402
from holmes_dust.simulation import (  # noqa: E402
    build_batched_simulation,
    legacy_radial_acceleration,
    get_body_state,
    get_dust_state,
    integrate_to,
)
from holmes_dust import constants  # noqa: E402


class _FakeEphemeris:
    """Deterministic, offline stand-in for a real EphemerisProvider -- just
    enough to exercise simulation.py without any network access."""

    def get_state(self, body: Body, epoch_utc: dt.datetime, comet_designation: str | None = None) -> StateVector:
        # Spread bodies out along the x-axis with plausible AU-scale
        # distances and km/s-scale velocities -- values are not meant to be
        # astronomically accurate, only well-formed.
        offsets = {
            Body.SUN: 0.0,
            Body.MERCURY: 0.4,
            Body.VENUS: 0.7,
            Body.EARTH_MOON_BARYCENTER: 1.0,
            Body.EARTH: 1.0,
            Body.MARS: 1.5,
            Body.JUPITER: 5.2,
            Body.SATURN: 9.5,
            Body.URANUS: 19.2,
            Body.NEPTUNE: 30.1,
            Body.COMET: 3.4,
        }
        au = 1.495978707e11
        x = offsets[body] * au
        return StateVector(
            epoch_utc=epoch_utc,
            position_m=np.array([x, 0.0, 0.0]),
            velocity_m_s=np.array([0.0, 1.0e4, 0.0]),
        )

    def get_state_series(self, body, epochs_utc, comet_designation=None):
        return [self.get_state(body, e, comet_designation) for e in epochs_utc]


def _make_scenario(radiation_model: RadiationModel, include_comet_gravity: bool) -> ScenarioConfig:
    from holmes_dust.config import PARTICLE_BINS

    return ScenarioConfig(
        name="fake-test-scenario",
        comet_designation="17P",
        outburst_epoch_utc=dt.datetime(2007, 10, 24, 12, 0, 0),
        ephemeris_source=EphemerisSource.LEGACY_TABLE,
        gravitating_planets=ALL_PLANETS,
        include_comet_self_gravity=include_comet_gravity,
        comet_assumed_gm_m3_s2=(constants.GM_M3_S2[Body.SUN] / 2.0e15) if include_comet_gravity else None,
        radiation_model=radiation_model,
        particles=ParticleGenerationConfig(
            bins=[PARTICLE_BINS["b1_small"]],
            n_particles_per_bin=3,
            direction_mode=EjectionDirectionMode.ISOTROPIC_SPHERE,
            random_seed=123,
        ),
        integration=IntegrationConfig(duration_years=1.0),
        output_dir="./unused-in-this-test",
    )


class TestBatchedSimulation(unittest.TestCase):
    def setUp(self):
        self.ephemeris = _FakeEphemeris()

    def test_build_batched_simulation_particle_counts(self):
        scenario = _make_scenario(RadiationModel.REBOUNDX_BETA, include_comet_gravity=True)
        comet_state = self.ephemeris.get_state(Body.COMET, scenario.outburst_epoch_utc, scenario.comet_designation)
        rng = np.random.default_rng(scenario.particles.random_seed)
        dust = generate_particles(
            bins=scenario.particles.bins,
            n_particles_per_bin=scenario.particles.n_particles_per_bin,
            direction_mode=scenario.particles.direction_mode,
            comet_position_m=comet_state.position_m,
            comet_velocity_m_s=comet_state.velocity_m_s,
            rng=rng,
        )

        handle = build_batched_simulation(scenario, self.ephemeris, dust)

        # Sun + 8 planets + comet + 3 dust particles = 13
        self.assertEqual(handle.sim.N, 1 + len(ALL_PLANETS) + 1 + len(dust))
        self.assertEqual(len(handle.dust_index), len(dust))

        # Regression test for the real-REBOUND NaN bug found while
        # debugging this against the actual rebound/reboundx packages:
        # N_active MUST equal the number of massive bodies only (Sun +
        # planets + comet), NOT sim.N (which would also count the dust
        # particles). All dust particles start at the exact same position
        # (the comet's location at ejection), so leaving them "active"
        # makes REBOUND's gravity kernel compute a 1/r^2 between two
        # coincident test particles, producing 0 * inf = NaN that then
        # poisons the whole simulation on the very first integration step.
        expected_n_active = 1 + len(ALL_PLANETS) + 1  # Sun + planets + comet
        self.assertEqual(handle.sim.N_active, expected_n_active)

        # Sun must be flagged as the radiation source.
        sun_particle = handle.sim.particles[handle.body_index[Body.SUN]]
        self.assertEqual(sun_particle.params.get("radiation_source"), 1)

        # Every dust particle must carry its own beta, matching the value
        # computed by particles.py (not some default/shared value).
        for p in dust:
            sim_particle = handle.sim.particles[handle.dust_index[p.particle_id]]
            self.assertAlmostEqual(sim_particle.params["beta"], p.beta, places=12)
            self.assertEqual(sim_particle.m, 0.0)

    def test_comet_self_gravity_toggle_changes_particle_count(self):
        scenario_with = _make_scenario(RadiationModel.REBOUNDX_BETA, include_comet_gravity=True)
        scenario_without = _make_scenario(RadiationModel.REBOUNDX_BETA, include_comet_gravity=False)

        comet_state = self.ephemeris.get_state(Body.COMET, scenario_with.outburst_epoch_utc, "17P")
        rng = np.random.default_rng(1)
        dust = generate_particles(
            bins=scenario_with.particles.bins, n_particles_per_bin=2,
            direction_mode=EjectionDirectionMode.ISOTROPIC_SPHERE,
            comet_position_m=comet_state.position_m, comet_velocity_m_s=comet_state.velocity_m_s, rng=rng,
        )

        handle_with = build_batched_simulation(scenario_with, self.ephemeris, dust)
        handle_without = build_batched_simulation(scenario_without, self.ephemeris, dust)

        self.assertEqual(handle_with.sim.N, handle_without.sim.N + 1)
        self.assertIn(Body.COMET, handle_with.body_index)
        self.assertNotIn(Body.COMET, handle_without.body_index)

    def test_integrate_and_read_back_dust_state(self):
        scenario = _make_scenario(RadiationModel.REBOUNDX_BETA, include_comet_gravity=False)
        comet_state = self.ephemeris.get_state(Body.COMET, scenario.outburst_epoch_utc, "17P")
        rng = np.random.default_rng(2)
        dust = generate_particles(
            bins=scenario.particles.bins, n_particles_per_bin=2,
            direction_mode=EjectionDirectionMode.ISOTROPIC_SPHERE,
            comet_position_m=comet_state.position_m, comet_velocity_m_s=comet_state.velocity_m_s, rng=rng,
        )
        handle = build_batched_simulation(scenario, self.ephemeris, dust)

        integrate_to(handle, 86400.0)  # one day forward
        pos, vel = get_dust_state(handle, dust[0].particle_id)
        self.assertEqual(pos.shape, (3,))
        self.assertEqual(vel.shape, (3,))
        # With the fake (non-gravitating) integrator this is a pure drift,
        # so we can at least confirm time actually advanced. The simulation
        # internally tracks time in DAYS (AU/day/Msun unit system -- see
        # simulation.py's module docstring), so 86400 SI seconds requested
        # via integrate_to() becomes sim.t == 1.0 day internally.
        self.assertEqual(handle.sim.t, 1.0)

    def test_get_body_state_returns_planet_position(self):
        scenario = _make_scenario(RadiationModel.REBOUNDX_BETA, include_comet_gravity=False)
        handle = build_batched_simulation(scenario, self.ephemeris, dust_particles=[])
        pos, vel = get_body_state(handle, Body.JUPITER)
        self.assertEqual(pos.shape, (3,))


class TestLegacyForceModel(unittest.TestCase):
    """LEGACY_SUN_GM_REDUCTION = the Java force (gravity of a (1-beta) Sun on
    the dust), implemented as full-mass Sun + outward beta*GM/r^2 term."""

    def setUp(self):
        self.ephemeris = _FakeEphemeris()

    def _dust(self, scenario, n, seed):
        comet_state = self.ephemeris.get_state(Body.COMET, scenario.outburst_epoch_utc, "17P")
        return generate_particles(
            bins=scenario.particles.bins, n_particles_per_bin=n,
            direction_mode=EjectionDirectionMode.ISOTROPIC_SPHERE,
            comet_position_m=comet_state.position_m, comet_velocity_m_s=comet_state.velocity_m_s,
            rng=np.random.default_rng(seed),
        )

    def test_formula_equals_gravity_difference(self):
        # full-Sun gravity + legacy term == gravity of a (1-beta) Sun
        gm, beta = 2.959122e-4, 0.05
        r = np.array([[2.0, -1.0, 0.5]])
        sun = np.array([0.001, 0.002, -0.001])
        d = r[0] - sun
        full = -gm * d / np.linalg.norm(d) ** 3
        reduced = -(1 - beta) * gm * d / np.linalg.norm(d) ** 3
        extra = legacy_radial_acceleration(r, sun, np.array([beta * gm]))[0]
        np.testing.assert_allclose(full + extra, reduced, rtol=1e-13)

    def test_sun_keeps_full_mass_and_all_particles_share_one_sim(self):
        scenario = _make_scenario(RadiationModel.LEGACY_SUN_GM_REDUCTION, include_comet_gravity=False)
        dust = self._dust(scenario, 5, 3)
        handle = build_batched_simulation(scenario, self.ephemeris, dust)
        sun = handle.sim.particles[handle.body_index[Body.SUN]]
        # The planets must orbit the REAL Sun (the old one-sim-per-particle
        # version lowered it, putting Jupiter & co. on wrong orbits).
        self.assertAlmostEqual(sun.m, 1.0, places=12)
        self.assertIsNone(handle.rebx)                       # no REBOUNDx in legacy mode
        self.assertEqual(len(handle.dust_index), 5)          # batched: every particle in one sim
        self.assertEqual(handle.sim.N, 1 + len(ALL_PLANETS) + 5)
        self.assertEqual(handle.sim.N_active, 1 + len(ALL_PLANETS))
        self.assertIs(handle.sim.additional_forces, handle.extra_force_callback)

    def test_callback_adds_outward_acceleration_of_the_right_size(self):
        scenario = _make_scenario(RadiationModel.LEGACY_SUN_GM_REDUCTION, include_comet_gravity=False)
        dust = self._dust(scenario, 2, 5)
        handle = build_batched_simulation(scenario, self.ephemeris, dust)
        handle.extra_force_callback(None)                    # what REBOUND would call
        sim = handle.sim
        sun = sim.particles[handle.body_index[Body.SUN]]
        for particle in dust:
            p = sim.particles[handle.dust_index[particle.particle_id]]
            d = np.array([p.x - sun.x, p.y - sun.y, p.z - sun.z])
            a = np.array([p.ax, p.ay, p.az])
            expected = particle.beta * sim.G * sun.m / np.dot(d, d)
            self.assertAlmostEqual(np.linalg.norm(a), expected, delta=expected * 1e-12)
            self.assertGreater(np.dot(a, d), 0.0)             # pointing AWAY from the Sun

    def test_frame_is_not_recentred_by_default(self):
        # Initial states must stay in the Horizons SSB frame (the Earth
        # positions used for RA/DEC are in that frame too).
        scenario = _make_scenario(RadiationModel.REBOUNDX_BETA, include_comet_gravity=False)
        dust = self._dust(scenario, 1, 7)
        handle = build_batched_simulation(scenario, self.ephemeris, dust)
        pos, _ = get_dust_state(handle, dust[0].particle_id)
        np.testing.assert_allclose(pos, dust[0].initial_position_m, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
