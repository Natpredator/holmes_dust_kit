"""
Physics checks that need the REAL rebound (and reboundx) packages.

Skipped automatically when rebound is not installed, or when another test
module has already put the offline fakes in its place -- so run this file
ON ITS OWN to execute it:

    PYTHONPATH=. python3 -m unittest tests.test_real_rebound -v

What it checks
--------------
1. The LEGACY_SUN_GM_REDUCTION force (full-mass Sun + outward beta*GM/r^2
   via additional_forces) moves a dust grain exactly like a Sun of mass
   (1 - beta) would -- i.e. it is the Java `sun.setGM(mu*(1-beta))` model.
2. REBOUNDx radiation_forces with the same beta gives nearly the same orbit
   over a short time (it differs only by the small Poynting-Robertson drag).
"""

import sys
import unittest

import numpy as np


def _real_rebound():
    mod = sys.modules.get("rebound")
    if mod is not None and "fake" in (getattr(mod, "__file__", "") or ""):
        return None                                   # the offline fake is loaded in this process
    try:
        import rebound
    except ImportError:
        return None
    return rebound if "fake" not in (getattr(rebound, "__file__", "") or "") else None


REBOUND = _real_rebound()


@unittest.skipIf(REBOUND is None, "real rebound not available in this process (run this file on its own)")
class TestLegacyForceWithRealRebound(unittest.TestCase):
    def _sim(self, sun_mass):
        sim = REBOUND.Simulation()
        sim.units = ("AU", "day", "Msun")
        sim.integrator = "ias15"
        sim.add(m=sun_mass)                                   # Sun at the origin
        sim.add(m=0.0, x=2.5, y=0.0, z=0.1, vx=0.0, vy=0.011, vz=0.001)   # a dust grain, AU / AU per day
        sim.N_active = 1
        return sim

    def test_additional_force_equals_reduced_sun_mass(self):
        from holmes_dust.simulation import _install_legacy_radial_pressure

        class P:                                              # minimal stand-in for a DustParticle
            particle_id, beta = 0, 0.06

        beta = P.beta
        ref = self._sim(1.0 - beta)                          # what the Java program did
        ref.integrate(3000.0)
        sim = self._sim(1.0)                                 # what this package does
        cb = _install_legacy_radial_pressure(sim, 0, [P], {0: 1})
        sim.integrate(3000.0)
        a = np.array(ref.particles[1].xyz)
        b = np.array(sim.particles[1].xyz)
        self.assertLess(np.linalg.norm(a - b), 1e-9)          # AU, after ~8 years
        self.assertIsNotNone(cb)

    def test_reboundx_radiation_is_close_to_legacy_over_short_times(self):
        try:
            import reboundx
        except ImportError:
            self.skipTest("reboundx not installed")
        from holmes_dust.simulation import SPEED_OF_LIGHT_AU_PER_DAY
        beta = 0.06
        ref = self._sim(1.0 - beta)
        ref.integrate(365.0)
        sim = self._sim(1.0)
        rebx = reboundx.Extras(sim)
        rf = rebx.load_force("radiation_forces")
        rebx.add_force(rf)
        rf.params["c"] = SPEED_OF_LIGHT_AU_PER_DAY
        sim.particles[0].params["radiation_source"] = 1
        sim.particles[1].params["beta"] = beta
        sim.integrate(365.0)
        d = np.linalg.norm(np.array(ref.particles[1].xyz) - np.array(sim.particles[1].xyz))
        self.assertGreater(d, 0.0)                           # PR drag is not zero ...
        self.assertLess(d, 1e-4)                             # ... but tiny after one year (AU)


if __name__ == "__main__":
    unittest.main()
