"""
fake_rebound.py
================
A DELIBERATELY MINIMAL stand-in for the real `rebound` package, used only to
smoke-test the ORCHESTRATION logic in holmes_dust/simulation.py (correct
attribute names, correct indexing, correct call order) offline, in an
environment where the real rebound/reboundx packages cannot be installed
(no network access -- see simulation.py's module docstring).

This fake does NOT implement real gravitational dynamics -- `integrate()`
below only performs a straight-line drift (position += velocity * dt). It
exists purely to catch Python-level bugs in holmes_dust.simulation (wrong
attribute name, off-by-one in indexing, etc.), NOT to validate physics or
confirm the real rebound API is being used correctly. Always test against
the real rebound/reboundx packages before trusting a production run.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Particle:
    m: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    params: dict = dataclasses.field(default_factory=dict)


class Simulation:
    def __init__(self):
        self.particles: list[Particle] = []
        self.units = None
        self.integrator = "ias15"
        self.G = 1.0
        self.t = 0.0

    @property
    def N(self) -> int:
        return len(self.particles)

    def add(self, m=0.0, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0):
        self.particles.append(Particle(m=m, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz))

    def move_to_com(self):
        # Real center-of-mass removal, using only the massive particles'
        # contribution -- good enough to exercise the "must run after all
        # particles are added" ordering requirement documented in
        # simulation.py.
        total_mass = sum(p.m for p in self.particles)
        if total_mass == 0:
            return
        com_pos = [sum(p.m * getattr(p, axis) for p in self.particles) / total_mass for axis in ("x", "y", "z")]
        com_vel = [sum(p.m * getattr(p, axis) for p in self.particles) / total_mass for axis in ("vx", "vy", "vz")]
        for p in self.particles:
            p.x -= com_pos[0]
            p.y -= com_pos[1]
            p.z -= com_pos[2]
            p.vx -= com_vel[0]
            p.vy -= com_vel[1]
            p.vz -= com_vel[2]

    def integrate(self, t):
        dt = t - self.t
        for p in self.particles:
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.z += p.vz * dt
        self.t = t
