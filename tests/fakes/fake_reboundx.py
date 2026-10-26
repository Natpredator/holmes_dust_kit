"""
fake_reboundx.py
=================
See fake_rebound.py's module docstring -- same purpose, for reboundx.
Only implements the small slice of the documented API that
holmes_dust/simulation.py actually calls: Extras(sim), load_force(name),
add_force(force), and force.params[...] assignment.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Force:
    name: str
    params: dict = dataclasses.field(default_factory=dict)


class Extras:
    def __init__(self, sim):
        self.sim = sim
        self.forces: list[Force] = []

    def load_force(self, name: str) -> Force:
        return Force(name=name)

    def add_force(self, force: Force) -> None:
        self.forces.append(force)
