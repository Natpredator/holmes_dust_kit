"""
constants.py
============
Physical constants needed to turn ephemeris state vectors into REBOUND
particles with correct masses.

The old Java code worked in terms of GM (`mu = G*M`) throughout -- e.g.
`final double mu = 1.32712440018e+20;` for the Sun, and never needed a
separate G or a separate mass, because Orekit's force models are
GM-based. REBOUND particles, by contrast, are given a MASS directly (in
whatever unit system `sim.units` is set to), so we need an explicit value
of G to convert GM -> mass consistently: mass_kg = GM_m3_s2 / G_SI.

All GM values below are standard planetary/solar values (DE-ephemeris
class precision, ~6 significant figures) commonly used in solar-system
dynamics work. They are NOT re-derived from the uploaded Java files (which
only ever needed the Sun's GM) -- verify them against whatever ephemeris
kernel / constants file the rest of the project already uses, so the
assumed masses stay consistent with the Horizons-derived positions and
velocities.
"""

from __future__ import annotations  # lets type hints like `dict[Body, float]` work on older Python

from .config import Body  # Body enum: identifies which celestial body a GM value belongs to

# CODATA gravitational constant, SI units (m^3 kg^-1 s^-2). This is the
# single conversion factor everything else in this file is built from.
G_SI = 6.6743e-11

# GM values, m^3/s^2. Body.EARTH is intentionally absent: it is only ever
# used in this project as an *observer* location for RA/DEC, never added to
# the simulation as an independent gravitating mass (its gravity is folded
# into Body.EARTH_MOON_BARYCENTER instead, exactly like the old Java code,
# which added `earthmoonbarycenter` as the gravitating body and used
# `CelestialBodyFactory.getEarth()` only for the observer vector).
GM_M3_S2: dict[Body, float] = {
    Body.SUN: 1.32712440018e20,          # Sun -- identical to the old Java `mu`
    Body.MERCURY: 2.2032e13,             # Mercury
    Body.VENUS: 3.24859e14,              # Venus
    Body.EARTH_MOON_BARYCENTER: 4.0350320e14,  # Earth+Moon combined (see note above)
    Body.MARS: 4.282837e13,              # Mars
    Body.JUPITER: 1.26686534e17,         # Jupiter -- by far the largest planetary GM here
    Body.SATURN: 3.7931187e16,           # Saturn
    Body.URANUS: 5.793939e15,            # Uranus
    Body.NEPTUNE: 6.836529e15,           # Neptune
}


def mass_kg(body: Body) -> float:
    """GM -> mass (kg), using G_SI. Raises KeyError for bodies that have no
    GM entry above (e.g. Body.EARTH, Body.COMET -- the comet's assumed GM
    is scenario-specific and lives in ScenarioConfig.comet_assumed_gm_m3_s2
    instead of in this fixed table)."""
    return GM_M3_S2[body] / G_SI  # rearrange GM = G*M  ->  M = GM/G
