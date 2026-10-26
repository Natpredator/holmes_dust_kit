"""
simulation.py
=============
Builds and runs the actual N-body simulation using REBOUND for the
gravitational dynamics and REBOUNDx for the radiation-pressure / Poynting-
Robertson-drag effect.

STATUS OF TESTING
-----------------
Both radiation models have been run end-to-end with real rebound
(5.1 / 5.2) and reboundx 5.1: the 78 finished particles of the 1892 Java
run were re-integrated for the full 50140 days, and
LEGACY_SUN_GM_REDUCTION reproduces the Java positions to a median of
0.006 deg on the sky (see README, "Validation"). The offline unit tests
use fakes for rebound/reboundx; tests/test_real_rebound.py holds the
physics checks that need the real packages.

INTERNAL UNITS: AU / day / solar mass, NOT SI
-----------------------------------------------
Every OTHER module in this package (ephemeris.py, particles.py,
observations.py, io_export.py) works in plain SI (metres, seconds,
kilograms), and that is deliberate -- SI is what JPL Horizons naturally
gives you and what the old Java program used throughout.

REBOUND's integrators, however, are numerically conditioned for values of
order unity to a few thousand, which is exactly what AU/day/solar-mass
values look like for solar-system dynamics (positions ~0.1-30, velocities
~0.01-2, masses ~1e-7 to 1). Handing REBOUND raw SI values instead
(positions ~1e11, masses ~1e30) pushes IAS15's predictor-corrector
convergence test into a regime where it silently loses precision and
diverges to NaN within the first few integrator steps -- this is exactly
what happened in the first version of this file (confirmed empirically:
NaN appeared in EVERY particle, including the Sun itself, after
integrating for as little as one simulated minute, in both the REBOUNDx
and the plain-REBOUND-only configurations -- a signature of a numerical
conditioning problem, not a real gravitational event, since a genuine
close encounter would not corrupt the Sun and all eight planets at once).

The fix: this module converts every value AT THE BOUNDARY with REBOUND --
in _add_massive_body(), in the dust-particle-adding loops, in
integrate_to(), and in get_body_state()/get_dust_state() -- and nowhere
else. No other module needs to know REBOUND is internally using different
units; everything that crosses back OUT of this module (via
get_body_state/get_dust_state) is already converted back to SI.

Two radiation models, matching config.RadiationModel
----------------------------------------------------
Both share ONE simulation holding the Sun, the configured planets, the
optional comet nucleus AND every dust particle (as massless test
particles) -- the natural, efficient REBOUND way to integrate thousands of
test particles that all share the same massive-body dynamics.

* REBOUNDX_BETA (recommended): REBOUNDx `radiation_forces`, each dust
  particle carrying its own `beta`. Includes Poynting-Robertson drag.
* LEGACY_SUN_GM_REDUCTION: the exact force the old Java program applied
  (it set the Sun's GM to GM*(1-beta) for each particle): a purely radial
  outward push of beta*GM_sun/r^2, no PR drag. Implemented as an extra
  acceleration through REBOUND's `additional_forces` hook, while the Sun
  keeps its FULL mass -- so the planets stay on their correct orbits. (An
  earlier version of this package literally lowered the Sun's mass in a
  one-particle simulation; that silently put the integrated planets on
  wrong orbits, which the Java code never suffered from because its
  planets came from an ephemeris file, not from integration.)

Reference frame
---------------
The simulation stays in the frame the initial states come from: the JPL
Horizons Solar-System-Barycentre (SSB) frame, ICRF/J2000 equatorial axes --
the same frame the Java program used (Orekit ICRF) and the same frame the
Earth positions for RA/DEC come from. See IntegrationConfig.recenter_to_com
for why re-centring on the centre of mass is OFF by default.
"""

from __future__ import annotations  # allows modern type-hint syntax (e.g. `str | None`) on older Python

import dataclasses   # for the @dataclasses.dataclass decorator (see SimulationHandle)

import numpy as np   # for building/handling the position/velocity arrays returned to callers

from . import constants                                  # G_SI and per-body GM values
from .config import Body, RadiationModel, ScenarioConfig    # types this module reads from config.py
from .ephemeris import EphemerisProvider, StateVector         # where position/velocity data comes from
from .observations import SPEED_OF_LIGHT_M_S                   # exact SI speed of light, for unit conversion
from .particles import DustParticle                              # the type generate_particles() produces

# --- Unit-conversion constants: SI (what the rest of the package uses) <-> ---
# --- AU / day / solar mass (what this module hands to REBOUND) --------------
AU_M = 1.495978707e11      # 1 astronomical unit, in metres (IAU 2012 exact value)
DAY_S = 86400.0            # 1 day, in seconds

# Solar mass, in kilograms, derived from the SAME GM_Sun value the rest of
# the package already uses (constants.GM_M3_S2[Body.SUN]) rather than an
# independent "nominal solar mass" constant. This is deliberate: it makes
# the Sun's mass come out to EXACTLY 1.0 in the simulation's internal
# units by construction, which is both the cleanest possible reference
# point for an AU/day/Msun system and avoids a second, slightly different,
# solar-mass constant drifting out of sync with the GM value used
# everywhere else (e.g. for the LEGACY_SUN_GM_REDUCTION beta trick).
MSUN_KG = constants.GM_M3_S2[Body.SUN] / constants.G_SI   # GM = G*M  ->  M = GM/G

# Speed of light in AU/day (for REBOUNDx's radiation_forces `c` parameter,
# which must be expressed in the simulation's own units). This is the
# well-known constant c = 173.144632674... AU/day -- computing it here from
# SPEED_OF_LIGHT_M_S rather than hard-coding it keeps a single source of
# truth and doubles as a sanity check (the computed value should match the
# textbook constant to several significant figures).
SPEED_OF_LIGHT_AU_PER_DAY = SPEED_OF_LIGHT_M_S * DAY_S / AU_M   # m/s -> AU/day


# A plain container bundling together everything needed to keep stepping a
# REBOUND simulation forward and reading its state back out. `rebound` and
# `reboundx` are only imported INSIDE functions further down (see
# _import_rebound/_import_reboundx), so the type hints below are written as
# quoted strings ("rebound.Simulation") -- Python doesn't need the real
# module to exist just to read a hint written as text.
@dataclasses.dataclass
class SimulationHandle:
    """Everything the rest of the pipeline needs to step a simulation
    forward and read back particle states."""
    sim: "rebound.Simulation"           # noqa: F821 (rebound imported lazily below)
    rebx: "reboundx.Extras | None"      # noqa: F821
    body_index: dict[Body, int]          # massive-body Body -> particle index in sim
    dust_index: dict[int, int]           # DustParticle.particle_id -> particle index in sim
    # Keeps a Python reference to the legacy-force callback alive for as
    # long as the simulation exists (REBOUND calls it from C code; if the
    # Python function object were garbage-collected, that would crash).
    extra_force_callback: object = None


def _import_rebound():
    """Lazy import so the rest of the package (config/particles/observations/
    ephemeris/tests) works even without rebound installed."""
    import rebound   # only happens when this function is actually called
    return rebound


def _import_reboundx():
    import reboundx   # only happens when this function is actually called
    return reboundx


def _add_massive_body(sim, body: Body, state: StateVector, gm_m3_s2: float) -> int:
    """Add one massive body to the simulation from an ephemeris StateVector
    (given in SI: metres, m/s), returning its index in `sim.particles`.

    Converts SI -> AU/day/Msun at this single point (see module docstring)
    so every OTHER function in this file can simply trust that whatever is
    already inside `sim.particles` is in the simulation's internal units.
    """
    mass_kg = gm_m3_s2 / constants.G_SI   # GM -> mass, in kilograms
    sim.add(
        m=mass_kg / MSUN_KG,   # kilograms -> solar masses
        x=state.position_m[0] / AU_M, y=state.position_m[1] / AU_M, z=state.position_m[2] / AU_M,   # metres -> AU
        vx=state.velocity_m_s[0] * DAY_S / AU_M,   # m/s -> AU/day
        vy=state.velocity_m_s[1] * DAY_S / AU_M,
        vz=state.velocity_m_s[2] * DAY_S / AU_M,
    )
    return sim.N - 1   # index of the particle we just added (the last one in the list)


def _add_dust_particle(sim, particle: DustParticle) -> int:
    """Add one massless dust particle, converting its SI initial state
    (from particles.generate_particles) into the simulation's internal
    AU/day units -- same reasoning as _add_massive_body above."""
    sim.add(
        m=0.0,   # dust particles are treated as having zero mass (their own gravity is negligible)
        x=particle.initial_position_m[0] / AU_M,
        y=particle.initial_position_m[1] / AU_M,
        z=particle.initial_position_m[2] / AU_M,
        vx=particle.initial_velocity_m_s[0] * DAY_S / AU_M,
        vy=particle.initial_velocity_m_s[1] * DAY_S / AU_M,
        vz=particle.initial_velocity_m_s[2] * DAY_S / AU_M,
    )
    return sim.N - 1   # index of the particle we just added


def build_batched_simulation(
    scenario: ScenarioConfig,          # which planets/comet-gravity/etc. to use
    ephemeris: EphemerisProvider,       # where to fetch Sun/planet/comet positions from
    dust_particles: list[DustParticle],  # the already-generated particle population
) -> SimulationHandle:
    """Build the ONE shared simulation: Sun + configured planets + optional
    comet nucleus, as massive bodies, plus every dust particle as a
    massless test particle. Radiation pressure is attached according to
    scenario.radiation_model (see module docstring): REBOUNDx
    radiation_forces, or the exact Java force via additional_forces.
    """
    rebound = _import_rebound()     # only imports the real rebound package right here

    sim = rebound.Simulation()   # create a brand-new, empty simulation
    # AU / day / solar mass -- see the module docstring for why this, and
    # not SI, is used internally. sim.units computes the correct G for
    # this unit system automatically, as long as it is set BEFORE any
    # sim.add() call (which is the case here).
    sim.units = ("AU", "day", "Msun")
    sim.integrator = scenario.integration.integrator   # e.g. "ias15"
    # Tiny softening length (~1 km, utterly negligible at any real AU-scale
    # separation) to guard against an exact r=0 singularity between a
    # massive comet nucleus and a dust particle ejected from its exact
    # position at t=0, when include_comet_self_gravity is enabled. Without
    # this, that one pair's acceleration would be genuinely infinite
    # (unsoftened point-mass gravity at zero separation), not just a
    # numerical-precision artefact like the N_active issue handled below.
    sim.softening = 1000.0 / AU_M  # 1000 metres -> AU

    epoch = scenario.outburst_epoch_utc   # the calendar moment everything is anchored to
    body_index: dict[Body, int] = {}       # will map Body -> its index inside sim.particles

    # --- Sun: always present, always the radiation source -----------------
    sun_state = ephemeris.get_state(Body.SUN, epoch)   # fetch the Sun's position/velocity at t=0
    body_index[Body.SUN] = _add_massive_body(sim, Body.SUN, sun_state, constants.GM_M3_S2[Body.SUN])

    # --- Planets: whichever subset this scenario turned on -----------------
    for planet in scenario.gravitating_planets:   # e.g. ALL_PLANETS or MAJOR_PLANETS_ONLY
        state = ephemeris.get_state(planet, epoch)   # fetch this planet's position/velocity
        body_index[planet] = _add_massive_body(sim, planet, state, constants.GM_M3_S2[planet])

    # --- Comet nucleus: optional, scenario-specific assumed GM -------------
    comet_state = ephemeris.get_state(Body.COMET, epoch, comet_designation=scenario.comet_designation)
    if scenario.include_comet_self_gravity:
        assert scenario.comet_assumed_gm_m3_s2 is not None, (
            "include_comet_self_gravity=True requires comet_assumed_gm_m3_s2 to be set"
        )
        body_index[Body.COMET] = _add_massive_body(
            sim, Body.COMET, comet_state, scenario.comet_assumed_gm_m3_s2
        )

    # --- CRITICAL: freeze which particles are gravity SOURCES before adding
    # any dust. REBOUND's default (sim.N_active = -1) treats every particle
    # added so far, and every one added AFTER this point, as an active
    # gravity source for everyone else -- including other massless test
    # particles. Since every dust particle starts at the EXACT SAME position
    # (the comet's location at the ejection epoch -- see particles.py), if
    # test particles were left "active" the gravity kernel would compute a
    # literal 1/r^2 with r=0 between two coincident test particles; even
    # though mass=0 makes the true force zero, 0 * inf = NaN in floating
    # point, and that NaN then poisons every other particle in the
    # simulation within a single step (this is EXACTLY what was observed
    # when debugging this: every particle, including the Sun, turned to NaN
    # after the first tiny integration step). Setting N_active here to "only
    # the massive bodies added so far" tells REBOUND's gravity kernel to
    # skip test particles as sources entirely, so no such pairwise
    # computation between coincident test particles ever happens.
    sim.N_active = sim.N   # everything added so far (Sun + planets [+ comet]) counts as "active"

    # --- Dust particles: massless test particles ---------------------------
    # Test particles feel gravity from the massive bodies above but do not
    # perturb them or each other -- this matches the physical picture (a
    # dust grain's own gravity is utterly negligible) and is exactly what
    # REBOUND's test-particle mode is designed to make efficient for
    # thousands of particles at once.
    dust_index: dict[int, int] = {}   # will map DustParticle.particle_id -> its index in sim.particles
    for particle in dust_particles:
        dust_index[particle.particle_id] = _add_dust_particle(sim, particle)

    # Optional re-centring on the centre of mass/momentum (OFF by default,
    # see IntegrationConfig.recenter_to_com). If used, it must come AFTER
    # every particle has been added: move_to_com() shifts ALL particles by
    # the same offset, so doing it earlier would leave test particles added
    # afterwards inconsistent with the shifted massive bodies.
    if scenario.integration.recenter_to_com:
        sim.move_to_com()

    if scenario.radiation_model is RadiationModel.LEGACY_SUN_GM_REDUCTION:
        # Exact Java force model, no REBOUNDx needed at all.
        callback = _install_legacy_radial_pressure(sim, body_index[Body.SUN], dust_particles, dust_index)
        return SimulationHandle(sim=sim, rebx=None, body_index=body_index, dust_index=dust_index,
                                extra_force_callback=callback)

    # --- REBOUNDx: radiation pressure + Poynting-Robertson drag -------------
    reboundx = _import_reboundx()                       # only imported when actually needed
    rebx = reboundx.Extras(sim)                        # attach the REBOUNDx "extras" layer to this simulation
    radiation_force = rebx.load_force("radiation_forces")   # load the specific effect we want
    rebx.add_force(radiation_force)                     # actually enable it for this simulation
    # Speed of light, expressed in the simulation's own units (AU/day here,
    # since sim.units == ("AU","day","Msun")). If you ever change sim.units,
    # this MUST be reconverted -- reboundx does not do it automatically.
    radiation_force.params["c"] = SPEED_OF_LIGHT_AU_PER_DAY
    # Flag the Sun particle as the radiation source.
    sim.particles[body_index[Body.SUN]].params["radiation_source"] = 1
    # Give every dust particle its own beta -- each particle keeps its own
    # radiation response inside ONE shared simulation. beta is
    # dimensionless (a ratio of two forces), so it needs no unit conversion.
    for particle in dust_particles:
        sim.particles[dust_index[particle.particle_id]].params["beta"] = particle.beta

    # Bundle the finished simulation, its REBOUNDx layer, and both index
    # dictionaries into one handle for the rest of the pipeline to use.
    return SimulationHandle(sim=sim, rebx=rebx, body_index=body_index, dust_index=dust_index)


def legacy_radial_acceleration(dust_xyz: np.ndarray, sun_xyz: np.ndarray, beta_gm: np.ndarray) -> np.ndarray:
    """Extra acceleration of the Java force model, for many particles at once.

    a_i = + beta_i * GM_sun * (r_i - r_sun) / |r_i - r_sun|^3

    i.e. exactly the part of the Sun's gravity that the Java program removed
    with GM_sun -> GM_sun*(1-beta): gravity of the full Sun (applied by
    REBOUND) plus this term = gravity of a (1-beta) Sun. Units are whatever
    the inputs use (AU, AU/day^2 and GM in AU^3/day^2 inside the simulation).
    """
    d = dust_xyz - sun_xyz[None, :]                     # Sun -> particle vectors, shape (N, 3)
    r3 = np.sum(d * d, axis=1) ** 1.5                   # |d|^3 for every particle
    return (beta_gm / r3)[:, None] * d                  # outward, magnitude beta*GM/r^2


def _particle_array_view(sim):
    """(numpy view of REBOUND's C particle array as rows of doubles,
    (column of x, column of ax)), or None if that is not possible (e.g. the
    offline test fakes). Reading/writing this view is thousands of times
    faster than touching sim.particles[i].x one attribute at a time.
    Relies on REBOUND's documented ctypes layout: `sim._particles` is a
    POINTER(rebound.Particle) whose x..az fields are doubles."""
    try:
        import ctypes
        ptr = sim._particles
        ptype = type(ptr.contents)
        size = ctypes.sizeof(ptype)
        ox, oax = ptype.x.offset, ptype.ax.offset
        if size % 8 or ox % 8 or oax % 8 or ptype.y.offset != ox + 8 or ptype.ay.offset != oax + 8:
            return None
        buf = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))
        arr = np.ctypeslib.as_array(buf, shape=(sim.N * size // 8,)).reshape(sim.N, size // 8)
        return arr, (ox // 8, oax // 8)
    except Exception:
        return None


def _install_legacy_radial_pressure(sim, sun_index: int, dust_particles, dust_index: dict[int, int]):
    """Attach legacy_radial_acceleration() to the simulation through
    REBOUND's `additional_forces` hook. Returns the callback (the caller
    must keep a reference to it -- see SimulationHandle.extra_force_callback).

    Performance note: REBOUND calls this Python function at every force
    evaluation (several per IAS15 step). Positions are read and
    accelerations written through a numpy view of REBOUND's particle array
    (see _particle_array_view), so the cost is a few numpy operations per
    call, not a Python loop over particles.
    """
    ps = sim.particles                                   # live view of the particle array
    indices = [dust_index[p.particle_id] for p in dust_particles]         # simulation index of each dust grain
    idx = np.array(indices, dtype=int)
    # beta * GM_sun in simulation units: G (set by sim.units) * Sun mass (Msun).
    beta_gm = np.array([p.beta for p in dust_particles]) * sim.G * ps[sun_index].m

    def force(_reb_sim_pointer):                           # REBOUND passes a pointer we do not need
        view = _particle_array_view(sim)                   # fast path: numpy view of the C array
        if view is not None:
            arr, (ox, oax) = view
            xyz = arr[idx, ox:ox + 3]                        # dust positions, shape (N, 3)
            arr[idx, oax:oax + 3] += legacy_radial_acceleration(xyz, arr[sun_index, ox:ox + 3], beta_gm)
            return
        # Slow but API-only path (also used by the offline test fakes).
        sun = ps[sun_index]
        sun_xyz = np.array([sun.x, sun.y, sun.z])
        xyz = np.array([[ps[i].x, ps[i].y, ps[i].z] for i in indices])   # current dust positions
        acc = legacy_radial_acceleration(xyz, sun_xyz, beta_gm)
        for k, i in enumerate(indices):                     # ADD to the gravity REBOUND already computed
            p = ps[i]
            p.ax += acc[k, 0]
            p.ay += acc[k, 1]
            p.az += acc[k, 2]

    if indices:                                             # nothing to do without dust particles
        sim.additional_forces = force
        # Position-only force: lets IAS15 skip velocity-dependent iterations.
        sim.force_is_velocity_dependent = 0
    return force


def integrate_to(handle: SimulationHandle, t_seconds: float) -> None:
    """Advance the simulation to an absolute simulation time, given in
    SECONDS since the scenario's outburst epoch (t=0 at ejection) -- this
    keeps the function's PUBLIC contract in SI, matching every other
    time value in the package (run_scenario.py's output-epoch loop counts
    in seconds too). Internally converts to days, since the simulation's
    own clock (sim.t) advances in the AU/day/Msun unit system.
    """
    handle.sim.integrate(t_seconds / DAY_S)   # seconds -> days, then hand off to REBOUND's own integrator


def get_body_state(handle: SimulationHandle, body: Body) -> tuple[np.ndarray, np.ndarray]:
    """Current (position_m, velocity_m_s) of a massive body already added
    to the simulation -- converted back to SI here, so nothing outside
    this module ever sees an AU/day value."""
    p = handle.sim.particles[handle.body_index[body]]   # look up this body's REBOUND particle object
    position_m = np.array([p.x, p.y, p.z]) * AU_M        # AU -> metres
    velocity_m_s = np.array([p.vx, p.vy, p.vz]) * AU_M / DAY_S   # AU/day -> m/s
    return position_m, velocity_m_s


def get_dust_state(handle: SimulationHandle, particle_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Current (position_m, velocity_m_s) of one dust particle -- converted
    back to SI here, same as get_body_state above."""
    p = handle.sim.particles[handle.dust_index[particle_id]]   # look up this particle's REBOUND object
    position_m = np.array([p.x, p.y, p.z]) * AU_M
    velocity_m_s = np.array([p.vx, p.vy, p.vz]) * AU_M / DAY_S
    return position_m, velocity_m_s
