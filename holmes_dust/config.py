"""
config.py
=========
Central place for ALL scenario and simulation configuration.

Design goal
-----------
In the original Java program every "knob" (which historical outburst to
model, which planets to include, how many particles, which dates to export)
was a plain ``int``/``double`` variable hand-edited inside one giant
``main()`` method (``OutburstAndInitialDateIndex``, ``testingmode``,
``onlymajorplanets``, ``maxparticles`` ...). Changing scenario meant editing
source code and recompiling.

Here, every one of those knobs becomes an explicit, named, documented field
on a dataclass. Nothing in this file does any physics, numerics or I/O --
it is pure data. Adding a new historical scenario or a new output snapshot
date should only ever require adding an entry here (or in an external
config file loaded at runtime), never touching simulation/particles/
observations code.
"""

# Lets type hints below (like `tuple[Body, ...]`) work without needing the
# newest Python version -- Python evaluates annotations lazily instead of
# immediately, so slightly newer syntax is tolerated on slightly older
# interpreters.
from __future__ import annotations

import dataclasses          # gives us the @dataclasses.dataclass decorator used everywhere below
import datetime as dt        # for the `datetime` type used by epoch/date fields
import enum                  # gives us the base class for our enums (Body, RadiationModel, ...)
from typing import Optional, Sequence  # Optional[x] = "x or None"; Sequence = "list-like, read-only"


# ---------------------------------------------------------------------------
# Bodies that can be added as gravitating masses in the simulation
# ---------------------------------------------------------------------------

# `class Body(str, enum.Enum)` means each member is BOTH a proper enum value
# (so code can compare `body is Body.SUN` safely) AND behaves like a plain
# string (so it prints nicely and can be used as a dict key / JSON value).
class Body(str, enum.Enum):
    """Identifiers for every body the pipeline knows about.

    These strings double as the keys passed to the ephemeris backend
    (see ephemeris.py), so keep them stable once other code depends on them.
    """
    SUN = "sun"                              # the Sun -- always present, always the radiation source
    MERCURY = "mercury"                      # innermost planet
    VENUS = "venus"                          # second planet
    EARTH = "earth"                          # observer location (see observations.py) -- NOT added to the N-body sim as a mass
    EARTH_MOON_BARYCENTER = "earth_moon_barycenter"  # Earth+Moon combined -- THIS is what's added as a gravitating mass
    MARS = "mars"                            # fourth planet
    JUPITER = "jupiter"                      # largest planet, biggest perturber of dust orbits
    SATURN = "saturn"                        # second-largest planet
    URANUS = "uranus"                        # ice giant
    NEPTUNE = "neptune"                      # outermost planet modelled here
    COMET = "comet"                          # the modelled comet nucleus itself (17P/Holmes, 10P/Tempel2, ...)


# The old Java flag `onlymajorplanets == 1` skipped Mercury/Uranus/Neptune.
# We replace that opaque integer with two named, explicit tuples instead.
# A "tuple of Body values" is just a fixed, ordered, read-only list of which
# planets a scenario should include as gravitating masses.
MAJOR_PLANETS_ONLY: tuple[Body, ...] = (
    Body.VENUS,                  # include Venus's gravity
    Body.EARTH_MOON_BARYCENTER,  # include Earth+Moon's gravity
    Body.MARS,                   # include Mars's gravity
    Body.JUPITER,                # include Jupiter's gravity
    Body.SATURN,                 # include Saturn's gravity
    # Mercury, Uranus, Neptune deliberately excluded from this tuple.
)
ALL_PLANETS: tuple[Body, ...] = (
    Body.MERCURY,                # include Mercury's gravity
    Body.VENUS,                  # include Venus's gravity
    Body.EARTH_MOON_BARYCENTER,  # include Earth+Moon's gravity
    Body.MARS,                   # include Mars's gravity
    Body.JUPITER,                # include Jupiter's gravity
    Body.SATURN,                 # include Saturn's gravity
    Body.URANUS,                 # include Uranus's gravity
    Body.NEPTUNE,                # include Neptune's gravity
)


# ---------------------------------------------------------------------------
# Dust particle size bins  (old Java "bvalue" 1 / 2 / 3 loop)
# ---------------------------------------------------------------------------

# @dataclasses.dataclass auto-generates a constructor (__init__), a nice
# printable representation (__repr__), and equality comparison (__eq__) for
# this class, from just the field list below -- no need to write any of
# that by hand. `frozen=True` makes instances read-only after creation
# (trying to change a field afterwards raises an error), which is a safety
# net against accidentally mutating shared config objects.
@dataclasses.dataclass(frozen=True)
class ParticleSizeBin:
    """One dust-size population.

    Radius bounds are in MILLIMETRES on purpose: the empirical beta(radius)
    and ejection-speed(radius) formulas ported into particles.py expect
    millimetres, matching the units implied by the original Java comments
    ("0.001 mm - 0.01 mm small particles").
    """
    name: str                  # short identifier, e.g. "b1_small" -- used in filenames/columns
    radius_min_mm: float       # smallest possible radius for a particle in this bin (millimetres)
    radius_max_mm: float       # largest possible radius for a particle in this bin (millimetres)
    color_hint: str = ""       # cosmetic only, copied from the old SRON BRIGHT color comments


# The three size classes used throughout the old Java code. This is a plain
# Python dict: keys are short string names, values are the ParticleSizeBin
# objects defined just above.
#
# NOTE ON A DEAD PARAMETER: the old code also defined per-bin "maximum
# ejection speed" constants (550 / 440 / 360 m/s). Those constants only ever
# fed the *disabled* legacy branch `locked_beta_and_ejection_speed == 0`,
# which is never turned on in any of the six source files supplied. The
# branch that IS always active (`== 1`) computes ejection speed purely from
# particle radius (see particles.ejection_speed_from_radius_mm). Because of
# this, the per-bin speed caps are legacy/dead and are intentionally NOT
# ported here.
PARTICLE_BINS: dict[str, ParticleSizeBin] = {
    # name -> ParticleSizeBin(name, radius_min_mm, radius_max_mm, color_hint)
    "b1_small": ParticleSizeBin("b1_small", 0.001, 0.01, "blue"),     # smallest, most beta-affected dust
    "b2_medium": ParticleSizeBin("b2_medium", 0.01, 0.1, "yellow"),   # medium-sized dust
    "b3_big": ParticleSizeBin("b3_big", 0.1, 1.0, "red"),             # largest, least beta-affected dust
}


# Another enum -- see the comment on `class Body` above for what
# `(str, enum.Enum)` means.
class EjectionDirectionMode(str, enum.Enum):
    """How the ejection-velocity direction is sampled for each particle."""
    ISOTROPIC_SPHERE = "isotropic_sphere"  # the only mode actually enabled in the old code
    TOWARD_SUN = "toward_sun"              # implemented in the old code, but never enabled
                                            # (locked_ejection_speed_toward_sun was always 0)


class RadiationModel(str, enum.Enum):
    """How solar-radiation pressure (the ``beta`` effect) is applied.

    REBOUNDX_BETA (recommended default)
        Uses REBOUNDx's built-in ``radiation_forces`` effect. Every test
        particle in ONE shared N-body simulation can carry its own
        ``beta`` parameter, and the effect also includes Poynting-Robertson
        drag -- a real physical effect the old Java code never modelled at
        all (it only ever touched the *gravity* term).

    LEGACY_SUN_GM_REDUCTION
        Reproduces the old Java force model exactly: the Java code scaled
        the Sun's GM by ``(1 - beta)`` for each particle
        (``sun.setGM(mu*(1-Ybeta))``), i.e. a purely radial radiation-
        pressure push of ``beta * GM_sun / r^2`` away from the Sun, with NO
        Poynting-Robertson drag.

        IMPORTANT: this package does NOT literally lower the Sun's mass.
        In the Java program the planets were read from an ephemeris file, so
        a lighter Sun only affected the dust. In an N-body code the planets
        are integrated too -- a lighter Sun would put Jupiter, Saturn, ...
        on wrong orbits (a ~3-6 % period error for beta ~0.03-0.06, i.e.
        tens of degrees of orbital phase after a century). Instead, the Sun
        keeps its full mass and every dust particle gets an extra outward
        acceleration of ``+beta * GM_sun * r_hat / r^2`` through REBOUND's
        ``additional_forces`` hook (see simulation.py). For the dust grain
        this is mathematically identical to the Java ``(1-beta)`` Sun, and
        it keeps the planets correct -- and it lets all particles share one
        batched simulation, just like REBOUNDX_BETA.
    """
    REBOUNDX_BETA = "reboundx_beta"                      # modern, recommended, batched
    LEGACY_SUN_GM_REDUCTION = "legacy_sun_gm_reduction"  # exact old-Java force: radial push, no PR drag


class OutputFormat(str, enum.Enum):
    """Which CSV layout(s) run_scenario.py writes (see io_export.py)."""
    MODERN = "modern"   # trajectory_full.csv + snapshot_<name>.csv (named columns, header row)
    LEGACY = "legacy"   # the old Java layout: one b<bin>p<N>.csv per particle + b<bin>f<NN>/m<N>.csv
    BOTH = "both"       # write both of the above in the same run


class EphemerisSource(str, enum.Enum):
    """Where initial state vectors for the comet and the planets come from."""
    JPL_HORIZONS = "jpl_horizons"  # live query, any target body, any epoch (recommended)
    # Hybrid used to REPRODUCE an old Java run: the comet's initial state at
    # the outburst epoch is taken from the hard-coded Java table (so every
    # particle starts from exactly the same point/velocity the Java program
    # used), while the Sun, the planets and the Earth (for RA/DEC) still
    # come from JPL Horizons.
    HORIZONS_WITH_LEGACY_COMET = "horizons_with_legacy_comet"
    # Same idea, fully OFFLINE: Sun/planets/Earth from a local JPL
    # planetary ephemeris file (de440.bsp etc., via the jplephem package),
    # comet start state from the Java table. Needs ScenarioConfig.spk_path.
    SPK_FILE_WITH_LEGACY_COMET = "spk_file_with_legacy_comet"
    LEGACY_TABLE = "legacy_table"  # hard-coded comet+Sun vectors ported from the Java
                                    # `if (OutburstAndInitialDateIndex == N)` blocks;
                                    # ONLY valid for Body.SUN and Body.COMET, and only at
                                    # the specific epochs listed in ephemeris.LEGACY_STATES
                                    # -- see the docstring of LegacyTableEphemerisProvider.


@dataclasses.dataclass(frozen=True)
class SnapshotSpec:
    """Definition of a single named output slice.

    This one dataclass replaces ALL of the old hand-written
    ``PrintWriter b1outm2`` / ``b1outf10`` / ... variables together with
    their matching
    ``if (finalState2.getDate().toString().equals("2026-08-01T..."))``
    blocks. Adding a new forecast date, or a new historical-observation
    match window, now means adding one list entry -- no source change.
    """
    name: str                              # used to build the output file name
    date_utc: dt.datetime                  # calendar epoch this slice targets
    # If the two fields below are set, this behaves like an old "m"
    # (measurement / matching) file: a row is kept only if the *computed*
    # RA/DEC lands within `tolerance_deg` of a real historical observation.
    # If left as None, this behaves like an old "f" (forecast) file: every
    # particle's row at this date is kept unconditionally.
    observed_ra_deg: Optional[float] = None    # real historical RA to compare against, or None
    observed_dec_deg: Optional[float] = None   # real historical DEC to compare against, or None
    tolerance_deg: float = 40.0                # old code's hardcoded RA/DEC box size, in degrees
    note: str = ""                               # free text, e.g. which Java file/variable this mirrors


@dataclasses.dataclass(frozen=True)
class IntegrationConfig:
    """Everything about *how* the propagation is run, independent of which
    comet outburst is being modelled."""
    integrator: str = "ias15"          # adaptive; safe default for velocity-dependent
                                        # forces (radiation pressure + PR drag).
                                        # "whfast" is a faster alternative for large
                                        # particle counts -- see simulation.py docstring
                                        # for the extra care it needs.
    min_step_s: float = 1.0            # (currently informational only -- see simulation.py)
    max_step_s: float = 7200.0         # (currently informational only -- see simulation.py)
    output_cadence_s: float = 86400.0  # how often to record/export a state: one day, matches the old Java `stepT`
    duration_years: float = 25.0       # total length of the propagation, in Julian years
    # Exact length in days. When set, it WINS over duration_years. The Java
    # programs hard-coded their durations in seconds (e.g. 4332096000 s =
    # exactly 50140 days for the 1892 run: 1892-11-04 -> 2030-02-14), and a
    # "years" number cannot hit that exactly, so the profiles use this.
    duration_days: Optional[float] = None
    apply_light_time_correction: bool = True  # NEW capability, absent from the old code
    # Re-centre the massive bodies on their own centre of mass before
    # integrating. OFF by default: the Earth positions used for RA/DEC come
    # from Horizons in the Solar-System-Barycentre frame, so the simulation
    # must stay in that same frame. Re-centring shifts every particle by the
    # (non-zero) offset between the SSB and the centre of mass of just the
    # bodies included -- ~0.003 AU and ~0.5 m/s when Uranus/Neptune are left
    # out -- which then shows up as a spurious RA/DEC error.
    recenter_to_com: bool = False


@dataclasses.dataclass(frozen=True)
class ParticleGenerationConfig:
    """Everything about *how many* dust particles to make and *how* to
    pick their ejection direction."""
    bins: Sequence[ParticleSizeBin]    # which size classes to generate (e.g. just b1, or all three)
    n_particles_per_bin: int = 400     # how many particles to generate PER bin listed above
    direction_mode: EjectionDirectionMode = EjectionDirectionMode.ISOTROPIC_SPHERE  # see enum above
    random_seed: Optional[int] = None  # None -> nondeterministic, like the old Math.random() seed


@dataclasses.dataclass(frozen=True)
class ScenarioConfig:
    """Top-level configuration for one full run.

    Roughly equivalent to picking one ``OutburstAndInitialDateIndex`` value
    plus one particular combination of flags in the old code -- but here the
    two concerns (WHICH epoch, WHICH physics/flag profile) are independent
    and can be freely combined (see scenarios.py for ready-made examples).
    """
    name: str                                # human-readable label for this run, e.g. "1892-11-04_17P"
    comet_designation: str                 # e.g. "17P" (Holmes) or "10P" (Tempel 2) --
                                            # passed straight to the ephemeris backend,
                                            # so the pipeline is not hardwired to one comet
    outburst_epoch_utc: dt.datetime          # calendar date/time of the modelled outburst (t=0)
    ephemeris_source: EphemerisSource        # JPL_HORIZONS or LEGACY_TABLE -- see enum above
    gravitating_planets: Sequence[Body]       # which planets to add as masses -- e.g. ALL_PLANETS
    include_comet_self_gravity: bool          # True -> the comet nucleus is itself a gravitating mass
    comet_assumed_gm_m3_s2: Optional[float]  # None if include_comet_self_gravity is False;
                                              # see ephemeris.py for how the old value
                                              # (mu / 2e15) was derived -- it is an assumed
                                              # number, not a measured comet mass
    radiation_model: RadiationModel           # REBOUNDX_BETA or LEGACY_SUN_GM_REDUCTION -- see enum above
    particles: ParticleGenerationConfig       # how many/which particles to generate -- see dataclass above
    integration: IntegrationConfig            # how to run the propagation -- see dataclass above
    snapshots: Sequence[SnapshotSpec] = ()   # named output-date slices to also export; empty by default
    output_dir: str = "./output"              # folder where CSV output files get written
    output_format: OutputFormat = OutputFormat.MODERN   # modern CSV, legacy Java CSV layout, or both
    # The Java `OutburstAndInitialDateIndex` of this epoch (e.g. 9 for
    # 1892-11-04). Only used to fill the "index" column of the legacy
    # Java-layout files; None -> written as -1.
    legacy_epoch_index: Optional[int] = None
    # Path to a JPL .bsp planetary ephemeris, for EphemerisSource.SPK_FILE_WITH_LEGACY_COMET.
    spk_path: Optional[str] = None

    def duration_seconds(self) -> float:
        """Total propagation length in seconds, derived from the configured
        number of years (365.25-day Julian year, same convention implicitly
        used by the old code's hand-computed `duration` constants).
        If `integration.duration_days` is set, it is used instead (exact)."""
        if self.integration.duration_days is not None:
            return self.integration.duration_days * 86400.0   # exact day count -> seconds
        # years -> days -> seconds, using the standard 365.25-day Julian year.
        return self.integration.duration_years * 365.25 * 86400.0
