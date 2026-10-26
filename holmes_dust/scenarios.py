"""
scenarios.py
============
Concrete, ready-to-run scenarios, built by combining:

    1. an OUTBURST EPOCH  (was: ``OutburstAndInitialDateIndex`` 0..15), and
    2. a FLAG PROFILE     (was: the surrounding testingmode /
                            withoutholmesgravitation / onlymajorplanets /
                            bvalue-loop-range block that differed between
                            the six Java files).

The old code welded these two things together -- every date lived inside
one specific file with one specific set of flags. Splitting them here means
any of the 16 historical epochs can be combined with any physics profile,
which is both simpler and strictly more capable than the original.

IMPORTANT / VERIFIED-FROM-CODE vs INTERPRETATION
-------------------------------------------------
* Epoch indices 0-8 and the "no comment" cases were all found inside files
  named ``SlaveMode_Holmes...``, so they are treated here as apparitions of
  17P/Holmes. This is a reasonable inference from context, not a fact
  stated in-line for every single date.
* Epoch index 9 (1892-11-04) and 10 (1893-01-16) match the historically
  documented discovery outburst and secondary flare-up of 17P/Holmes --
  consistent with comet_designation="17P" here.
* Epoch indices 13, 14, 15 are UNAMBIGUOUSLY labelled in the source code
  comments as **"10P/Tempel2"**, NOT Holmes. This was missed in the first
  (analysis-only) pass and is corrected here: these three are registered
  with comet_designation="10P".
* Epoch index 11 (1899-07-04) and 12 (2015-01-26) carry no comet name in
  the comments at all. They are left tagged "17P" (matching the file they
  live in) but this cannot be confirmed from the code alone -- verify
  against the wider project before trusting these two for real analysis.
"""

from __future__ import annotations  # allows the `str | None` union syntax below

import datetime as dt   # for the `datetime` values in OUTBURST_EPOCHS

# Pull in the names this module needs from config.py.
from .config import (
    ALL_PLANETS,               # tuple: every planet Mercury..Neptune
    MAJOR_PLANETS_ONLY,         # tuple: Venus/Earth/Mars/Jupiter/Saturn only
    PARTICLE_BINS,               # dict: name -> ParticleSizeBin
    Body,                          # enum: Sun/planets/comet identifiers
    EjectionDirectionMode,          # enum: isotropic sphere vs toward-Sun
    EphemerisSource,                 # enum: Horizons vs legacy table
    IntegrationConfig,                # dataclass: how to run the propagation
    ParticleGenerationConfig,          # dataclass: how many/which particles
    RadiationModel,                     # enum: REBOUNDx vs legacy GM trick
    ScenarioConfig,                      # dataclass: the final, combined result
    SnapshotSpec,                         # dataclass: one forecast/matching output slice
)
from .legacy_java import (
    JAVA_1892_SNAPSHOTS,            # f10..f22 + m2 + m14, as in SlaveMode_Holmes1892_Refactor_1892.java
    JAVA_2007_REFACTOR_SNAPSHOTS,   # f10..f25 + m2 + m14, as in SlaveMode_Holmes1892_Refactor_2007.java
)

# ---------------------------------------------------------------------------
# 1. Outburst epoch table -- ported 1:1 from the `if (OutburstAndInitialDateIndex
#    == N) { initialDate = new AbsoluteDate(...); }` chain found (in slightly
#    extended form) in every one of the six Java files.
# ---------------------------------------------------------------------------

# A plain dict: short date-string key -> (the actual datetime, which comet
# it belongs to, a human-readable note about where this came from).
# key -> (epoch_utc, comet_designation, human-readable note)
OUTBURST_EPOCHS: dict[str, tuple[dt.datetime, str, str]] = {
    "2007-10-20": (dt.datetime(2007, 10, 20, 12, 0, 0), "17P", "index 0"),
    "2007-10-21": (dt.datetime(2007, 10, 21, 12, 0, 0), "17P", "index 1"),
    "2007-10-22": (dt.datetime(2007, 10, 22, 12, 0, 0), "17P", "index 2"),
    "2007-10-23": (dt.datetime(2007, 10, 23, 12, 0, 0), "17P", "index 3 (default in most files)"),
    "2007-10-24": (dt.datetime(2007, 10, 24, 12, 0, 0), "17P", "index 4 -- the well-known 2007 outburst date"),
    "2007-10-25": (dt.datetime(2007, 10, 25, 12, 0, 0), "17P", "index 5"),
    "2007-10-26": (dt.datetime(2007, 10, 26, 12, 0, 0), "17P", "index 6"),
    "2007-10-27": (dt.datetime(2007, 10, 27, 12, 0, 0), "17P", "index 7"),
    "2007-10-28": (dt.datetime(2007, 10, 28, 12, 0, 0), "17P", "index 8"),
    "1892-11-04": (dt.datetime(1892, 11, 4, 12, 0, 0), "17P", "index 9 -- Holmes discovery outburst"),
    "1893-01-16": (dt.datetime(1893, 1, 16, 12, 0, 0), "17P", "index 10 -- Holmes secondary flare-up"),
    "1899-07-04": (dt.datetime(1899, 7, 4, 12, 0, 0), "17P", "index 11 -- comet identity NOT confirmed in comments"),
    "2015-01-26": (dt.datetime(2015, 1, 26, 12, 0, 0), "17P", "index 12 -- comet identity NOT confirmed in comments"),
    "1873-06-25": (dt.datetime(1873, 6, 25, 12, 0, 0), "10P", "index 13 -- explicitly commented '10P/Tempel2'"),
    "1894-04-23": (dt.datetime(1894, 4, 23, 12, 0, 0), "10P", "index 14 -- explicitly commented '10P/Tempel2'"),
    "1904-11-10": (dt.datetime(1904, 11, 10, 12, 0, 0), "10P", "index 15 -- explicitly commented '10P/Tempel2'"),
}


# ---------------------------------------------------------------------------
# 2. Flag profiles -- the distinct (testingmode, withoutholmesgravitation,
#    onlymajorplanets, bvalue-range, duration) combinations actually found
#    across the six files. Physics-irrelevant knobs (testingmode, the
#    disabled Gaussian ejection branch, t1testing) are simply not
#    represented any more because they either never changed the physics in
#    the supplied files (testingmode was always 0) or were dead branches.
# ---------------------------------------------------------------------------


def java_epoch_index(epoch_key: str) -> int | None:
    """The Java `OutburstAndInitialDateIndex` of an epoch key (e.g. 9 for
    "1892-11-04"), read from the "index N" note in OUTBURST_EPOCHS."""
    import re
    m = re.match(r"index (\d+)", OUTBURST_EPOCHS[epoch_key][2])
    return int(m.group(1)) if m else None


def snapshot_specs_from_legacy(legacy_snapshots) -> tuple[SnapshotSpec, ...]:
    """Turn the old Java f/m file definitions (legacy_java.LegacySnapshot)
    into SnapshotSpec entries, so the SAME dates/boxes drive both the
    modern snapshot_<name>.csv files and the legacy b1f10.csv-style files."""
    out = []
    for snap in legacy_snapshots:
        d = snap.date                                     # calendar date of that Julian Day Number
        out.append(SnapshotSpec(
            name=snap.stem,                               # "f10", "m2", ...
            date_utc=dt.datetime(d.year, d.month, d.day, 12, 0, 0),   # noon UTC = the daily output epoch
            observed_ra_deg=snap.ra_center_deg,
            observed_dec_deg=snap.dec_center_deg,
            tolerance_deg=snap.tolerance_deg,
            note="old Java b<bin>%s.csv" % snap.stem,
        ))
    return tuple(out)


def _bins(*names: str):
    """Tiny helper: turn a list of bin-name strings, e.g. "b1_small", into
    the actual ParticleSizeBin objects from config.PARTICLE_BINS, in order.
    Just saves repeating `PARTICLE_BINS["..."]` several times per profile
    below."""
    return tuple(PARTICLE_BINS[n] for n in names)


# Each PROFILE_* below is a plain dict (not a dataclass) holding one
# specific combination of physics settings -- build_scenario() further down
# reads these dicts and turns them into an actual ScenarioConfig.
PROFILE_2007_MAJOR_PLANETS_ONLY = dict(
    # Matches SlaveMode_Holmes2007.java / SlaveMode_Holmes2007_Refactor.java /
    # SlaveMode_Holmes1892_Refactor_original.java (all three are configured
    # identically for this profile).
    gravitating_planets=MAJOR_PLANETS_ONLY,               # Venus/Earth/Mars/Jupiter/Saturn only
    include_comet_self_gravity=True,                       # comet's own gravity is switched ON
    comet_assumed_gm_m3_s2=1.32712440018e20 / 2.0e15,      # old: holmes.setGM(mu/2.E15)
    radiation_model=RadiationModel.REBOUNDX_BETA,           # modern REBOUNDx radiation model
    bins=_bins("b1_small", "b2_medium", "b3_big"),           # all three particle-size classes
    n_particles_per_bin=50,                                   # 50 particles generated per class
    duration_days=6205,   # old `duration = 536112000.` seconds = exactly 6205 days (~17.0 yr)
    # The older 2007 files use a different set of output files (f1..f9,
    # m1..m16, ejtest1..4) that is not modelled here; add SnapshotSpecs if needed.
    snapshots=(),
)

PROFILE_2007_ALL_PLANETS_EXTENDED = dict(
    # Matches SlaveMode_Holmes1892_Refactor_2007.java: same 2007 event, but
    # Mercury/Uranus/Neptune switched back on, more particles, longer run.
    gravitating_planets=ALL_PLANETS,                        # every planet Mercury..Neptune
    include_comet_self_gravity=True,                         # comet's own gravity ON
    comet_assumed_gm_m3_s2=1.32712440018e20 / 2.0e15,
    radiation_model=RadiationModel.REBOUNDX_BETA,
    bins=_bins("b1_small", "b2_medium", "b3_big"),
    n_particles_per_bin=500,                                   # more particles than the profile above
    duration_days=9131,   # old `duration = 788918400.` seconds = exactly 9131 days (~25.0 yr)
    snapshots=snapshot_specs_from_legacy(JAVA_2007_REFACTOR_SNAPSHOTS),
)

PROFILE_1892_NO_COMET_GRAVITY = dict(
    # Matches SlaveMode_Holmes1892_Refactor_1892.java: comet's own gravity
    # switched OFF, and only the smallest particle bin modelled.
    gravitating_planets=ALL_PLANETS,
    include_comet_self_gravity=False,                        # comet's own gravity OFF this time
    comet_assumed_gm_m3_s2=None,                               # no GM needed, since gravity is off
    radiation_model=RadiationModel.REBOUNDX_BETA,
    bins=_bins("b1_small"),                                      # only the smallest size class
    n_particles_per_bin=400,
    # old `duration = 4332096000.` seconds = exactly 50140 days:
    # 1892-11-04 -> 2030-02-14, 50141 daily rows per particle (verified on
    # the real Java output). The earlier "137.0 years" (= 50039.25 days)
    # stopped ~100 days too early and missed the last forecast (2030-02-01).
    duration_days=50140,
    snapshots=snapshot_specs_from_legacy(JAVA_1892_SNAPSHOTS),
)

PROFILE_HISTORICAL_CANDIDATE_DEFAULT = dict(
    # Matches the "current work" file SlaveMode_Holmes1892_Refactor.java:
    # used for exploring newly-added candidate historical outburst epochs,
    # comet's own gravity off, all three particle bins active again.
    gravitating_planets=ALL_PLANETS,
    include_comet_self_gravity=False,
    comet_assumed_gm_m3_s2=None,
    radiation_model=RadiationModel.REBOUNDX_BETA,
    bins=_bins("b1_small", "b2_medium", "b3_big"),
    n_particles_per_bin=200,
    duration_days=48910,  # old `duration = 4225824000.` seconds = exactly 48910 days (~133.9 yr)
    # SlaveMode_Holmes1892_Refactor.java also writes f10..f22/m2/m14 (plus a
    # few extra 2010/2026 dates whose file names are not modelled here).
    snapshots=snapshot_specs_from_legacy(JAVA_1892_SNAPSHOTS),
)


# Name -> profile dict, purely so the command-line entry point
# (run_scenario.py) can accept a plain string instead of importing every
# PROFILE_* constant individually.
PROFILES: dict[str, dict] = {
    "2007_major_planets_only": PROFILE_2007_MAJOR_PLANETS_ONLY,
    "2007_all_planets_extended": PROFILE_2007_ALL_PLANETS_EXTENDED,
    "1892_no_comet_gravity": PROFILE_1892_NO_COMET_GRAVITY,
    "historical_candidate_default": PROFILE_HISTORICAL_CANDIDATE_DEFAULT,
}


def build_scenario(
    epoch_key: str,                       # e.g. "1892-11-04" -- key into OUTBURST_EPOCHS above
    profile: dict,                        # one of the PROFILE_* dicts above
    *,
    name: str | None = None,               # optional custom label; auto-generated if omitted
    ephemeris_source: EphemerisSource = EphemerisSource.JPL_HORIZONS,   # where position data comes from
    direction_mode: EjectionDirectionMode = EjectionDirectionMode.ISOTROPIC_SPHERE,  # ejection direction mode
    random_seed: int | None = None,        # None -> different random particles every run
    output_dir: str = "./output",           # where to write CSV output
    **overrides,                             # any other ScenarioConfig field, to override the profile's own value
) -> ScenarioConfig:
    """Combine one epoch (from OUTBURST_EPOCHS) with one flag profile (one of
    the PROFILE_* dicts above) into a runnable ScenarioConfig.

    Any field can be overridden via **overrides, e.g.:
        build_scenario("1892-11-04", PROFILE_1892_NO_COMET_GRAVITY,
                        particles__n_particles_per_bin=1000)
    """
    # Fail fast with a clear, helpful error if the caller typo'd the epoch key.
    if epoch_key not in OUTBURST_EPOCHS:
        raise KeyError(f"Unknown outburst epoch key: {epoch_key!r}. "
                        f"Known keys: {sorted(OUTBURST_EPOCHS)}")

    # Look up the (date, comet, note) tuple for this epoch key.
    epoch_utc, comet_designation, note = OUTBURST_EPOCHS[epoch_key]

    # Build the particle-generation settings: take the profile's own bins,
    # but let a caller-supplied override replace n_particles_per_bin if
    # they passed one (via `overrides.pop`, which removes it from
    # `overrides` too, so it isn't accidentally applied twice below).
    particles_cfg = ParticleGenerationConfig(
        bins=profile["bins"],
        n_particles_per_bin=overrides.pop("n_particles_per_bin", profile["n_particles_per_bin"]),
        direction_mode=direction_mode,
        random_seed=random_seed,
    )
    # Same idea for the integration/duration settings.
    # duration: an explicit caller override (days or years) wins over the
    # profile's exact Java day count.
    duration_days = overrides.pop("duration_days", None)
    duration_years = overrides.pop("duration_years", None)
    if duration_days is None and duration_years is None:
        duration_days = profile["duration_days"]              # the profile's exact Java length
    integration_kwargs = dict(
        duration_days=duration_days,
        duration_years=duration_years if duration_years is not None else (duration_days or 0) / 365.25,
    )
    # Any IntegrationConfig field can also be overridden directly
    # (e.g. apply_light_time_correction=False for a Java reproduction).
    for field_name in ("apply_light_time_correction", "recenter_to_com", "integrator", "output_cadence_s"):
        if field_name in overrides:
            integration_kwargs[field_name] = overrides.pop(field_name)
    integration_cfg = IntegrationConfig(**integration_kwargs)

    # Assemble the final, complete ScenarioConfig from: the epoch info
    # looked up above, the profile's own physics settings, the two
    # sub-configs just built, and whatever's left in **overrides (any other
    # ScenarioConfig field the caller wants to set directly, e.g. `snapshots=[...]`).
    return ScenarioConfig(
        name=name or f"{epoch_key}_{comet_designation}",   # auto-generate a name if none given
        comet_designation=comet_designation,
        outburst_epoch_utc=epoch_utc,
        ephemeris_source=ephemeris_source,
        gravitating_planets=profile["gravitating_planets"],
        include_comet_self_gravity=profile["include_comet_self_gravity"],
        comet_assumed_gm_m3_s2=profile["comet_assumed_gm_m3_s2"],
        radiation_model=overrides.pop("radiation_model", profile["radiation_model"]),
        particles=particles_cfg,
        snapshots=overrides.pop("snapshots", profile.get("snapshots", ())),
        legacy_epoch_index=overrides.pop("legacy_epoch_index", java_epoch_index(epoch_key)),
        integration=integration_cfg,
        output_dir=output_dir,
        **overrides,   # anything else the caller passed in, applied last
    )
