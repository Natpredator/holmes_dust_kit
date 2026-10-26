#!/usr/bin/env python3
"""
run_scenario.py
================
Command-line entry point: pick a historical outburst epoch + a flag
profile, generate the dust population, run the simulation, and export CSV
output.

This file plays the role the old Java `main()` method played -- but here it
is only ORCHESTRATION (a few dozen lines calling into the real modules),
not physics, not I/O formatting, not particle generation. All of that lives
in the dedicated modules under holmes_dust/.

Examples
--------
Normal (modern) run, 1892 outburst, exact Java length (50140 days):

    python run_scenario.py --epoch 1892-11-04 --profile 1892_no_comet_gravity \\
        --output-dir ./output/holmes_1892 --seed 42

Same run, but ALSO writing the old Java file layout (b1p1.csv ... b1p400.csv,
b1f10.csv ... b1f22.csv, b1m2.csv, b1m14.csv):

    python run_scenario.py --epoch 1892-11-04 --profile 1892_no_comet_gravity \\
        --output-format both --output-dir ./output/holmes_1892 --seed 42

Reproduce an old Java run particle-for-particle (same particles, same comet
start state, same force model, no light-time correction, Java file layout):

    python run_scenario.py --epoch 1892-11-04 --profile 1892_no_comet_gravity \\
        --reproduce-java /path/to/orekit-results --output-dir ./output/repro_1892
    python tools/compare_with_java.py /path/to/orekit-results ./output/repro_1892
"""

from __future__ import annotations  # allows modern type-hint syntax on older Python

import argparse        # for parsing the --epoch/--profile/... command-line flags
import dataclasses     # dataclasses.replace() to adjust a finished ScenarioConfig
import datetime as dt   # for building the list of output-epoch datetimes
import sys               # for sys.exit() and passing argv through to main()

import numpy as np   # for the random-number generator used in particle generation

from holmes_dust import scenarios                       # OUTBURST_EPOCHS, PROFILES, build_scenario
from holmes_dust.config import (                         # enums used to branch on scenario settings
    Body,
    EphemerisSource,
    OutputFormat,
    RadiationModel,
)
from holmes_dust.ephemeris import make_provider             # turns EphemerisSource into a real provider object
from holmes_dust.io_export import ObservationRecord, OutputManager   # modern CSV output machinery
from holmes_dust.legacy_java import (                        # old Java file layout
    LegacyJavaWriter,
    LegacySnapshot,
    julian_day_number,
    particles_from_java_run,
)
from holmes_dust.observations import compute_observation      # RA/DEC/phase-angle calculation
from holmes_dust.particles import generate_particles            # dust-particle population generator
from holmes_dust.simulation import (                             # REBOUND simulation building & stepping
    build_batched_simulation,
    get_dust_state,
    integrate_to,
)


def _build_output_epochs(scenario) -> list[dt.datetime]:
    """Every daily-cadence calendar epoch from t=0 (outburst) to the
    configured run duration, INCLUDING both ends -- equivalent to the old
    code's `for (extrapDate = initialDate; extrapDate <= finalDate;
    extrapDate = extrapDate.shiftedBy(stepT))` loop (50140 days -> 50141
    rows per particle, exactly like the Java files)."""
    cadence_s = scenario.integration.output_cadence_s   # seconds between output rows (usually one day)
    total_s = scenario.duration_seconds()                 # total length of the run, in seconds
    # +1e-6 guards against 4332096000/86400 landing a hair below 50140 in floating point.
    n_steps = int(total_s / cadence_s + 1e-6) + 1           # how many output epochs fit, including both endpoints
    return [
        scenario.outburst_epoch_utc + dt.timedelta(seconds=i * cadence_s)   # t=0, t=+1 day, t=+2 days, ...
        for i in range(n_steps)
    ]


def _legacy_snapshots(scenario) -> list[LegacySnapshot]:
    """The scenario's SnapshotSpecs, expressed as old-style f/m definitions
    (matched on the Julian Day Number of the date, like the Java code)."""
    return [
        LegacySnapshot(stem=s.name, jdn=julian_day_number(s.date_utc),
                       ra_center_deg=s.observed_ra_deg, dec_center_deg=s.observed_dec_deg,
                       tolerance_deg=s.tolerance_deg)
        for s in scenario.snapshots
    ]


def run(scenario, ephemeris=None, dust_particles=None) -> None:
    """Run one scenario end to end.

    `ephemeris` can be injected explicitly (e.g. a test double, or a
    provider already warmed up with a cache) -- if omitted, a fresh
    provider is built from `scenario.ephemeris_source` as usual.
    `dust_particles` can be injected too (e.g. particles rebuilt from an
    old Java run by legacy_java.particles_from_java_run) -- if omitted, a
    fresh random population is generated from the scenario settings.
    """
    if ephemeris is None:
        ephemeris = make_provider(scenario.ephemeris_source, scenario.spk_path)   # Horizons / legacy table / .bsp file

    if dust_particles is None:
        # --- Comet's own state at the outburst epoch: every dust particle is
        #     ejected from here. ---------------------------------------------
        comet_state = ephemeris.get_state(
            Body.COMET, scenario.outburst_epoch_utc, comet_designation=scenario.comet_designation
        )
        # Only actually needed for EjectionDirectionMode.TOWARD_SUN, but cheap
        # enough to always fetch (one extra call at t=0 only).
        sun_state_at_outburst = ephemeris.get_state(Body.SUN, scenario.outburst_epoch_utc)

        # --- Generate the dust population -----------------------------------
        rng = np.random.default_rng(scenario.particles.random_seed)   # None seed -> different particles every run
        dust_particles = generate_particles(
            bins=scenario.particles.bins,
            n_particles_per_bin=scenario.particles.n_particles_per_bin,
            direction_mode=scenario.particles.direction_mode,
            comet_position_m=comet_state.position_m,
            comet_velocity_m_s=comet_state.velocity_m_s,
            sun_position_m=sun_state_at_outburst.position_m,
            rng=rng,
        )
        print(f"[run_scenario] generated {len(dust_particles)} dust particles "
              f"across {len(scenario.particles.bins)} size bin(s)")
    else:
        print(f"[run_scenario] using {len(dust_particles)} supplied dust particles")

    # --- Pre-compute the calendar epochs we will report on, and fetch
    #     Earth's + the Sun's positions at ALL of them in one batched
    #     ephemeris call each (see ephemeris.get_state_series docstring for
    #     why this must be batched rather than queried one day at a time).
    output_epochs = _build_output_epochs(scenario)   # the full list of dates we'll write a CSV row for
    print(f"[run_scenario] {len(output_epochs)} output epochs "
          f"({output_epochs[0].date()} -> {output_epochs[-1].date()}, "
          f"{scenario.integration.output_cadence_s / 86400:.1f}-day cadence), "
          f"radiation model: {scenario.radiation_model.value}, "
          f"light-time: {'on' if scenario.integration.apply_light_time_correction else 'off'}")
    earth_states = ephemeris.get_state_series(Body.EARTH, output_epochs)   # Earth's position on every output date
    sun_states = ephemeris.get_state_series(Body.SUN, output_epochs)        # Sun's position on every output date

    fmt = scenario.output_format
    modern = OutputManager(scenario.output_dir, list(scenario.snapshots)) if fmt in (OutputFormat.MODERN, OutputFormat.BOTH) else None
    legacy = (LegacyJavaWriter(scenario.output_dir, _legacy_snapshots(scenario), scenario.legacy_epoch_index)
              if fmt in (OutputFormat.LEGACY, OutputFormat.BOTH) else None)
    try:
        if modern is not None:
            modern.__enter__()                               # opens trajectory_full.csv
        _run_batched(scenario, ephemeris, dust_particles, output_epochs, earth_states, sun_states, modern, legacy)
    finally:
        # Always flush/close every file, even if the run is interrupted --
        # the old Java program lost its f/m files exactly this way.
        if modern is not None:
            modern.close()
        if legacy is not None:
            legacy.close()
    print(f"[run_scenario] done -- output written to {scenario.output_dir}")


def _run_batched(scenario, ephemeris, dust_particles, output_epochs, earth_states, sun_states,
                 modern: OutputManager | None, legacy: LegacyJavaWriter | None) -> None:
    """One shared simulation for every particle (both radiation models)."""
    handle = build_batched_simulation(scenario, ephemeris, dust_particles)   # build the ONE shared REBOUND sim
    cadence_s = scenario.integration.output_cadence_s   # seconds between output rows
    for step_index, epoch in enumerate(output_epochs):    # walk through every output date, in order
        t_seconds = step_index * cadence_s                  # how far (in seconds) this date is from t=0
        integrate_to(handle, t_seconds)                       # advance the whole simulation to that time
        earth_pos = earth_states[step_index].position_m   # Earth's position on this particular date
        sun_pos = sun_states[step_index].position_m         # Sun's position on this particular date
        for particle in dust_particles:   # once per particle, at this SAME moment in time
            position_m, velocity_m_s = get_dust_state(handle, particle.particle_id)   # read its current state
            observation = compute_observation(
                target_position_m=position_m,
                target_velocity_m_s=velocity_m_s,
                sun_position_m=sun_pos,
                earth_position_m=earth_pos,
                apply_light_time=scenario.integration.apply_light_time_correction,
            )
            if modern is not None:
                modern.write(
                    ObservationRecord(
                        date_utc=epoch,
                        particle_id=particle.particle_id,
                        bin_name=particle.bin_name,
                        radius_mm=particle.radius_mm,
                        beta=particle.beta,
                        ejection_speed_m_s=particle.ejection_speed_m_s,
                        observation=observation,
                        barycentric_position_m=position_m,   # geometric SSB position (Java x/y/z)
                        theta=particle.theta,
                        u=particle.u,
                    )
                )
            if legacy is not None:
                legacy.write(epoch_utc=epoch, particle=particle, barycentric_position_m=position_m,
                             ra_deg=observation.ra_deg, dec_deg=observation.dec_deg,
                             phase_angle_deg=observation.phase_angle_deg)
        if step_index % 100 == 0:   # print progress every 100 steps, not every single one (too noisy otherwise)
            print(f"[run_scenario]   step {step_index}/{len(output_epochs)}  ({epoch.date()})")


def main(argv: list[str] | None = None) -> int:
    # Build the command-line argument parser: this is what makes
    # `python run_scenario.py --epoch ... --profile ...` work.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epoch", required=True, choices=sorted(scenarios.OUTBURST_EPOCHS),
                         help="Which historical outburst epoch to model.")
    parser.add_argument("--profile", required=True, choices=sorted(scenarios.PROFILES),
                         help="Which flag/physics profile to apply.")
    parser.add_argument("--ephemeris-source", default=None,
                         choices=[e.value for e in EphemerisSource],
                         help="Default: jpl_horizons (horizons_with_legacy_comet with --reproduce-java).")
    parser.add_argument("--spk", metavar="FILE.bsp", default=None,
                         help="Local JPL planetary ephemeris (e.g. de440.bsp; needs `pip install jplephem`). "
                              "Implies --ephemeris-source spk_file_with_legacy_comet: fully offline, "
                              "comet start state from the Java table.")
    parser.add_argument("--radiation-model", default=None, choices=[r.value for r in RadiationModel],
                         help="Default: the profile's model (legacy_sun_gm_reduction with --reproduce-java).")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--output-format", default=None, choices=[f.value for f in OutputFormat],
                         help="modern = trajectory_full.csv + snapshot_*.csv; legacy = old Java b1p*.csv/"
                              "b1f*.csv/b1m*.csv layout; both. Default: modern (both with --reproduce-java).")
    parser.add_argument("--n-particles-per-bin", type=int, default=None,
                         help="Override the profile's default particle count per size bin.")
    parser.add_argument("--duration-days", type=float, default=None,
                         help="Override the profile's integration length, in days (exact).")
    parser.add_argument("--duration-years", type=float, default=None,
                         help="Override the profile's integration length, in Julian years.")
    parser.add_argument("--no-light-time", action="store_true",
                         help="Do not apply the light-time correction to RA/DEC (the Java code never did).")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for particle generation (default: nondeterministic).")
    parser.add_argument("--reproduce-java", metavar="JAVA_DIR", default=None,
                         help="Rebuild the particles of an old Java run from its b<bin>p<N>.csv files "
                              "and re-run them with the Java settings (see module docstring).")
    parser.add_argument("--max-particles", type=int, default=None,
                         help="With --reproduce-java: only the first N particle files.")
    parser.add_argument("--only-complete", action="store_true",
                         help="With --reproduce-java: skip particle files that are shorter than a full run "
                              "(the particle the Java program was still computing when it was stopped).")
    args = parser.parse_args(argv)   # actually parse sys.argv (or the argv passed in, e.g. from a test)

    reproducing = args.reproduce_java is not None
    default_source = ("spk_file_with_legacy_comet" if args.spk else
                      "horizons_with_legacy_comet" if reproducing else "jpl_horizons")
    ephemeris_source = EphemerisSource(args.ephemeris_source or default_source)

    # Only pass along overrides the user actually specified on the command
    # line -- an override left at its default (None) means "use whatever
    # the chosen profile already says".
    overrides = {}
    if args.n_particles_per_bin is not None:
        overrides["n_particles_per_bin"] = args.n_particles_per_bin
    if args.duration_days is not None:
        overrides["duration_days"] = args.duration_days
    if args.duration_years is not None:
        overrides["duration_years"] = args.duration_years
    if args.radiation_model is not None:
        overrides["radiation_model"] = RadiationModel(args.radiation_model)
    elif reproducing:
        overrides["radiation_model"] = RadiationModel.LEGACY_SUN_GM_REDUCTION   # the Java force model
    if args.no_light_time or reproducing:
        overrides["apply_light_time_correction"] = False                        # Java RA/DEC had none
    output_format = OutputFormat(args.output_format or ("both" if reproducing else "modern"))

    # Combine the chosen epoch + profile (+ any overrides) into one
    # complete ScenarioConfig.
    scenario = scenarios.build_scenario(
        args.epoch,
        scenarios.PROFILES[args.profile],
        ephemeris_source=ephemeris_source,
        random_seed=args.seed,
        output_dir=args.output_dir,
        output_format=output_format,
        spk_path=args.spk,
        **overrides,
    )

    dust_particles = None
    ephemeris = None
    if reproducing:
        ephemeris = make_provider(scenario.ephemeris_source, scenario.spk_path)
        comet = ephemeris.get_state(Body.COMET, scenario.outburst_epoch_utc, scenario.comet_designation)
        expected_rows = len(_build_output_epochs(scenario))
        dust_particles = particles_from_java_run(
            args.reproduce_java, comet.position_m, comet.velocity_m_s,
            max_particles=args.max_particles, only_complete=args.only_complete, expected_rows=expected_rows,
        )
        if not dust_particles:
            print(f"[run_scenario] no b<bin>p<N>.csv files found in {args.reproduce_java}", file=sys.stderr)
            return 1
        print(f"[run_scenario] rebuilt {len(dust_particles)} particles from {args.reproduce_java} "
              f"(files b?p{dust_particles[0].legacy_number}.csv ... b?p{dust_particles[-1].legacy_number}.csv)")

    run(scenario, ephemeris=ephemeris, dust_particles=dust_particles)   # generate/simulate/export
    return 0          # exit code 0 = success, for the shell/CI to check


if __name__ == "__main__":
    # This block only runs when the file is executed directly (e.g.
    # `python run_scenario.py ...`), NOT when it's imported as a module
    # (e.g. by tests) -- that's what `if __name__ == "__main__":` means.
    sys.exit(main())
