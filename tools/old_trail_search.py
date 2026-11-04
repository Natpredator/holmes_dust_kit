#!/usr/bin/env python3
"""
old_trail_search.py
====================
Exploratory tool for the "old trail" hypothesis.

Background: the REBOUND reproduction of 10P/Tempel 2's 1873/1894/1904
perihelion dust (see `holmes_dust/scenarios.py`'s `10p_1873`/`10p_1894`/
`10p_1904` profiles, validated against the supplied Java/Orekit runs) lands
about 190 degrees away in RA from Jorma Ryske's actual 2026 observations of
the trail, on all three of his dates. That is not "slightly off" -- it is
the opposite side of the sky -- so dust from any recent perihelion passage
under 10P's CURRENT orbit is very unlikely to explain what he sees.

Markku's second idea: the real trail may be much OLDER -- large, slow grains
ejected many returns ago that have decoupled from the comet's current,
repeatedly-perturbed orbit. 10P/Tempel 2 is a Jupiter-family comet
(Tisserand parameter ~3), and JFC orbits are CHAOTIC on a timescale of
decades to a couple of centuries because of repeated close Jupiter
encounters. A single backward integration over thousands of years does not
reconstruct "the" true orbit history -- this script integrates a handful of
CLONES (small perturbations of the nominal state) alongside the nominal
trajectory specifically to make that divergence visible and quantified,
rather than silently trusting one trajectory.

What this script does
----------------------
PASS 1 (one backward integration, Sun + 8 planets + comet clones, REBOUND/
IAS15, no dust): walks 10P's current best-fit orbit backward up to
`--years-back` years, detects every perihelion passage of the NOMINAL clone
along the way (local minima of heliocentric distance), and logs how far the
clones have spread from the nominal trajectory over time (the chaos/
Lyapunov horizon, in AU). Near each of `--test-years-ago`, the full system
state (Sun + 8 planets + nominal comet) at the nearest actual perihelion is
saved as a release candidate.

PASS 2 (one FRESH forward integration per candidate): seeds a new
simulation at that ancient perihelion from the saved state, generates a
swarm of large, slow grains there (same beta(radius)/ejection_speed(radius)
physics as the rest of this package -- see holmes_dust/particles.py),
and integrates forward (Sun + 8 planets + REBOUNDx radiation pressure) all
the way to the present, computing geocentric RA/DEC on Jorma's three 2026
observation dates. Results are ranked by angular separation from his points.

This is deliberately a SCAN/triage tool: coarse backward cadence, modest
particle counts. A promising candidate epoch is worth a much more careful,
larger-N follow-up (e.g. feeding it into run_scenario.py-style daily output)
-- not a final answer by itself.

Usage (from the project root):
    python tools/old_trail_search.py
    python tools/old_trail_search.py --years-back 20000 --cadence-days 25 \\
        --test-years-ago 500 1000 2000 4000 7000 11000 16000 20000 \\
        --n-dust 20 --out-dir ./output/old_trail
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # project root

from holmes_dust.config import ALL_PLANETS, Body, EjectionDirectionMode, PARTICLE_BINS  # noqa: E402
from holmes_dust import constants  # noqa: E402
from holmes_dust.ephemeris import HorizonsEphemerisProvider  # noqa: E402
from holmes_dust.observations import compute_observation  # noqa: E402
from holmes_dust.particles import generate_particles  # noqa: E402
from holmes_dust.simulation import AU_M, DAY_S, MSUN_KG, SPEED_OF_LIGHT_AU_PER_DAY  # noqa: E402

COMET_DESIGNATION = "10P"

# Jorma's three observed sky positions of the trail (RA, DEC, degrees),
# read from the Proton Drive drop (2026_06.rtf / 2026_07.rtf / 2026_08.rtf).
# All three are ~190 deg away in RA from the modelled 1873/1894/1904-release
# clouds (see README / project notes) -- this is the mismatch this script
# is trying to explain.
JORMA_OBSERVATIONS = [
    ("2026-06-15", dt.datetime(2026, 6, 15, 12, 0, 0), 310.28785, -9.28507),
    ("2026-07-15", dt.datetime(2026, 7, 15, 12, 0, 0), 323.32304, -16.82935),
    ("2026-08-15", dt.datetime(2026, 8, 15, 12, 0, 0), 333.32711, -29.79141),
]


def _add_massive(sim, position_m, velocity_m_s, mass_kg):
    sim.add(
        m=mass_kg / MSUN_KG,
        x=position_m[0] / AU_M, y=position_m[1] / AU_M, z=position_m[2] / AU_M,
        vx=velocity_m_s[0] * DAY_S / AU_M, vy=velocity_m_s[1] * DAY_S / AU_M, vz=velocity_m_s[2] * DAY_S / AU_M,
    )
    return sim.N - 1


def _add_massless(sim, position_m, velocity_m_s):
    return _add_massive(sim, position_m, velocity_m_s, 0.0)


def _state_m(p):
    """REBOUND particle -> (position_m, velocity_m_s), SI."""
    return (np.array([p.x, p.y, p.z]) * AU_M,
            np.array([p.vx, p.vy, p.vz]) * AU_M / DAY_S)


def fetch_reference_state(reference_epoch: dt.datetime):
    """Sun + 8 planets + 10P nucleus at the reference epoch, from JPL
    Horizons (closest-apparition orbital solution for the comet)."""
    provider = HorizonsEphemerisProvider()
    print(f"[old_trail] fetching Sun + planets + {COMET_DESIGNATION} nucleus at {reference_epoch} from Horizons ...")
    bodies = {}
    bodies[Body.SUN] = provider.get_state(Body.SUN, reference_epoch)
    for planet in ALL_PLANETS:
        bodies[planet] = provider.get_state(planet, reference_epoch)
    bodies[Body.COMET] = provider.get_state(Body.COMET, reference_epoch, comet_designation=COMET_DESIGNATION)
    return provider, bodies


def build_backward_sim(bodies, n_clones: int, clone_pos_sigma_km: float, clone_vel_sigma_mm_s: float, seed: int):
    """Sun + 8 planets (massive) + (1 nominal + n_clones perturbed) 10P
    clones (massless, non-perturbing -- the nucleus's own gravity on the
    planets is negligible at this precision and is irrelevant to the
    dust-release question this script asks)."""
    import rebound

    sim = rebound.Simulation()
    sim.units = ("AU", "day", "Msun")
    sim.integrator = "ias15"

    body_index = {}
    body_index[Body.SUN] = _add_massive(sim, bodies[Body.SUN].position_m, bodies[Body.SUN].velocity_m_s,
                                         constants.GM_M3_S2[Body.SUN] / constants.G_SI)
    for planet in ALL_PLANETS:
        body_index[planet] = _add_massive(sim, bodies[planet].position_m, bodies[planet].velocity_m_s,
                                           constants.GM_M3_S2[planet] / constants.G_SI)
    sim.N_active = sim.N   # only Sun + planets are gravity sources from here on

    rng = np.random.default_rng(seed)
    sigma_pos_m = clone_pos_sigma_km * 1000.0
    sigma_vel_m_s = clone_vel_sigma_mm_s / 1000.0
    comet = bodies[Body.COMET]
    clone_indices = [_add_massless(sim, comet.position_m, comet.velocity_m_s)]   # clone 0 = nominal, unperturbed
    for _ in range(n_clones):
        dp = rng.normal(scale=sigma_pos_m, size=3)
        dv = rng.normal(scale=sigma_vel_m_s, size=3)
        clone_indices.append(_add_massless(sim, comet.position_m + dp, comet.velocity_m_s + dv))

    return sim, body_index, clone_indices


def backward_scan(sim, body_index, clone_indices, years_back: float, cadence_days: float,
                   test_years_ago: list[float]):
    """Walk the simulation backward from t=0, detecting the nominal clone's
    perihelion passages and the clone swarm's divergence. Returns
    (perihelia, divergence_log, candidates) where candidates maps each
    requested years-ago target to a full system snapshot at the nearest
    detected perihelion: {years_ago: {"t_days": t, "date": approx date,
    "distance_au": q, Body.SUN: (pos,vel), <planet>: (pos,vel), ...,
    "comet": (pos,vel)}}.
    """
    sun_idx = body_index[Body.SUN]
    nominal_idx = clone_indices[0]
    target_t = -years_back * 365.25

    perihelia: list[tuple[float, float]] = []            # (t_days, distance_au) of each detected minimum
    divergence_log: list[tuple[float, float]] = []        # (years_ago, max clone separation from nominal, AU)

    prev_prev_d = prev_d = prev_t = None
    last_logged_years = -1e9
    sorted_targets = sorted(set(test_years_ago))
    remaining_targets = list(sorted_targets)
    candidates: dict[float, dict] = {}

    t = 0.0
    step = 0
    while t > target_t:
        t_next = max(t - cadence_days, target_t)
        sim.integrate(t_next)
        t = sim.t

        sun = sim.particles[sun_idx]
        nom = sim.particles[nominal_idx]
        d = ((nom.x - sun.x) ** 2 + (nom.y - sun.y) ** 2 + (nom.z - sun.z) ** 2) ** 0.5

        is_min = prev_prev_d is not None and prev_d is not None and prev_d < prev_prev_d and prev_d < d
        if is_min:
            perihelia.append((prev_t, prev_d))
            years_ago_here = -prev_t / 365.25
            # Did we just pass (going further back) one of the requested targets?
            while remaining_targets and remaining_targets[0] <= years_ago_here + cadence_days / 365.25:
                target = remaining_targets.pop(0)
                snapshot = {"t_days": prev_t, "distance_au": prev_d, "years_ago_actual": years_ago_here,
                            Body.COMET: (np.array([nom.x, nom.y, nom.z]) * AU_M,
                                         np.array([nom.vx, nom.vy, nom.vz]) * AU_M / DAY_S)}
                for body, idx in body_index.items():
                    p = sim.particles[idx]
                    snapshot[body] = (np.array([p.x, p.y, p.z]) * AU_M, np.array([p.vx, p.vy, p.vz]) * AU_M / DAY_S)
                candidates[target] = snapshot
                print(f"[old_trail] candidate ~{target:.0f} yr ago -> nearest perihelion "
                      f"{years_ago_here:.1f} yr ago (q={prev_d:.3f} AU)")

        prev_prev_d, prev_d, prev_t = prev_d, d, t

        years_ago = -t / 365.25
        if years_ago - last_logged_years >= 100.0 or t <= target_t:
            coords = np.array([[sim.particles[i].x, sim.particles[i].y, sim.particles[i].z] for i in clone_indices])
            sep = float(np.linalg.norm(coords[1:] - coords[0], axis=1).max()) if len(coords) > 1 else 0.0
            divergence_log.append((years_ago, sep))
            last_logged_years = years_ago

        step += 1
        if step % 2000 == 0:
            print(f"[old_trail] backward scan: {years_ago:8.0f} / {years_back:.0f} yr ago "
                  f"({len(perihelia)} perihelia found so far)")

    # Any still-unmatched targets (deeper than the last detected perihelion): use the last one found.
    for target in remaining_targets:
        if perihelia:
            t_last, d_last = perihelia[-1]
            print(f"[old_trail] candidate ~{target:.0f} yr ago -> beyond scan depth, "
                  f"using deepest perihelion found ({-t_last / 365.25:.1f} yr ago) instead")
    return perihelia, divergence_log, candidates


def release_and_propagate(snapshot: dict, n_dust: int, bin_name: str, seed: int,
                           target_dates: list[dt.datetime], reference_epoch: dt.datetime):
    """Fresh forward simulation: Sun + 8 planets + a dust swarm released at
    `snapshot`'s ancient comet state, integrated to the present and then to
    each of `target_dates`. Returns a dict date -> list of (ra_deg, dec_deg)
    per particle, using the SIMULATION's own Sun position (self-consistent
    with the propagated trajectory) and Earth positions supplied by the
    caller (ephemeris -- the nucleus/dust propagation here does not include
    Earth as a gravitating mass, matching the rest of this package).
    """
    import rebound
    import reboundx

    sim = rebound.Simulation()
    sim.units = ("AU", "day", "Msun")
    sim.integrator = "ias15"

    body_index = {}
    for body in (Body.SUN, *ALL_PLANETS):
        pos_m, vel_m_s = snapshot[body]
        gm = constants.GM_M3_S2[body]
        body_index[body] = _add_massive(sim, pos_m, vel_m_s, gm / constants.G_SI)
    sim.N_active = sim.N

    comet_pos_m, comet_vel_m_s = snapshot[Body.COMET]
    rng = np.random.default_rng(seed)
    particles = generate_particles(
        bins=[PARTICLE_BINS[bin_name]], n_particles_per_bin=n_dust,
        direction_mode=EjectionDirectionMode.ISOTROPIC_SPHERE,
        comet_position_m=comet_pos_m, comet_velocity_m_s=comet_vel_m_s, rng=rng,
    )
    dust_index = {}
    for particle in particles:
        dust_index[particle.particle_id] = _add_massless(sim, particle.initial_position_m, particle.initial_velocity_m_s)

    rebx = reboundx.Extras(sim)
    radiation_force = rebx.load_force("radiation_forces")
    rebx.add_force(radiation_force)
    radiation_force.params["c"] = SPEED_OF_LIGHT_AU_PER_DAY
    sim.particles[body_index[Body.SUN]].params["radiation_source"] = 1
    for particle in particles:
        sim.particles[dust_index[particle.particle_id]].params["beta"] = particle.beta

    t0_days = snapshot["t_days"]   # this ancient epoch's time, relative to the PASS-1 reference epoch
    results = {}
    for target_date in target_dates:
        t_target = (target_date - reference_epoch).total_seconds() / DAY_S   # days since reference epoch
        sim.integrate(t_target - t0_days)   # sim clock starts at 0 = t0_days; advance to this target
        sun_pos_m, _ = _state_m(sim.particles[body_index[Body.SUN]])
        rows = []
        for particle in particles:
            pos_m, vel_m_s = _state_m(sim.particles[dust_index[particle.particle_id]])
            rows.append((pos_m, vel_m_s))
        results[target_date] = (sun_pos_m, rows, particles)
    return results


def format_ancient_date(reference_epoch: dt.datetime, t_days: float) -> str:
    """Calendar label for a release epoch `t_days` days (usually negative)
    from `reference_epoch`. Falls back to a plain year count when the real
    date would be before 1 CE (Python's datetime.MINYEAR) or otherwise out
    of datetime's range -- this is a PRINTING/labelling concern only, the
    simulation itself always works in t_days, never real calendar dates,
    for anything beyond the reference epoch."""
    try:
        return str((reference_epoch + dt.timedelta(days=t_days)).date())
    except OverflowError:
        proleptic_year = reference_epoch.year + t_days / 365.25
        if proleptic_year > 0:
            return f"~{proleptic_year:.0f} CE"
        return f"~{-proleptic_year:.0f} BCE"


def angular_sep_deg(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    cos_sep = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-epoch", default="2026-08-02",
                         help="Recent, well-determined epoch to anchor the backward integration (YYYY-MM-DD). "
                              "Default: 10P's most recent perihelion per the Seiichi Yoshida page.")
    parser.add_argument("--years-back", type=float, default=20000.0)
    parser.add_argument("--cadence-days", type=float, default=25.0,
                         help="Sampling interval for perihelion detection during the backward scan.")
    parser.add_argument("--n-clones", type=int, default=6,
                         help="Extra perturbed comet clones, for the chaos/divergence report.")
    parser.add_argument("--clone-pos-sigma-km", type=float, default=50.0)
    parser.add_argument("--clone-vel-sigma-mm-s", type=float, default=2.0)
    parser.add_argument("--test-years-ago", type=float, nargs="+",
                         default=[500, 1000, 2000, 4000, 7000, 11000, 16000, 20000])
    parser.add_argument("--n-dust", type=int, default=20)
    parser.add_argument("--bin", default="b3_big", choices=sorted(PARTICLE_BINS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="./output/old_trail")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    reference_epoch = dt.datetime.strptime(args.reference_epoch, "%Y-%m-%d").replace(hour=12)
    test_years_ago = [y for y in args.test_years_ago if y <= args.years_back]

    provider, bodies = fetch_reference_state(reference_epoch)

    # Detecting a local-minimum perihelion needs one more sample PAST it, so
    # a requested target at (or very near) the scan boundary would otherwise
    # never be confirmed. Scan a little deeper than requested to avoid that
    # edge case; perihelia.csv/divergence log end up with a bit of bonus
    # depth too, which is harmless.
    scan_years_back = args.years_back + max(60.0, 3 * 5.5)   # +60 yr, >= ~3 of 10P's ~5.5 yr orbital periods
    print(f"[old_trail] PASS 1: backward scan, {args.years_back:.0f} yr requested "
          f"(scanning to {scan_years_back:.0f} yr for edge margin), {args.n_clones} clones, "
          f"{args.cadence_days:.0f}-day cadence ...")
    sim, body_index, clone_indices = build_backward_sim(
        bodies, args.n_clones, args.clone_pos_sigma_km, args.clone_vel_sigma_mm_s, args.seed)
    perihelia, divergence_log, candidates = backward_scan(
        sim, body_index, clone_indices, scan_years_back, args.cadence_days, test_years_ago)

    div_path = os.path.join(args.out_dir, "clone_divergence.csv")
    with open(div_path, "w") as f:
        f.write("years_ago,max_clone_separation_au\n")
        for years_ago, sep in divergence_log:
            f.write(f"{years_ago:.1f},{sep:.6e}\n")
    print(f"[old_trail] wrote {div_path} ({len(divergence_log)} rows) -- "
          f"clone spread at depth: {divergence_log[-1][1]:.4f} AU after {divergence_log[-1][0]:.0f} yr "
          f"(started at ~{args.clone_pos_sigma_km} km)")

    peri_path = os.path.join(args.out_dir, "perihelia.csv")
    with open(peri_path, "w") as f:
        f.write("years_ago,t_days,distance_au\n")
        for t_days, d_au in perihelia:
            f.write(f"{-t_days / 365.25:.2f},{t_days:.1f},{d_au:.4f}\n")
    print(f"[old_trail] wrote {peri_path} ({len(perihelia)} perihelion passages over {args.years_back:.0f} yr)")

    if not candidates:
        print("[old_trail] no candidates captured -- nothing to propagate forward. Exiting.")
        return

    target_dates = [d for _, d, _, _ in JORMA_OBSERVATIONS]
    earth_states = {d: provider.get_state(Body.EARTH, d) for d in target_dates}   # fetched once, reused below
    summary_path = os.path.join(args.out_dir, "summary.csv")
    print(f"[old_trail] PASS 2: releasing {args.n_dust} '{args.bin}' particles at {len(candidates)} candidate "
          f"epochs and propagating to {len(target_dates)} target dates ...")

    with open(summary_path, "w") as f:
        f.write("target_years_ago,actual_years_ago,release_date_approx,obs_date,obs_ra_deg,obs_dec_deg,"
                "model_median_ra_deg,model_median_dec_deg,median_sep_deg,min_sep_deg,n_particles\n")
        ranked = []
        for target_years_ago in sorted(candidates):
            snapshot = candidates[target_years_ago]
            release_date = format_ancient_date(reference_epoch, snapshot["t_days"])
            print(f"[old_trail]   candidate ~{target_years_ago:.0f} yr ago "
                  f"(release ~{release_date}, q={snapshot['distance_au']:.3f} AU) -- integrating forward ...")
            results = release_and_propagate(snapshot, args.n_dust, args.bin, args.seed + int(target_years_ago),
                                             target_dates, reference_epoch)
            overall_seps = []
            for label, obs_date, obs_ra, obs_dec in JORMA_OBSERVATIONS:
                sun_pos_m, rows, particles = results[obs_date]
                earth_state = earth_states[obs_date]   # one Horizons lookup per date, shared by every particle
                ra_list, dec_list, sep_list = [], [], []
                for (pos_m, vel_m_s), particle in zip(rows, particles):
                    obs = compute_observation(target_position_m=pos_m, target_velocity_m_s=vel_m_s,
                                               sun_position_m=sun_pos_m, earth_position_m=earth_state.position_m,
                                               apply_light_time=True)
                    ra_list.append(obs.ra_deg)
                    dec_list.append(obs.dec_deg)
                    sep_list.append(angular_sep_deg(obs.ra_deg, obs.dec_deg, obs_ra, obs_dec))
                med_ra, med_dec = float(np.median(ra_list)), float(np.median(dec_list))
                med_sep, min_sep = float(np.median(sep_list)), float(np.min(sep_list))
                overall_seps.append(med_sep)
                f.write(f"{target_years_ago:.0f},{snapshot['years_ago_actual']:.1f},{release_date},"
                        f"{label},{obs_ra:.3f},{obs_dec:.3f},{med_ra:.3f},{med_dec:.3f},"
                        f"{med_sep:.2f},{min_sep:.2f},{len(ra_list)}\n")
            ranked.append((float(np.mean(overall_seps)), target_years_ago, release_date))

    print(f"[old_trail] wrote {summary_path}")
    print("\n[old_trail] ranked by mean separation across all 3 of Jorma's dates (smaller = better match):")
    for mean_sep, target_years_ago, release_date in sorted(ranked):
        print(f"    ~{target_years_ago:>6.0f} yr ago (release ~{release_date}): mean sep {mean_sep:7.2f} deg")


if __name__ == "__main__":
    main()
