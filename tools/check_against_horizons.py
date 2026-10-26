#!/usr/bin/env python3
"""
check_against_horizons.py
==========================
Independent validation, WITHOUT needing the old Java program's output at
all: for a handful of dates, this compares holmes_dust's own RA/DEC
computation (Cartesian state from Horizons' `vectors()` -> our own
observations.compute_observation) against JPL Horizons' OWN `ephemerides()`
output for the same object/date -- RA/DEC computed independently by NASA's
own, long-established astrometric code, not by anything in this project.

If the two agree closely (they should, to well under an arcminute), that is
strong, independent evidence that the observation-geometry chain in this
package (frame handling, light-time correction, RA/DEC formula) is correct
-- regardless of whether the old Java program's own output is available for
comparison.

Requests Horizons' "astrometric" RA/DEC (QUANTITIES=1), NOT "apparent"
RA/DEC: astrometric is the light-time-corrected geometric position (what
observations.compute_observation in this package also computes), while
"apparent" additionally applies stellar aberration and precession/nutation
corrections that this package does not attempt -- comparing against
"apparent" would show a real but IRRELEVANT few-arcsecond difference here.

Usage:
    python tools/check_against_horizons.py --comet 17P --dates 1892-11-04 1892-11-14 2007-10-24
"""

from __future__ import annotations

import argparse
import datetime as dt

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # project root

from holmes_dust.config import Body  # noqa: E402
from holmes_dust.ephemeris import HorizonsEphemerisProvider, datetime_to_jd  # noqa: E402
from holmes_dust.observations import compute_observation  # noqa: E402


def horizons_own_radec(comet_designation: str, epoch_utc: dt.datetime) -> tuple[float, float]:
    """Ask Horizons to compute astrometric geocentric RA/DEC itself, via
    its OBSERVER ephemeris type -- completely independent of this
    package's own Cartesian-to-RA/DEC code path."""
    from astroquery.jplhorizons import Horizons

    jd = datetime_to_jd(epoch_utc)
    # "500@399" = Earth geocenter (body 399), observer-center code "500" --
    # explicit form, since a bare "500" can be ambiguous about which body.
    query = Horizons(id=comet_designation, location="500@399", id_type="designation", epochs=jd)
    table = query.ephemerides(quantities="1", closest_apparition=True, no_fragments=True)
    row = table[0]
    return float(row["RA"]), float(row["DEC"])


def our_own_radec(provider: HorizonsEphemerisProvider, comet_designation: str, epoch_utc: dt.datetime) -> tuple[float, float, float]:
    """Same computation run() uses: comet + Sun + Earth vectors from
    Horizons, then this package's own compute_observation."""
    comet_state = provider.get_state(Body.COMET, epoch_utc, comet_designation)
    sun_state = provider.get_state(Body.SUN, epoch_utc)
    earth_state = provider.get_state(Body.EARTH, epoch_utc)

    obs = compute_observation(
        target_position_m=comet_state.position_m,
        target_velocity_m_s=comet_state.velocity_m_s,
        sun_position_m=sun_state.position_m,
        earth_position_m=earth_state.position_m,
        apply_light_time=True,
    )
    return obs.ra_deg, obs.dec_deg, obs.light_time_s


def angular_separation_deg(ra1, dec1, ra2, dec2) -> float:
    """Great-circle separation between two RA/DEC points, in degrees."""
    r1, d1, r2, d2 = map(np.radians, [ra1, dec1, ra2, dec2])
    cos_sep = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return float(np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comet", default="17P")
    parser.add_argument("--dates", nargs="+", required=True,
                         help="One or more YYYY-MM-DD dates to check (noon UTC is assumed).")
    args = parser.parse_args()

    provider = HorizonsEphemerisProvider()

    print(f"{'date':12s} {'our RA':>10s} {'our DEC':>10s} {'Horizons RA':>12s} {'Horizons DEC':>13s} "
          f"{'sep (arcsec)':>13s}")
    for date_str in args.dates:
        epoch = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(hour=12)

        our_ra, our_dec, light_time_s = our_own_radec(provider, args.comet, epoch)
        their_ra, their_dec = horizons_own_radec(args.comet, epoch)
        sep_deg = angular_separation_deg(our_ra, our_dec, their_ra, their_dec)

        print(f"{date_str:12s} {our_ra:10.5f} {our_dec:10.5f} {their_ra:12.5f} {their_dec:13.5f} "
              f"{sep_deg * 3600:13.2f}")


if __name__ == "__main__":
    main()
