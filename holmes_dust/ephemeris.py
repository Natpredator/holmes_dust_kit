"""
ephemeris.py
============
Everything about getting a Cartesian state vector (position + velocity) for
the Sun, a planet, or a comet, at a given UTC epoch, in the ICRF/J2000
frame, expressed in SI units (metres, metres/second), centred on the Solar
System Barycenter (SSB) -- exactly the frame/units the old Java code used
(it is stated explicitly in the source comments: "ICRF/J2000 Solar System
Barycenter and Earth mean equator and equinox of reference epoch").

Two backends are provided behind one common interface:

* HorizonsEphemerisProvider (recommended)
    Queries JPL Horizons live via ``astroquery``. Works for ANY body and
    ANY epoch -- this is what lets the new pipeline model a brand new
    candidate outburst date just by typing a date, instead of manually
    copy-pasting a Horizons vector table into source code the way the old
    program did for every single scenario.

* LegacyTableEphemerisProvider (validation only)
    Reproduces the exact hard-coded Sun/comet position vectors that were
    typed into the Java ``if (OutburstAndInitialDateIndex == N)`` blocks.
    Use this ONLY to reproduce/validate a historical Java run byte-for-byte;
    it deliberately does NOT know about planets, because the old Java
    program never hard-coded planet vectors either -- it pulled them
    transparently from Orekit's bundled JPL DE ephemeris kernel at
    runtime. There is no equivalent "kernel" bundled here, so planet
    positions always come from HorizonsEphemerisProvider (or another
    ephemeris source you plug in) even when comet/Sun vectors come from
    the legacy table.
"""

from __future__ import annotations  # allows modern type-hint syntax (e.g. `str | None`) on older Python

import dataclasses          # for the @dataclasses.dataclass decorator (see StateVector)
import datetime as dt        # for the `datetime` type used throughout this file
import json                   # for reading/writing the on-disk ephemeris cache as JSON
import os                      # for creating the cache folder and building file paths
import time                     # for time.sleep() between retry attempts
from typing import Protocol      # Protocol lets us describe an interface without inheritance

import numpy as np   # for building/handling the 3-element position/velocity arrays

from .config import Body   # enum identifying Sun/planets/comet/Earth

# Astronomical Unit in metres and a day in seconds -- used to convert
# Horizons' native AU / (AU per day) units into the SI units this whole
# pipeline works in.
AU_M = 1.495978707e11    # 1 astronomical unit, in metres
DAY_S = 86400.0            # 1 day, in seconds


# A plain, read-only container: one body's position+velocity at one moment.
@dataclasses.dataclass(frozen=True)
class StateVector:
    """A Cartesian position+velocity state at one epoch.

    position_m / velocity_m_s are length-3 numpy arrays in SI units,
    expressed in the ICRF/J2000-equatorial frame, centred on the Solar
    System Barycenter (SSB) -- matching the old Java program's frame.
    """
    epoch_utc: dt.datetime    # the calendar date/time this state belongs to
    position_m: np.ndarray     # length-3 array: x, y, z, in metres
    velocity_m_s: np.ndarray    # length-3 array: vx, vy, vz, in metres/second

    def __post_init__(self):
        # __post_init__ runs automatically right after the dataclass's
        # auto-generated constructor finishes -- used here to validate and
        # normalise the two array fields.
        #
        # Defensive check: catches unit-mixups (e.g. accidentally passing
        # AU instead of metres) early, before they silently corrupt a run.
        # `object.__setattr__` is needed instead of plain `self.x = ...`
        # because this dataclass is frozen (read-only) -- this is the one
        # sanctioned way to fill in a frozen field during initialisation.
        object.__setattr__(self, "position_m", np.asarray(self.position_m, dtype=float))
        object.__setattr__(self, "velocity_m_s", np.asarray(self.velocity_m_s, dtype=float))
        if self.position_m.shape != (3,) or self.velocity_m_s.shape != (3,):
            # Something passed in the wrong number of components -- fail
            # loudly now rather than producing confusing errors later.
            raise ValueError("StateVector position/velocity must be length-3 vectors")


# A Protocol is like an interface: it says WHAT METHODS a class must have to
# count as an "EphemerisProvider", without requiring that class to actually
# inherit from this one. Both HorizonsEphemerisProvider and
# LegacyTableEphemerisProvider satisfy this shape below.
class EphemerisProvider(Protocol):
    """Common interface every ephemeris backend must implement."""

    def get_state(self, body: Body, epoch_utc: dt.datetime, comet_designation: str | None = None) -> StateVector:
        """Return the SSB-centred ICRF/J2000 state of `body` at `epoch_utc`.

        `comet_designation` (e.g. "17P", "10P") is only used when
        `body is Body.COMET`, so the same provider instance can serve
        different target comets across different scenarios.
        """
        ...   # "..." here just means "Protocol doesn't implement this itself"

    def get_state_series(
        self, body: Body, epochs_utc: list[dt.datetime], comet_designation: str | None = None
    ) -> list[StateVector]:
        """Same as get_state, but for many epochs at once.

        This exists because run_scenario.py needs Earth's (and the Sun's)
        position at EVERY daily output epoch of a run that can span more
        than a century -- calling get_state() once per day would mean tens
        of thousands of individual network round-trips. A real ephemeris
        backend should implement this as one (or a few, chunked) batched
        query instead of a naive per-epoch loop.
        """
        ...


# ---------------------------------------------------------------------------
# JPL Horizons backend (recommended / primary)
# ---------------------------------------------------------------------------

# Horizons body IDs for the barycenters/planets we care about. Using the
# *barycenter* IDs (1-8) for the giant/rocky planets mirrors what the old
# Orekit `CelestialBodyFactory.getEarthMoonBarycenter()` etc. effectively
# provided for gravity purposes; Body.EARTH (399) is kept separate because
# it is used as the *observer* location for RA/DEC, where the actual planet
# center (not the Earth-Moon barycenter) is the correct choice -- this
# matches the old code's explicit use of `CelestialBodyFactory.getEarth()`
# for the observer vector.
_HORIZONS_IDS: dict[Body, str] = {
    Body.SUN: "10",                    # Horizons major-body id for the Sun
    Body.MERCURY: "1",                  # Mercury barycenter
    Body.VENUS: "2",                     # Venus barycenter
    Body.EARTH_MOON_BARYCENTER: "3",      # Earth+Moon combined barycenter
    Body.EARTH: "399",                     # Earth itself (for observer position only)
    Body.MARS: "4",                         # Mars barycenter
    Body.JUPITER: "5",                       # Jupiter barycenter
    Body.SATURN: "6",                         # Saturn barycenter
    Body.URANUS: "7",                          # Uranus barycenter
    Body.NEPTUNE: "8",                          # Neptune barycenter
}


class HorizonsEphemerisProvider:
    """Live JPL Horizons ephemeris lookup via `astroquery.jplhorizons`.

    Requires the optional dependency `astroquery` (see requirements.txt)
    and network access at run time.

    ROBUSTNESS NOTES (added after hitting these in practice against the
    real service on a 50040-epoch, 137-year request):

    * A single request spanning tens of thousands of epochs -- even using
      the compact start/stop/step RANGE syntax, which keeps the request
      URL itself short -- can still make the Horizons SERVER take longer
      than astroquery's default 30-second read timeout to compute and
      return the response. This class (a) raises that timeout, and
      (b) splits a range request into moderate-sized sub-chunks
      (RANGE_CHUNK_SIZE epochs each) so no single request has to wait on
      a huge response.
    * Any single HTTP request (range or list) can transiently fail
      (timeout, connection reset, brief server-side hiccup) with no
      relation to the query itself. Every network call here goes through
      a retry-with-backoff wrapper instead of propagating the first
      failure immediately.
    * A multi-hour ephemeris fetch (thousands of sub-requests for a very
      long scenario) losing ALL its progress to one late network blip is
      unacceptable, so every successfully-fetched sub-chunk is written to
      a small on-disk JSON cache (`cache_dir`, default `.horizons_cache/`
      in the current working directory) immediately. Re-running the exact
      same scenario after a crash re-uses everything already cached and
      only fetches what's still missing. Pass `cache_dir=None` to disable
      caching entirely (e.g. in tests).
    """

    # Sub-chunk size for the compact start/stop/step RANGE request. Even
    # though the range syntax keeps the request URL short regardless of
    # how many epochs are covered, the SERVER still has to compute and
    # return that many rows in one response -- see class docstring.
    RANGE_CHUNK_SIZE = 3000   # how many epochs to ask for in one range request

    # Sub-chunk size for the explicit-epoch-list fallback path. Kept small
    # (see the module's earlier revision history: a 366-epoch single list
    # request already came back as a 502/timeout against the real
    # service), and now additionally protected by the retry wrapper.
    LIST_CHUNK_SIZE = 50   # how many epochs to ask for in one list-style request

    # Per-request retry policy shared by every Horizons call in this class.
    MAX_ATTEMPTS = 5                # try up to this many times before giving up
    INITIAL_RETRY_DELAY_S = 3.0     # wait this long before the FIRST retry
    RETRY_BACKOFF_FACTOR = 2.5      # multiply the wait by this much after each failed retry

    def __init__(self, cache_dir: str | None = ".horizons_cache"):
        # Import lazily (i.e. only when a HorizonsEphemerisProvider is
        # actually constructed) so the rest of the package (particles,
        # config, observations, tests) works even if astroquery is not
        # installed at all.
        from astroquery.jplhorizons import Horizons  # noqa: F401 (import-checked here)
        self._Horizons = Horizons   # remember the class so other methods can call it

        self.cache_dir = cache_dir           # where to read/write the on-disk cache, or None to disable it
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)   # create the cache folder if it's missing

        # Raise astroquery/Horizons' read timeout above its ~30s default --
        # observed in practice to be too short for a response covering
        # thousands of epochs at once. Best-effort: older/newer astroquery
        # versions may expose this configuration differently, so this is
        # wrapped defensively rather than assumed to always succeed.
        try:
            from astroquery.jplhorizons import conf as _horizons_conf
            _horizons_conf.timeout = 180   # seconds
        except Exception:
            pass   # if this fails, we just keep astroquery's own default timeout

    # -- single-epoch lookup -------------------------------------------------

    def get_state(self, body: Body, epoch_utc: dt.datetime, comet_designation: str | None = None) -> StateVector:
        jd = datetime_to_jd(epoch_utc)                                      # convert the date to a Julian Day number
        query, vector_kwargs = self._build_query(body, jd, comet_designation)  # build the Horizons request
        table = self._fetch_vectors_with_retry(query, vector_kwargs)         # actually run it (with retries)
        row = table[0]                                                       # only one row, since we asked for one date

        # Horizons returns AU / AU-per-day; convert to this project's SI units.
        position_m = np.array([row["x"], row["y"], row["z"]]) * AU_M
        velocity_m_s = np.array([row["vx"], row["vy"], row["vz"]]) * AU_M / DAY_S
        return StateVector(epoch_utc=epoch_utc, position_m=position_m, velocity_m_s=velocity_m_s)

    # -- many-epoch lookup, cached + chunked + retried -----------------------

    def get_state_series(
        self, body: Body, epochs_utc: list[dt.datetime], comet_designation: str | None = None
    ) -> list[StateVector]:
        """Fetch many epochs efficiently, robustly, and resumably.

        See the class docstring for the reasoning behind caching, chunking
        and retrying. This package always generates evenly-spaced daily
        epoch lists (see run_scenario._build_output_epochs), so on a fresh
        (uncached) call the fast RANGE path is used; anything left missing
        after that (including on a resumed/partially-cached run, where the
        remaining gaps are not necessarily evenly spaced any more) is
        fetched through the slower, always-correct explicit-list path.
        """
        if len(epochs_utc) == 0:
            return []   # nothing was asked for -- nothing to do
        if len(epochs_utc) == 1:
            # Just one date -- no need for any of the batching machinery below.
            return [self.get_state(body, epochs_utc[0], comet_designation)]

        cache = self._load_cache(body, comet_designation)               # read whatever is already on disk
        missing = [e for e in epochs_utc if e.isoformat() not in cache]  # which requested dates aren't cached yet

        if missing:   # only do any network work if something is actually missing
            all_missing = len(missing) == len(epochs_utc)   # True on a totally fresh (uncached) call
            # Work out the time gap between consecutive missing dates, to
            # check whether they are evenly spaced (needed for the fast
            # range-request path below).
            deltas_s = [(missing[i + 1] - missing[i]).total_seconds() for i in range(len(missing) - 1)]
            evenly_spaced = bool(deltas_s) and deltas_s[0] > 0 and all(
                abs(d - deltas_s[0]) < 1e-6 for d in deltas_s
            )

            range_fully_succeeded = False
            if all_missing and evenly_spaced:
                # Common case: nothing cached yet, dates are evenly spaced
                # (as they always are for a fresh run) -- try the fast,
                # compact range-request path first.
                range_fully_succeeded = self._fetch_range_chunked(
                    body, missing, deltas_s[0], comet_designation, cache
                )

            if not range_fully_succeeded:
                # Either the range path wasn't applicable (irregular gaps,
                # e.g. resuming a partially-cached run) or it partially
                # failed -- fetch whatever is STILL missing via the slower
                # but always-correct explicit-list path.
                still_missing = [e for e in epochs_utc if e.isoformat() not in cache]
                if still_missing:
                    self._fetch_list_chunked(body, still_missing, comet_designation, cache)

        # By this point every requested epoch is guaranteed to be in the
        # cache (or an exception was already raised trying to get it
        # there) -- so just read the final answer straight out of it, in
        # the SAME order the caller originally asked for.
        return [self._cache_entry_to_state(cache[e.isoformat()], e) for e in epochs_utc]

    # -- internals: query construction & retry -------------------------------

    def _build_query(self, body: Body, epochs, comet_designation: str | None):
        """`epochs` is either a single JD float, a list of JD floats, or a
        Horizons range dict ({"start", "stop", "step"}). Returns
        (query, vector_kwargs) ready for self._fetch_vectors_with_retry."""
        vector_kwargs = dict(refplane="earth")   # ICRF-aligned reference plane, not Horizons' ecliptic default
        if body is Body.COMET:
            if not comet_designation:
                # Can't query a comet without knowing WHICH comet.
                raise ValueError("comet_designation is required when body is Body.COMET")
            query = self._Horizons(id=comet_designation, location="@0", id_type="designation", epochs=epochs)
            # See get_state()'s original docstring note: a periodic comet
            # has one orbital-solution record per apparition, so this
            # disambiguates which one Horizons should use.
            vector_kwargs.update(closest_apparition=True, no_fragments=True)
        else:
            horizons_id = _HORIZONS_IDS[body]   # look up the numeric Horizons id for this body
            query = self._Horizons(id=horizons_id, location="@0", id_type=None, epochs=epochs)
        return query, vector_kwargs   # caller passes these straight to _fetch_vectors_with_retry

    def _fetch_vectors_with_retry(self, query, vector_kwargs):
        delay = self.INITIAL_RETRY_DELAY_S      # how long to wait before the next retry
        last_exc: Exception | None = None        # remembers the most recent failure, for the final raise
        for attempt in range(1, self.MAX_ATTEMPTS + 1):   # try up to MAX_ATTEMPTS times
            try:
                return query.vectors(**vector_kwargs)   # the actual network call -- success returns immediately
            except Exception as exc:  # noqa: BLE001 -- deliberately broad: network/server
                # failures here can surface as several different exception
                # types (requests.ReadTimeout, ConnectionError, HTTPError,
                # astroquery-internal errors); all of them are equally
                # worth retrying, and anything NOT worth retrying (e.g. a
                # genuine bad request) will simply keep failing and be
                # re-raised once attempts are exhausted.
                last_exc = exc
                if attempt < self.MAX_ATTEMPTS:
                    print(f"[ephemeris] Horizons request failed (attempt {attempt}/{self.MAX_ATTEMPTS}): "
                          f"{exc!r} -- retrying in {delay:.0f}s")
                    time.sleep(delay)                        # wait before trying again
                    delay *= self.RETRY_BACKOFF_FACTOR         # wait longer next time (exponential backoff)
        assert last_exc is not None   # we only get here after exhausting every attempt
        raise last_exc                 # give up -- re-raise the last failure we saw

    # -- internals: the two fetch strategies ---------------------------------

    def _fetch_range_chunked(
        self, body: Body, epochs: list[dt.datetime], step_s: float, comet_designation: str | None, cache: dict
    ) -> bool:
        """Fetch `epochs` (evenly spaced, step_s apart) via one or more
        compact start/stop/step RANGE requests, writing each successful
        sub-chunk straight into `cache` (persisted to disk immediately).
        Returns True only if EVERY sub-chunk succeeded and lined up
        exactly with what was requested; False otherwise (the caller then
        falls back to the always-correct explicit-list path for whatever
        is still missing from `cache`)."""
        n = len(epochs)     # total number of epochs to fetch
        all_ok = True        # becomes False if any sub-chunk fails or mismatches

        # Horizons wants the step size as an integer + unit letter, e.g.
        # "1d" for one day. Figure out the best-fitting unit for this
        # scenario's cadence.
        if step_s % DAY_S == 0:
            step_str = f"{int(step_s / DAY_S)}d"
        elif step_s % 3600.0 == 0:
            step_str = f"{int(step_s / 3600.0)}h"
        else:
            step_str = f"{int(round(step_s / 60.0))}m"

        # Walk through `epochs` in blocks of RANGE_CHUNK_SIZE, one Horizons
        # request per block.
        for start_idx in range(0, n, self.RANGE_CHUNK_SIZE):
            sub = epochs[start_idx:start_idx + self.RANGE_CHUNK_SIZE]   # this block's epochs
            print(f"[ephemeris] fetching {body.value} {start_idx + 1}-{start_idx + len(sub)} of {n} "
                  f"({sub[0].date()}..{sub[-1].date()}, range request)")

            # Build the compact {"start", "stop", "step"} range spec for
            # Horizons, covering exactly this block.
            epochs_spec = {
                "start": sub[0].strftime("%Y-%m-%d %H:%M"),
                "stop": sub[-1].strftime("%Y-%m-%d %H:%M"),
                "step": step_str,
            }
            query, vector_kwargs = self._build_query(body, epochs_spec, comet_designation)

            try:
                table = self._fetch_vectors_with_retry(query, vector_kwargs)
            except Exception as exc:
                # This block's request failed even after all retries --
                # note the failure and move on; the caller will pick up
                # whatever's still missing via the list-fallback path.
                print(f"[ephemeris] range sub-request permanently failed after retries: {exc!r} "
                      f"-- will fall back to per-epoch fetching for this span")
                all_ok = False
                continue

            if len(table) != len(sub):
                # Horizons didn't return exactly one row per requested
                # epoch (can happen at the edges of a range) -- don't trust
                # this block's alignment; skip it and let the fallback
                # path fetch it epoch-by-epoch instead.
                print(f"[ephemeris] range sub-request returned {len(table)} rows for {len(sub)} "
                      f"requested epochs -- falling back to per-epoch fetching for this span")
                all_ok = False
                continue

            # Success: convert every returned row to SI units and store it
            # in the cache, keyed by that row's own epoch (isoformat string).
            for row, epoch in zip(table, sub):
                position_m = np.array([row["x"], row["y"], row["z"]]) * AU_M
                velocity_m_s = np.array([row["vx"], row["vy"], row["vz"]]) * AU_M / DAY_S
                cache[epoch.isoformat()] = [*position_m.tolist(), *velocity_m_s.tolist()]
            self._save_cache(body, comet_designation, cache)   # persist to disk immediately, not at the very end

        return all_ok   # True only if every single block above succeeded and matched

    def _fetch_list_chunked(
        self, body: Body, epochs: list[dt.datetime], comet_designation: str | None, cache: dict
    ) -> None:
        """Fetch `epochs` (any spacing) by listing individual Julian Dates,
        in small batches, writing each successful batch straight into
        `cache` (persisted to disk immediately)."""
        n = len(epochs)   # total number of epochs to fetch this way
        for start_idx in range(0, n, self.LIST_CHUNK_SIZE):   # walk through in small batches
            chunk = epochs[start_idx:start_idx + self.LIST_CHUNK_SIZE]
            print(f"[ephemeris] fetching {body.value} {start_idx + 1}-{start_idx + len(chunk)} of {n} "
                  f"(list fallback)")

            jds = [datetime_to_jd(t) for t in chunk]   # convert each date to a Julian Day number
            query, vector_kwargs = self._build_query(body, jds, comet_designation)
            table = self._fetch_vectors_with_retry(query, vector_kwargs)   # no try/except: let a persistent
            # failure propagate up and stop the run, since this is already
            # the last-resort fallback path.

            for row, epoch in zip(table, chunk):
                position_m = np.array([row["x"], row["y"], row["z"]]) * AU_M
                velocity_m_s = np.array([row["vx"], row["vy"], row["vz"]]) * AU_M / DAY_S
                cache[epoch.isoformat()] = [*position_m.tolist(), *velocity_m_s.tolist()]
            self._save_cache(body, comet_designation, cache)   # persist after every batch, not just at the end

    # -- internals: on-disk cache --------------------------------------------

    def _cache_path(self, body: Body, comet_designation: str | None) -> str:
        # One cache file per body (or per comet designation, since two
        # different comets must never share a file).
        name = body.value if body is not Body.COMET else f"comet_{comet_designation}"
        return os.path.join(self.cache_dir, f"{name}.json")

    def _load_cache(self, body: Body, comet_designation: str | None) -> dict:
        if not self.cache_dir:
            return {}   # caching disabled -- behave as if nothing is ever cached
        path = self._cache_path(body, comet_designation)
        if not os.path.exists(path):
            return {}   # no cache file yet for this body -- nothing cached
        try:
            with open(path) as f:
                return json.load(f)   # parse the JSON dict of {iso_date: [x,y,z,vx,vy,vz]}
        except Exception:
            # A corrupted/partial cache file should never break a run --
            # just treat it as empty and re-fetch everything.
            return {}

    def _save_cache(self, body: Body, comet_designation: str | None, cache: dict) -> None:
        if not self.cache_dir:
            return   # caching disabled -- nothing to save
        path = self._cache_path(body, comet_designation)
        tmp_path = path + ".tmp"
        # Write-to-temp-then-rename so a crash mid-write can never leave a
        # half-written, unreadable cache file behind.
        with open(tmp_path, "w") as f:
            json.dump(cache, f)
        os.replace(tmp_path, path)   # atomic on POSIX systems -- either the old or the new file, never a mix

    @staticmethod
    def _cache_entry_to_state(entry: list, epoch: dt.datetime) -> StateVector:
        # `entry` is the 6-number list [x, y, z, vx, vy, vz] as stored by
        # the fetch methods above -- turn it back into a proper StateVector.
        arr = np.array(entry, dtype=float)
        return StateVector(epoch_utc=epoch, position_m=arr[:3], velocity_m_s=arr[3:])


def datetime_to_jd(epoch_utc: dt.datetime) -> float:
    """Plain-Python UTC calendar date/time -> Julian Date, no external
    dependency required (standard Fliegel & Van Flandern algorithm).

    Public (no leading underscore) because io_export.py also needs it to
    reproduce the old code's `JulianFields.JULIAN_DAY.getFrom(localdate)`
    column, and duplicating this algorithm in two places would risk them
    drifting apart.
    """
    y, m, d = epoch_utc.year, epoch_utc.month, epoch_utc.day   # calendar year/month/day
    # Fraction of the day already elapsed, from the hour/minute/second.
    frac_day = (epoch_utc.hour + epoch_utc.minute / 60.0 + epoch_utc.second / 3600.0) / 24.0
    # The rest is the standard Fliegel & Van Flandern integer-arithmetic
    # formula for the Julian Day Number (JDN) of a Gregorian calendar date.
    a = (14 - m) // 12
    y2 = y + 4800 - a
    m2 = m + 12 * a - 3
    jdn = d + (153 * m2 + 2) // 5 + 365 * y2 + y2 // 4 - y2 // 100 + y2 // 400 - 32045
    return jdn + frac_day - 0.5   # -0.5 because JD 0.0 begins at NOON, not midnight


# Backward-compatible private alias (kept in case other code imported the
# old name during development).
_datetime_to_jd = datetime_to_jd


# ---------------------------------------------------------------------------
# Legacy hard-coded table -- ported from the Java `if (OutburstAndInitialDateIndex
# == N) { x = ...; y = ...; ... }` blocks, for validation only.
# ---------------------------------------------------------------------------

# key: epoch date (UTC, always 12:00:00 in the old code) -> comet position/
# velocity (m, m/s) and Sun position (m). The old code never stored Sun
# VELOCITY (it was only ever used to build a unit direction vector for the
# unused "eject toward Sun" mode), so it is not available here either.
#
# Every entry below has the same shape:
#   "YYYY-MM-DD": dict(comet_pos=(x, y, z), comet_vel=(vx, vy, vz), sun_pos=(x, y, z))
# with all values in SI (metres, metres/second), copied verbatim from the
# old Java source files' hard-coded numbers.
#
# NOTE: indices 13/14/15 in the newest Java file are explicitly commented
# "10P/Tempel2" (a *different* comet from Holmes) -- see scenarios.py for
# the corrected assignment. This table only stores numbers; the comet
# identity/label lives in scenarios.OUTBURST_EPOCHS.
LEGACY_STATES: dict[str, dict[str, tuple[float, float, float]]] = {
    "2007-10-20": dict(comet_pos=(2.149708659901957e11, 1.985464748640370e11, 2.148410467255068e11),
                        comet_vel=(-1.269529581336842e4, 1.527289475433542e4, 9.647949489296433e3),
                        sun_pos=(9.186143988528989e7, 6.750014214546776e8, 2.798428003353999e8)),
    "2007-10-21": dict(comet_pos=(2.138717639149751e11, 1.998639926181459e11, 2.156723976352332e11),
                        comet_vel=(-1.274679777933582e4, 1.522518579177643e4, 9.596291459987105e3),
                        sun_pos=(9.092061258155614e7, 6.751632906029922e8, 2.799293525779953e8)),
    "2007-10-22": dict(comet_pos=(2.127682338300585e11, 2.011773843166096e11, 2.164992871173197e11),
                        comet_vel=(-1.279779640811368e4, 1.517738473614347e4, 9.544676051995710e3),
                        sun_pos=(8.997981764124756e7, 6.753240865420705e8, 2.800154465395136e8)),
    "2007-10-23": dict(comet_pos=(2.116603191274868e11, 2.024866421392310e11, 2.173217189446428e11),
                        comet_vel=(-1.284829393225311e4, 1.512949474458570e4, 9.493105352051524e3),
                        sun_pos=(8.903905049642708e7, 6.754838184226374e8, 2.801010870999491e8)),
    "2007-10-24": dict(comet_pos=(2.105480630054963e11, 2.037917585370573e11, 2.181396970688875e11),
                        comet_vel=(-1.289829260529057e4, 1.508151893876524e4, 9.441581412545085e3),
                        sun_pos=(8.809830568679403e7, 6.756424954462152e8, 2.801862792656101e8)),
    "2007-10-25": dict(comet_pos=(2.094315084667401e11, 2.050927262293164e11, 2.189532256175906e11),
                        comet_vel=(-1.294779470093767e4, 1.503346040491757e4, 9.390106251769435e3),
                        sun_pos=(8.715757676632144e7, 6.758001267542561e8, 2.802710281199869e8)),
    "2007-10-26": dict(comet_pos=(2.083106983165800e11, 2.063895382003590e11, 2.197623088912049e11),
                        comet_vel=(-1.299680251228356e4, 1.498532219392086e4, 9.338681854165580e3),
                        sun_pos=(8.621685622891926e7, 6.759567212907231e8, 2.803553387584577e8)),
    "2007-10-27": dict(comet_pos=(2.071856751614460e11, 2.076821876966071e11, 2.205669513601859e11),
                        comet_vel=(-1.304531835100950e4, 1.493710732137414e4, 9.287310170572539e3),
                        sun_pos=(8.527613546399308e7, 6.761122876411887e8, 2.804392162073454e8)),
    "2007-10-28": dict(comet_pos=(2.060564814072638e11, 2.089706682235093e11, 2.213671576620989e11),
                        comet_vel=(-1.309334454661598e4, 1.488881876768405e4, 9.235993118481266e3),
                        sun_pos=(8.433540475146330e7, 6.762668338557850e8, 2.805226653302500e8)),
    "1892-11-04": dict(comet_pos=(2.744228527724833e11, 1.354994421045742e11, 1.829999385978398e11),
                        comet_vel=(-9.275429660245276e3, 1.601729393482798e4, 1.248215654577809e4),
                        sun_pos=(-2.218174288568667e8, -3.515433552603099e8, -1.478811819394609e8)),
    "1893-01-16": dict(comet_pos=(2.025539877509266e11, 2.277497148985030e11, 2.508464560507500e11),
                        comet_vel=(-1.312842601308980e4, 1.313771410033890e4, 9.028092855712938e3),
                        sun_pos=(-1.855748813186603e8, -4.019889783256958e8, -1.705960650178921e8)),
    "1899-07-04": dict(comet_pos=(3.145724611263309e11, 2.316354028384432e10, 8.949056855507714e10),
                        comet_vel=(-2.613717418634613e3, 1.781615431901904e4, 1.556512644576749e4),
                        sun_pos=(6.607203517507074e8, 7.687810548483531e8, 3.118209453505847e8)),
    "2015-01-26": dict(comet_pos=(4.012815999349772e10, 3.387872427325165e11, 2.895184585672163e11),
                        comet_vel=(-1.607039510286238e4, 8.810472967226959e3, 3.479910203303075e3),
                        sun_pos=(4.397658773499823e8, -9.048143391681249e7, -6.198681206272462e7)),
    "1873-06-25": dict(comet_pos=(1.227229030868569e11, -1.449522145950082e11, -6.702370892488015e10),
                        comet_vel=(2.466810946652150e4, 2.037913678691285e4, 1.166430796704814e3),
                        sun_pos=(3.715695879290334e8, -1.062399250667559e8, -5.391486596793431e7)),
    "1894-04-23": dict(comet_pos=(1.229355738180368e11, -1.462767438648659e11, -6.761058779160886e10),
                        comet_vel=(2.463776370315925e4, 2.027096894171970e4, 1.149801140437672e3),
                        sun_pos=(1.775582190505921e8, -5.980208998131936e8, -2.645375795985709e8)),
    "1904-11-10": dict(comet_pos=(1.257957841231100e11, -1.492975501442130e11, -6.941417166350786e10),
                        comet_vel=(2.414920340623019e4, 2.002001325664141e4, 1.225948015608240e3),
                        sun_pos=(-9.239989344502321e8, -1.763466118280641e8, -4.547022087513360e7)),
}


class LegacyTableEphemerisProvider:
    """Serves the hard-coded Sun/comet vectors above, for validating a new
    run against a historical Java output at one of the supported epochs.

    Deliberately raises for any other body/epoch instead of silently
    returning something wrong -- see module docstring for why planets are
    out of scope for this backend.
    """

    def get_state(self, body: Body, epoch_utc: dt.datetime, comet_designation: str | None = None) -> StateVector:
        key = epoch_utc.strftime("%Y-%m-%d")   # look up by plain calendar date, ignoring time-of-day
        if key not in LEGACY_STATES:
            # This date simply isn't one of the ones hard-coded in the old
            # Java files -- say so clearly instead of guessing.
            raise KeyError(
                f"No legacy table entry for {key}. LegacyTableEphemerisProvider only "
                f"covers the historical epochs already present in the old Java code: "
                f"{sorted(LEGACY_STATES)}"
            )
        entry = LEGACY_STATES[key]   # the dict of comet_pos/comet_vel/sun_pos for this date
        if body is Body.COMET:
            return StateVector(epoch_utc, np.array(entry["comet_pos"]), np.array(entry["comet_vel"]))
        if body is Body.SUN:
            # No Sun velocity was ever recorded in the old code (see comment
            # above LEGACY_STATES) -- returning zero velocity is only valid
            # for the direction-to-Sun computation in particles.py, NOT for
            # adding the Sun as a moving body in a real N-body simulation.
            return StateVector(epoch_utc, np.array(entry["sun_pos"]), np.zeros(3))
        # Anything else (a planet) is simply not something this backend can
        # provide -- see the class/module docstrings for why.
        raise NotImplementedError(
            f"LegacyTableEphemerisProvider has no data for {body!r}. The old Java "
            f"program obtained planet positions from Orekit's bundled JPL DE "
            f"ephemeris kernel, not from hard-coded numbers -- use "
            f"HorizonsEphemerisProvider (or another real ephemeris source) for "
            f"planets even when validating against a legacy comet/Sun epoch."
        )

    def get_state_series(
        self, body: Body, epochs_utc: list[dt.datetime], comet_designation: str | None = None
    ) -> list[StateVector]:
        # The legacy table only ever holds ONE epoch per historical
        # scenario (the outburst date itself), so a "series" is trivial
        # here -- provided mainly to satisfy the EphemerisProvider
        # interface for validation scripts that call it generically.
        return [self.get_state(body, epoch, comet_designation) for epoch in epochs_utc]


# NAIF id chains (SSB -> ... -> body) inside a JPL planetary SPK kernel
# (de405.bsp, de421.bsp, de440.bsp, ...). Same bodies as _HORIZONS_IDS.
_SPK_CHAINS: dict[Body, list[tuple[int, int]]] = {
    Body.SUN: [(0, 10)],
    Body.MERCURY: [(0, 1)],
    Body.VENUS: [(0, 2)],
    Body.EARTH_MOON_BARYCENTER: [(0, 3)],
    Body.EARTH: [(0, 3), (3, 399)],          # SSB -> Earth-Moon barycentre -> Earth centre
    Body.MARS: [(0, 4)],
    Body.JUPITER: [(0, 5)],
    Body.SATURN: [(0, 6)],
    Body.URANUS: [(0, 7)],
    Body.NEPTUNE: [(0, 8)],
}


class SpkEphemerisProvider:
    """OFFLINE Sun/planet/Earth states from a JPL planetary ephemeris file
    (.bsp, e.g. de440.bsp covering 1550-2650, or de405.bsp 1600-2200),
    read with the `jplephem` package -- no network, no Horizons timeouts.

    The numbers are the same JPL DE solutions Horizons itself serves (and
    the same kind of file Orekit read for the old Java program), so results
    agree with the Horizons provider to far below anything that matters for
    dust. Epochs are passed as Julian Dates exactly like the Horizons
    provider does (interpreted as TDB, the ephemeris time scale; the
    TDB-UTC difference of about a minute is irrelevant here).

    The comet is NOT in a planetary kernel: its state comes from
    `comet_provider` (default: the hard-coded Java table, i.e. this provider
    then reproduces the Java start conditions completely offline).
    """

    def __init__(self, path: str, comet_provider: EphemerisProvider | None = None):
        from jplephem.spk import SPK   # optional dependency: pip install jplephem
        self.path = path
        self.kernel = SPK.open(path)
        self.comet_provider = comet_provider if comet_provider is not None else LegacyTableEphemerisProvider()

    def _compute(self, body: Body, jd):
        pos_km = 0.0
        vel_km_day = 0.0
        for center, target in _SPK_CHAINS[body]:
            p, v = self.kernel[center, target].compute_and_differentiate(jd)
            pos_km = pos_km + p
            vel_km_day = vel_km_day + v
        return np.asarray(pos_km) * 1000.0, np.asarray(vel_km_day) * 1000.0 / 86400.0   # -> m, m/s

    def get_state(self, body: Body, epoch_utc: dt.datetime, comet_designation: str | None = None) -> StateVector:
        if body is Body.COMET:
            return self.comet_provider.get_state(body, epoch_utc, comet_designation)
        pos, vel = self._compute(body, datetime_to_jd(epoch_utc))
        return StateVector(epoch_utc, pos, vel)

    def get_state_series(
        self, body: Body, epochs_utc: list[dt.datetime], comet_designation: str | None = None
    ) -> list[StateVector]:
        if body is Body.COMET:
            return self.comet_provider.get_state_series(body, epochs_utc, comet_designation)
        jd = np.array([datetime_to_jd(e) for e in epochs_utc])   # one vectorised evaluation for all epochs
        pos, vel = self._compute(body, jd)                         # shapes (3, N)
        return [StateVector(e, pos[:, i].copy(), vel[:, i].copy()) for i, e in enumerate(epochs_utc)]


class LegacyCometHybridProvider:
    """Comet state at the outburst epoch from the hard-coded Java table,
    EVERYTHING else (Sun, planets, Earth, and the comet at any other date)
    from a real ephemeris provider (normally Horizons).

    Why this exists: to reproduce an old Java run particle-for-particle.
    The Java program started every particle from a comet state vector that
    was typed into the source code. The Horizons orbit solution for 17P
    that astroquery picks today is a slightly different solution (the
    difference at 1892-11-04 is ~0.0024 AU, visible as a ~0.09 deg RA/DEC
    offset of the whole cloud). Starting from the Java numbers removes that
    difference, so the remaining differences are purely numerical/physical.
    """

    def __init__(self, base: EphemerisProvider | None = None):
        self.base = base if base is not None else HorizonsEphemerisProvider()   # real ephemeris for everything else
        self.legacy = LegacyTableEphemerisProvider()                               # the Java hard-coded comet states

    def get_state(self, body: Body, epoch_utc: dt.datetime, comet_designation: str | None = None) -> StateVector:
        # Only the comet, and only at a date present in the Java table, is
        # served from the table; anything else is passed straight through.
        if body is Body.COMET and epoch_utc.strftime("%Y-%m-%d") in LEGACY_STATES:
            return self.legacy.get_state(body, epoch_utc, comet_designation)
        return self.base.get_state(body, epoch_utc, comet_designation)

    def get_state_series(
        self, body: Body, epochs_utc: list[dt.datetime], comet_designation: str | None = None
    ) -> list[StateVector]:
        # Series are only ever requested for the Earth and the Sun (for
        # RA/DEC and phase angle) -- always from the real ephemeris.
        return self.base.get_state_series(body, epochs_utc, comet_designation)


def make_provider(source, spk_path: str | None = None) -> EphemerisProvider:
    """Small factory so callers can go from config.EphemerisSource straight
    to a provider instance without importing both enum and classes.
    `spk_path` is required for EphemerisSource.SPK_FILE_WITH_LEGACY_COMET."""
    from .config import EphemerisSource  # local import avoids a cycle at module load time

    if source == EphemerisSource.JPL_HORIZONS:
        return HorizonsEphemerisProvider()   # live network-backed provider
    if source == EphemerisSource.LEGACY_TABLE:
        return LegacyTableEphemerisProvider()   # hard-coded validation-only provider
    if source == EphemerisSource.HORIZONS_WITH_LEGACY_COMET:
        return LegacyCometHybridProvider()      # Java comet start state + Horizons for everything else
    if source == EphemerisSource.SPK_FILE_WITH_LEGACY_COMET:
        if not spk_path:
            raise ValueError("EphemerisSource.SPK_FILE_WITH_LEGACY_COMET needs spk_path (e.g. de440.bsp)")
        return SpkEphemerisProvider(spk_path)   # offline planets from a .bsp file + Java comet start state
    raise ValueError(f"Unknown ephemeris source: {source!r}")   # caller passed something unexpected
