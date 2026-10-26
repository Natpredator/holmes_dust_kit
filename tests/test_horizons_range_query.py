"""
Unit tests for HorizonsEphemerisProvider.get_state_series's range-query
logic (holmes_dust.ephemeris), run OFFLINE against a fake stand-in for
astroquery's Horizons class -- no network, no real astroquery needed.

This targets the exact class of bug that caused a real 502 error against
the live service (a 366-epoch explicit list producing an oversized request
URL): the step-string formatting Horizons expects ("1d", "6h", ...) and the
fallback-to-chunked-list path when a range request doesn't come back with
one row per requested epoch.

Run with:  python3 -m unittest discover -s tests
"""

import datetime as dt
import unittest

from holmes_dust.config import Body
from holmes_dust.ephemeris import HorizonsEphemerisProvider


class _FakeRow(dict):
    """Mimics one row of an astropy Table returned by Horizons.vectors():
    dict-like access to x/y/z/vx/vy/vz."""
    pass


class _FakeTable(list):
    """Mimics the astropy Table returned by Horizons.vectors() -- just
    needs len() and iteration, both of which list already provides."""
    pass


class _FakeHorizonsQuery:
    """Stand-in for astroquery.jplhorizons.Horizons. Records how it was
    called, and returns a fake vectors() table sized either to match the
    request (default) or to a fixed, possibly-mismatched size (for testing
    the fallback path)."""

    calls: list[dict] = []          # populated by _FakeHorizonsFactory below
    rows_to_return: int | None = None  # None => "match whatever was asked for"

    def __init__(self, id, location, id_type, epochs):  # noqa: A002 (shadows builtin `id`, matches real API)
        self.id = id
        self.location = location
        self.id_type = id_type
        self.epochs = epochs
        _FakeHorizonsQuery.calls.append(
            {"id": id, "location": location, "id_type": id_type, "epochs": epochs}
        )

    def vectors(self, **kwargs):
        if isinstance(self.epochs, dict):
            # Range form: figure out how many rows a real inclusive
            # start/stop/step request would return, unless the test
            # explicitly wants to simulate a mismatch.
            if _FakeHorizonsQuery.rows_to_return is not None:
                n = _FakeHorizonsQuery.rows_to_return
            else:
                n = self._n_rows_for_own_range()
        elif isinstance(self.epochs, list):
            n = len(self.epochs)
        else:
            # A single bare Julian Date float, as get_state() (singular)
            # sends -- always exactly one row.
            n = 1

        return _FakeTable([_FakeRow(x=1.0, y=2.0, z=3.0, vx=0.01, vy=0.02, vz=0.03) for _ in range(n)])

    def _n_rows_for_own_range(self) -> int:
        start = dt.datetime.strptime(self.epochs["start"], "%Y-%m-%d %H:%M")
        stop = dt.datetime.strptime(self.epochs["stop"], "%Y-%m-%d %H:%M")
        step_str = self.epochs["step"]
        unit = step_str[-1]
        amount = int(step_str[:-1])
        step_s = {"d": 86400, "h": 3600, "m": 60}[unit] * amount
        return int((stop - start).total_seconds() // step_s) + 1


def _make_provider_with_fake_horizons() -> HorizonsEphemerisProvider:
    """Build a HorizonsEphemerisProvider WITHOUT going through its normal
    __init__ (which does `from astroquery.jplhorizons import Horizons` and
    would fail here, since astroquery is not installed in this test
    environment) -- inject the fake query class directly instead.

    cache_dir=None disables the on-disk cache entirely for these tests, so
    every call in a test actually reaches the fake and the existing
    call-count assertions below stay meaningful (a real run enables the
    cache by default -- see test_horizons_caching.py for cache-specific
    coverage).
    """
    provider = object.__new__(HorizonsEphemerisProvider)
    provider._Horizons = _FakeHorizonsQuery
    provider.cache_dir = None
    return provider


class TestHorizonsRangeQuery(unittest.TestCase):
    def setUp(self):
        _FakeHorizonsQuery.calls = []
        _FakeHorizonsQuery.rows_to_return = None
        self.provider = _make_provider_with_fake_horizons()

    def _daily_epochs(self, n_days: int) -> list[dt.datetime]:
        start = dt.datetime(2007, 10, 24, 12, 0, 0)
        return [start + dt.timedelta(days=i) for i in range(n_days)]

    def test_evenly_spaced_epochs_use_one_range_request(self):
        epochs = self._daily_epochs(366)
        result = self.provider.get_state_series(Body.EARTH, epochs)

        self.assertEqual(len(result), 366)
        # Exactly one call to Horizons, not one per epoch and not chunked.
        self.assertEqual(len(_FakeHorizonsQuery.calls), 1)
        self.assertIsInstance(_FakeHorizonsQuery.calls[0]["epochs"], dict)

    def test_range_request_uses_day_step_string(self):
        epochs = self._daily_epochs(10)
        self.provider.get_state_series(Body.EARTH, epochs)
        self.assertEqual(_FakeHorizonsQuery.calls[0]["epochs"]["step"], "1d")

    def test_comet_range_request_sets_apparition_disambiguation(self):
        # Can't directly inspect vector_kwargs from here (they're passed to
        # .vectors(), not stored on the query object), so this test instead
        # confirms the comet path at least succeeds end-to-end with the
        # fake and requests with id_type="designation".
        epochs = self._daily_epochs(5)
        result = self.provider.get_state_series(Body.COMET, epochs, comet_designation="17P")
        self.assertEqual(len(result), 5)
        self.assertEqual(_FakeHorizonsQuery.calls[0]["id_type"], "designation")
        self.assertEqual(_FakeHorizonsQuery.calls[0]["id"], "17P")

    def test_mismatched_row_count_falls_back_to_chunked_list(self):
        _FakeHorizonsQuery.rows_to_return = 365  # one short of the 366 requested
        epochs = self._daily_epochs(366)

        result = self.provider.get_state_series(Body.EARTH, epochs)

        # Still gets a correct, fully-aligned result...
        self.assertEqual(len(result), 366)
        # ...but via the chunked fallback: the first call is the (mismatched)
        # range attempt (dict-typed epochs), and every call AFTER that is a
        # chunked list request -- more than one call in total.
        self.assertGreater(len(_FakeHorizonsQuery.calls), 1)
        self.assertIsInstance(_FakeHorizonsQuery.calls[0]["epochs"], dict)
        self.assertTrue(all(isinstance(c["epochs"], list) for c in _FakeHorizonsQuery.calls[1:]))

    def test_irregular_epochs_go_straight_to_chunked_path(self):
        epochs = [
            dt.datetime(2007, 10, 24, 12, 0, 0),
            dt.datetime(2007, 10, 25, 12, 0, 0),
            dt.datetime(2007, 10, 27, 12, 0, 0),  # gap here -- not evenly spaced
        ]
        result = self.provider.get_state_series(Body.EARTH, epochs)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(isinstance(c["epochs"], list) for c in _FakeHorizonsQuery.calls))

    def test_single_epoch_uses_plain_get_state(self):
        epochs = self._daily_epochs(1)
        result = self.provider.get_state_series(Body.EARTH, epochs)
        self.assertEqual(len(result), 1)

    def test_empty_epoch_list_returns_empty(self):
        result = self.provider.get_state_series(Body.EARTH, [])
        self.assertEqual(result, [])

    def test_chunked_fallback_stays_under_a_safe_batch_size(self):
        # Force the fallback path (mismatch) and confirm it actually
        # chunks instead of sending all 366 in one oversized list request
        # -- this is the exact scenario that produced a real 502 error.
        _FakeHorizonsQuery.rows_to_return = 1  # force mismatch on the range attempt
        epochs = self._daily_epochs(366)

        self.provider.get_state_series(Body.EARTH, epochs)

        # First call was the (failed/mismatched) range attempt; the rest
        # are chunked list calls, each well under 366 epochs.
        list_calls = [c for c in _FakeHorizonsQuery.calls if isinstance(c["epochs"], list)]
        self.assertTrue(all(len(c["epochs"]) <= 50 for c in list_calls))


if __name__ == "__main__":
    unittest.main()
