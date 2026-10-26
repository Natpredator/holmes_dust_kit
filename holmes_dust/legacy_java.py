"""
legacy_java.py
==============
Everything needed to READ, WRITE and REPRODUCE the output files of the old
Java/Orekit program (SlaveMode_Holmes*.java).

What the old output looked like (verified against a real Java run folder,
`orekit-results`, 1892-11-04 outburst, bin b1):

* One file per particle, ``b<bin>p<N>.csv`` (bin 1..3, N = 1..maxparticles),
  NO header row, one line per output epoch (every 86400 s = daily, from the
  outburst epoch to the end of the run), 15 comma+space separated fields:

      JD, "step K", date, x, y, z, RA_hms, DEC_dms, RA_deg, DEC_deg,
      radius_mm, theta, u, epoch_index, phase_angle_deg

  - JD       : integer Julian Day NUMBER of the (local, in practice UTC)
               calendar date -- `JulianFields.JULIAN_DAY`, not a fractional JD.
  - step K   : a counter. In the Java code the SAME `cpt++` was also used by
               the console progress print every ~101 rows, so in real files K
               jumps by 2 once every 102 rows (e.g. the last row of a 50141-row
               file says "step 50632"). It carries no information. This module
               writes the clean row counter 1, 2, 3, ...
  - date     : Orekit AbsoluteDate.toString(), UTC, e.g.
               "1892-11-04T12:00:00.000Z". Because the Java loop added 86400 s
               of continuous (atomic) time per step, the UTC label slowly
               drifts through the leap seconds: "11:59:47" by 1974,
               "11:59:23" from 2017 on. The physical instants are ~37 s apart
               at most -- irrelevant for dust -- so this module writes the
               true UTC noon label ("12:00:00.000Z") and matches rows by JD.
  - x, y, z  : particle position in the ICRF, SOLAR-SYSTEM-BARYCENTRIC frame,
               in AU, using the Java constant 6.68458712E-12 m->AU.
               (NOT geocentric -- the modern trajectory_full.csv gives
               geocentric_*; the barycentric_* columns match this.)
  - RA/DEC   : geocentric (Earth centre), ICRF, NO light-time correction,
               RA normalised to [0, 360).
  - radius, theta, u : the particle's random radius (mm) and the two random
               numbers of its ejection direction -- constant on every row.
               These three numbers + the comet's start state fully define the
               particle, so a Java run can be rebuilt exactly
               (see particles_from_java_run below).
  - epoch_index : the Java `OutburstAndInitialDateIndex` (9 = 1892-11-04).
  - phase    : Sun-particle-Earth angle, degrees.

* "Forecast" files ``b<bin>f<NN>.csv`` (f10..f22 in the 1892 program): the
  rows of every particle on one forecast date -- same fields minus "step".
* "Matching" files ``b<bin>m<N>.csv``: rows on a date of a real historical
  observation, kept only if RA/DEC is inside a +-40 deg box around it.

Known defects of the Java output that this module fixes (not reproduces):
* the f/m files were never close()d -> only the part flushed before the
  program stopped is on disk (in the example run: f-files hold particles
  1-63 of 78 finished ones, m-files are empty although m2 should have rows).
  rebuild_snapshot_files() regenerates them completely from the p-files.
* the b2/b3 forecast conditions compared against a date string WITHOUT the
  trailing "Z" (e.g. "2026-08-01T11:59:23.000"), which can never match
  AbsoluteDate.toString() -> b2/b3 f-files would always stay empty.
* toDegMinSec() drops the minus sign for declinations between -1 and 0 deg
  ("0° 30' ..." for -0.5 deg). Kept as-is in java_deg_min_sec() on purpose
  (byte-compatibility); the decimal DEC column is always correct.
"""

from __future__ import annotations  # allows `list[str]`-style hints on older Python

import dataclasses       # for LegacySnapshot
import datetime as dt    # dates of forecast/matching files
import decimal           # exact digit extraction for Java-style double printing
import glob              # finding b1p*.csv files
import math              # floor / copysign for the sexagesimal formatters
import os                # paths
import re                # parsing file names like b1p17.csv

import numpy as np

from .particles import DustParticle, direction_from_theta_u, make_particle

# The old Java metres -> AU factor (pv.getPosition().getX()*6.68458712E-12).
METERS_TO_AU_LEGACY = 6.68458712e-12

# Column names of a p-file line, in the Java order (see module docstring).
P_COLUMNS = ["jd", "step", "date", "x_au", "y_au", "z_au", "ra_hms", "dec_dms",
             "ra_deg", "dec_deg", "radius_mm", "theta", "u", "epoch_index", "phase_deg"]
# f/m-file lines are identical except that the "step" field is missing.
F_COLUMNS = [c for c in P_COLUMNS if c != "step"]
_TEXT_COLUMNS = {"date", "ra_hms", "dec_dms"}   # everything else is numeric

# Java bin number (the "1" in b1p5.csv) <-> this package's bin names.
BIN_NUMBER = {"b1_small": 1, "b2_medium": 2, "b3_big": 3}
BIN_NAME = {v: k for k, v in BIN_NUMBER.items()}


# ---------------------------------------------------------------------------
# Small formatting helpers that mimic Java's output exactly
# ---------------------------------------------------------------------------

def julian_day_number(date: dt.date | dt.datetime) -> int:
    """Integer Julian Day Number of a calendar date (what Java's
    `JulianFields.JULIAN_DAY.getFrom(localdate)` returns): the JD at noon of
    that date. 2000-01-01 -> 2451545. Time of day is ignored."""
    if isinstance(date, dt.datetime):
        date = date.date()                     # only the calendar date matters
    return date.toordinal() + 1721425          # proleptic-Gregorian ordinal -> JDN


def date_from_julian_day_number(jdn: int) -> dt.date:
    """Inverse of julian_day_number()."""
    return dt.date.fromordinal(int(jdn) - 1721425)


def java_double_str(x: float) -> str:
    """Format a double the way Java's `%s` / Double.toString() does:
    shortest round-trip digits; plain notation for 1e-3 <= |x| < 1e7
    (always with at least one decimal, e.g. "9.0"), otherwise
    "d.dddE[-]n" (e.g. "1.0E-5"). NaN/Infinity as in Java."""
    if 1e-3 <= abs(x) < 1e7:
        # Fast path (almost every value): Python's repr() prints exactly the
        # same shortest digits in plain notation in this range ("123.0",
        # "0.0072453997304867").
        return repr(float(x))
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    if x == 0.0:
        return "-0.0" if math.copysign(1.0, x) < 0 else "0.0"
    sign = "-" if x < 0 else ""
    # repr() gives the shortest digit string that round-trips (same digits Java prints).
    tup = decimal.Decimal(repr(abs(x))).as_tuple()
    digits = "".join(map(str, tup.digits)).rstrip("0") or "0"
    # decimal exponent of the first significant digit
    exp10 = len(tup.digits) + tup.exponent - 1
    if 1e-3 <= abs(x) < 1e7:
        if exp10 >= 0:                                        # e.g. 123.45
            int_part = digits[:exp10 + 1].ljust(exp10 + 1, "0")
            frac = digits[exp10 + 1:] or "0"
        else:                                                 # e.g. 0.00123
            int_part = "0"
            frac = "0" * (-exp10 - 1) + digits
        return f"{sign}{int_part}.{frac}"
    mant = digits[0] + "." + (digits[1:] or "0")             # scientific: d.dddE n
    return f"{sign}{mant}E{exp10}"


def java_hmin_sec(ra_deg: float) -> str:
    """Exact port of the Java `toHMinSec`: "%02dH %02dmin %6.3fsec"."""
    s = abs(ra_deg / 15.0)
    h = int(math.copysign(math.floor(s), ra_deg))
    s = (s - abs(h)) * 60.0
    m = int(math.floor(s))
    s = (s - m) * 60.0
    return "%02dH %02dmin %6.3fsec" % (h, m, s)


def java_deg_min_sec(dec_deg: float) -> str:
    """Exact port of the Java `toDegMinSec`: "%d° %02d' %6.3f''".
    (Inherits the Java quirk: -0.5 deg prints as "0° 30' ...", sign lost.)"""
    s = abs(dec_deg)
    d = int(math.copysign(math.floor(s), dec_deg))
    s = (s - abs(d)) * 60.0
    m = int(math.floor(s))
    s = (s - m) * 60.0
    return "%d° %02d' %6.3f''" % (d, m, s)


def orekit_date_str(epoch_utc: dt.datetime) -> str:
    """UTC date in Orekit's AbsoluteDate.toString() layout:
    "1892-11-04T12:00:00.000Z" (millisecond precision, trailing Z)."""
    ms = epoch_utc.microsecond // 1000
    return epoch_utc.strftime("%Y-%m-%dT%H:%M:%S") + f".{ms:03d}Z"


def format_legacy_fields(
    *,
    epoch_utc: dt.datetime,
    barycentric_position_m,
    ra_deg: float,
    dec_deg: float,
    radius_mm: float,
    theta: float,
    u: float,
    epoch_index: int,
    phase_angle_deg: float,
    step: int | None = None,
) -> str:
    """One Java-layout line (including the Java trailing " " + newline).
    With `step` -> p-file line; without -> f/m-file line."""
    xyz = np.asarray(barycentric_position_m, dtype=float) * METERS_TO_AU_LEGACY
    fields = [str(julian_day_number(epoch_utc))]
    if step is not None:
        fields.append("step %2d" % step)                        # Java: "step %2d"
    fields += [
        orekit_date_str(epoch_utc),
        java_double_str(float(xyz[0])), java_double_str(float(xyz[1])), java_double_str(float(xyz[2])),
        java_hmin_sec(ra_deg), java_deg_min_sec(dec_deg),
        java_double_str(float(ra_deg)), java_double_str(float(dec_deg)),
        java_double_str(float(radius_mm)), java_double_str(float(theta)), java_double_str(float(u)),
        str(int(epoch_index)),
        java_double_str(float(phase_angle_deg)),
    ]
    return ", ".join(fields) + " \n"        # Java format ended with "%s %n"


# ---------------------------------------------------------------------------
# Snapshot ("f" forecast / "m" matching) definitions
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class LegacySnapshot:
    """One old f- or m-file: rows of every particle whose date has Julian
    Day Number `jdn`, and -- for m-files -- whose RA/DEC is inside the
    +-tolerance box around (ra_center_deg, dec_center_deg)."""
    stem: str                                  # "f10", "m2", ... -> file b1f10.csv, b1m2.csv
    jdn: int                                   # Julian Day Number of the target date
    ra_center_deg: float | None = None         # m-files only
    dec_center_deg: float | None = None        # m-files only
    tolerance_deg: float = 40.0                # Java: abs(alphaDeg - X) < 40.0 (strict)

    @property
    def date(self) -> dt.date:
        return date_from_julian_day_number(self.jdn)

    def matches(self, jdn: int, ra_deg: float, dec_deg: float) -> bool:
        if jdn != self.jdn:
            return False
        if self.ra_center_deg is None:
            return True                                           # forecast file: every particle
        return (abs(ra_deg - self.ra_center_deg) < self.tolerance_deg
                and abs(dec_deg - self.dec_center_deg) < self.tolerance_deg)


def _f(stem: str, y: int, m: int, d: int) -> LegacySnapshot:
    return LegacySnapshot(stem, julian_day_number(dt.date(y, m, d)))


# Forecast dates f10..f22 shared by SlaveMode_Holmes1892_Refactor_1892.java
# and SlaveMode_Holmes1892_Refactor.java (1st of the month, noon UTC).
_FORECASTS_F10_F22 = [
    _f("f10", 2026, 8, 1), _f("f11", 2026, 10, 1), _f("f12", 2026, 12, 1),
    _f("f13", 2027, 2, 1), _f("f14", 2027, 3, 1), _f("f15", 2027, 8, 1),
    _f("f16", 2027, 10, 1), _f("f17", 2027, 12, 1), _f("f18", 2028, 2, 1),
    _f("f19", 2028, 10, 1), _f("f20", 2029, 2, 1), _f("f21", 2029, 10, 1),
    _f("f22", 2030, 2, 1),
]
# Matching windows present in every Java version (dates = JDN in the code).
_MATCH_M2 = LegacySnapshot("m2", 2456341, 242.0, -40.4)    # 2013-02-17
_MATCH_M14 = LegacySnapshot("m14", 2457069, 21.0, 27.2)    # 2015-02-15

# SlaveMode_Holmes1892_Refactor_1892.java (the program of the example run).
JAVA_1892_SNAPSHOTS: tuple[LegacySnapshot, ...] = (*_FORECASTS_F10_F22, _MATCH_M2, _MATCH_M14)
# SlaveMode_Holmes1892_Refactor_2007.java: same + three later forecasts.
JAVA_2007_REFACTOR_SNAPSHOTS: tuple[LegacySnapshot, ...] = (
    *_FORECASTS_F10_F22,
    _f("f23", 2030, 10, 1), _f("f24", 2031, 2, 1), _f("f25", 2031, 10, 1),
    _MATCH_M2, _MATCH_M14,
)


# ---------------------------------------------------------------------------
# Writing (used by run_scenario.py when output_format is LEGACY/BOTH)
# ---------------------------------------------------------------------------

class LegacyJavaWriter:
    """Writes the old Java file layout for a run of THIS package.

    Rows arrive time-major (all particles at day 0, all at day 1, ...) from
    the batched simulation, but the layout is one file per particle. Keeping
    hundreds of files open is fragile (OS open-file limits), so lines are
    buffered in memory and appended to each file every `flush_every` epochs
    (default 50 epochs x 400 particles ~ 5 MB of RAM).
    Unlike the Java program, every file is always flushed and closed, also
    when the run is interrupted by an exception.
    """

    def __init__(self, output_dir: str, snapshots=(), epoch_index: int | None = None, flush_every: int = 50):
        self.output_dir = output_dir
        self.snapshots = tuple(snapshots)
        self.epoch_index = -1 if epoch_index is None else int(epoch_index)
        self.flush_every = int(flush_every)
        os.makedirs(output_dir, exist_ok=True)
        self._p_buffers: dict[str, list[str]] = {}           # file name -> pending lines
        self._steps: dict[str, int] = {}                     # file name -> rows written so far
        self._snap_buffers: dict[str, list[str]] = {}        # f/m file name -> pending lines
        self._created: set[str] = set()                      # files already truncated/created this run
        self._epochs_since_flush = 0
        self._last_epoch = None
        # Create (empty) every snapshot file up front, like the Java program.
        # An empty file then clearly means "no particle matched".
        self._bins_seen: set[int] = set()

    @staticmethod
    def particle_filename(particle: DustParticle) -> str:
        n = particle.legacy_number if particle.legacy_number is not None else particle.particle_id + 1
        return f"b{BIN_NUMBER.get(particle.bin_name, 0)}p{n}.csv"

    def write(self, *, epoch_utc: dt.datetime, particle: DustParticle, barycentric_position_m,
              ra_deg: float, dec_deg: float, phase_angle_deg: float) -> None:
        # A new epoch started -> maybe flush the buffers of the previous ones.
        if epoch_utc != self._last_epoch:
            if self._last_epoch is not None:
                self._epochs_since_flush += 1
                if self._epochs_since_flush >= self.flush_every:
                    self.flush()
            self._last_epoch = epoch_utc
        name = self.particle_filename(particle)
        step = self._steps.get(name, 0) + 1
        self._steps[name] = step
        common = dict(epoch_utc=epoch_utc, barycentric_position_m=barycentric_position_m, ra_deg=ra_deg,
                      dec_deg=dec_deg, radius_mm=particle.radius_mm, theta=particle.theta, u=particle.u,
                      epoch_index=self.epoch_index, phase_angle_deg=phase_angle_deg)
        self._p_buffers.setdefault(name, []).append(format_legacy_fields(step=step, **common))
        bin_no = BIN_NUMBER.get(particle.bin_name, 0)
        if bin_no not in self._bins_seen:
            self._bins_seen.add(bin_no)
            for snap in self.snapshots:                      # empty f/m files for this bin
                self._snap_buffers.setdefault(f"b{bin_no}{snap.stem}.csv", [])
        jdn = julian_day_number(epoch_utc)
        for snap in self.snapshots:
            if snap.matches(jdn, ra_deg, dec_deg):
                self._snap_buffers[f"b{bin_no}{snap.stem}.csv"].append(format_legacy_fields(**common))

    def _append(self, name: str, lines: list[str]) -> None:
        path = os.path.join(self.output_dir, name)
        mode = "a" if name in self._created else "w"          # first write of this run truncates old files
        with open(path, mode, encoding="utf-8", newline="") as f:
            f.writelines(lines)
        self._created.add(name)

    def flush(self) -> None:
        for buffers in (self._p_buffers, self._snap_buffers):
            for name, lines in buffers.items():
                if lines or name not in self._created:
                    self._append(name, lines)
                    lines.clear()
        self._epochs_since_flush = 0

    def close(self) -> None:
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

_FILE_RE = re.compile(r"^b(\d)([pfm])(\d+)\.csv$")


def list_legacy_files(folder: str) -> list[tuple[int, str, int, str]]:
    """All Java-layout files in `folder` as (bin_no, kind, number, path),
    kind in {"p", "f", "m"}, sorted by bin, kind, number (numerically, so
    b1p2 comes before b1p10)."""
    out = []
    for path in glob.glob(os.path.join(folder, "b*.csv")):
        m = _FILE_RE.match(os.path.basename(path))
        if m:
            out.append((int(m.group(1)), m.group(2), int(m.group(3)), path))
    return sorted(out)


def iter_legacy_rows(path: str, columns=None):
    """Yield one dict per COMPLETE line of a Java-layout file. Lines with
    the wrong number of fields (e.g. the half-written last line of a file
    that was never flushed/closed) are skipped. Numeric fields -> float,
    jd/epoch_index/step -> int."""
    if columns is None:
        columns = P_COLUMNS if re.search(r"b\dp\d+\.csv$", path) else F_COLUMNS
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = [p.strip() for p in line.rstrip("\r\n").split(",")]
            if len(parts) != len(columns):
                continue
            row = {}
            try:
                for name, value in zip(columns, parts):
                    if name in _TEXT_COLUMNS:
                        row[name] = value
                    elif name == "step":
                        row[name] = int(value.replace("step", ""))
                    elif name in ("jd", "epoch_index"):
                        row[name] = int(value)
                    else:
                        row[name] = float(value)
            except ValueError:
                continue                                    # corrupted line -> skip
            yield row


def read_first_row(path: str) -> dict | None:
    """First complete line of a Java-layout file, or None if empty."""
    for row in iter_legacy_rows(path):
        return row
    return None


def particles_from_java_run(folder: str, comet_position_m, comet_velocity_m_s,
                            bins: tuple[int, ...] = (1, 2, 3), max_particles: int | None = None,
                            only_complete: bool = False, expected_rows: int | None = None,
                            fill_from_forecasts: bool = True) -> list[DustParticle]:
    """Rebuild the exact particle population of an old Java run from its
    b<bin>p<N>.csv files: radius, theta and u are read from the first line
    of each file, and beta / ejection speed / direction are recomputed with
    the same formulas the Java code used (particles.make_particle).

    `comet_position_m` / `comet_velocity_m_s` must be the Java start state
    of the comet (ephemeris.LEGACY_STATES, i.e. EphemerisSource
    HORIZONS_WITH_LEGACY_COMET) for the rebuild to be exact.

    only_complete + expected_rows: skip particles whose file has fewer rows
    (e.g. the particle that was still running when the Java program was
    stopped).

    fill_from_forecasts: particles whose p-file is MISSING (deleted to save
    space, say) are rebuilt from a forecast file instead. The Java program
    ran the particles one after another, so line k of a forecast file
    (b1f10.csv ...) belongs to particle k (checked on a real run: lines
    101-200 of b1f13.csv equal the 2027-02-01 lines of b1p101..b1p200.csv),
    and every line carries radius, theta and u. The forecast file with the
    most lines is used.
    """
    found: list[tuple[int, int, dict]] = []                   # (bin, number, a row with radius/theta/u)
    seen: set[tuple[int, int]] = set()
    files = list_legacy_files(folder)
    for bin_no, kind, number, path in files:
        if kind != "p" or bin_no not in bins:
            continue
        first = read_first_row(path)
        if first is None:
            continue                                          # empty file
        seen.add((bin_no, number))                            # the p-file exists (even if we skip it below)
        if only_complete and expected_rows is not None:
            with open(path, "rb") as f:
                n_rows = sum(1 for _ in f)
            if n_rows < expected_rows:
                continue
        found.append((bin_no, number, first))
    if fill_from_forecasts:
        for bin_no in bins:
            forecasts = [path for b, kind, _n, path in files if b == bin_no and kind == "f"]
            rows = max((list(iter_legacy_rows(path)) for path in forecasts), key=len, default=[])
            for k, row in enumerate(rows):
                if (bin_no, k + 1) not in seen:
                    found.append((bin_no, k + 1, row))
    found.sort(key=lambda item: (item[0], item[1]))           # bin, then Java particle number
    particles: list[DustParticle] = []
    for bin_no, number, first in found:
        particles.append(make_particle(
            particle_id=len(particles),
            bin_name=BIN_NAME.get(bin_no, f"b{bin_no}"),
            radius_mm=first["radius_mm"],
            direction=direction_from_theta_u(first["theta"], first["u"]),
            comet_position_m=comet_position_m,
            comet_velocity_m_s=comet_velocity_m_s,
            theta=first["theta"],
            u=first["u"],
            legacy_number=number,                              # keep the Java file number
        ))
        if max_particles is not None and len(particles) >= max_particles:
            break
    return particles


def rebuild_snapshot_files(p_folder: str, out_folder: str, snapshots=JAVA_1892_SNAPSHOTS) -> dict[str, int]:
    """Regenerate complete f/m files from the per-particle p-files (works
    for the old Java folders -- whose f/m files are truncated -- and for
    legacy-layout output of this package). Rows are copied verbatim (minus
    the "step" field), in particle order. Returns {file name: row count}.
    """
    os.makedirs(out_folder, exist_ok=True)
    buffers: dict[str, list[str]] = {}
    for bin_no, kind, number, path in list_legacy_files(p_folder):
        if kind != "p":
            continue
        for snap in snapshots:
            buffers.setdefault(f"b{bin_no}{snap.stem}.csv", [])
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\r\n").split(",")
                if len(parts) != len(P_COLUMNS):
                    continue                                   # truncated last line
                try:
                    jdn = int(parts[0])
                    ra, dec = float(parts[8]), float(parts[9])
                except ValueError:
                    continue
                for snap in snapshots:
                    if snap.matches(jdn, ra, dec):
                        # drop field 1 ("step K"), keep everything else byte-for-byte
                        buffers[f"b{bin_no}{snap.stem}.csv"].append(",".join([parts[0]] + parts[2:]) + "\n")
    counts = {}
    for name, lines in sorted(buffers.items()):
        with open(os.path.join(out_folder, name), "w", encoding="utf-8", newline="") as f:
            f.writelines(lines)
        counts[name] = len(lines)
    return counts
