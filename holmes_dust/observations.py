"""
observations.py
================
Turns Cartesian state vectors into the "synthetic observation" quantities
the project actually cares about: geocentric Right Ascension / Declination
and the Sun-target-Earth phase angle. Also implements a real light-time
correction, which the old Java code explicitly did NOT do (see the comment
it left behind: "time of light travel must be compensated after RA DEC
calculation" -- it never actually was, in any of the six files).

Everything here is plain vector math on numpy arrays; there is no
dependency on REBOUND or any particular ephemeris backend, so it can be
unit-tested in isolation (see tests/test_observations.py).
"""

from __future__ import annotations  # allows modern type-hint syntax on older Python

import dataclasses   # for the @dataclasses.dataclass decorator (see Observation)

import numpy as np   # all vector math (dot products, norms, trig) uses numpy

# Speed of light, exact SI value (m/s) -- used for the light-time correction.
SPEED_OF_LIGHT_M_S = 299_792_458.0


# A plain, read-only bundle of everything one "synthetic observation"
# consists of, for one particle at one moment in time.
@dataclasses.dataclass(frozen=True)
class Observation:
    ra_deg: float               # Right Ascension, in degrees, in [0, 360)
    dec_deg: float               # Declination, in degrees, in [-90, 90]
    phase_angle_deg: float       # Sun-target-Earth angle, in degrees, in [0, 180]
    light_time_s: float          # how many seconds light took to arrive; 0.0 if correction was not applied
    # The geocentric vector actually used to derive ra_deg/dec_deg (i.e.
    # after light-time correction, if that was applied). Carried on the
    # result so downstream code (io_export.py) reports AU coordinates that
    # are consistent with the reported RA/DEC, instead of recomputing a
    # (possibly different) vector from the uncorrected true position.
    geocentric_position_m: np.ndarray   # length-3 vector, metres, "target minus Earth"


def ra_dec_from_geocentric_vector(geocentric_position_m: np.ndarray) -> tuple[float, float]:
    """Right Ascension / Declination (degrees) from a geocentric Cartesian
    position vector.

    Direct port of:
        alphaDeg = toDegrees(normalizeAngle(bodyEME2000.getAlpha(), PI))
        deltaDeg = toDegrees(bodyEME2000.getDelta())
    i.e. standard spherical angles of the vector, with RA normalised into
    [0, 360) degrees (Orekit's `getAlpha()` already returns a value in
    (-pi, pi], and `normalizeAngle(..., PI)` re-centres it on +pi, which is
    numerically equivalent to plain `% 360` on the degree value used below).
    """
    x, y, z = geocentric_position_m           # unpack the 3 components of the vector
    r_xy = np.hypot(x, y)                      # length of the vector's projection onto the x-y plane
    ra_rad = np.arctan2(y, x)                  # angle around the z-axis (RA), in radians
    dec_rad = np.arctan2(z, r_xy)              # angle up/down from the x-y plane (DEC), in radians

    ra_deg = np.degrees(ra_rad) % 360.0        # convert to degrees and wrap into [0, 360)
    dec_deg = np.degrees(dec_rad)              # convert to degrees (already naturally in [-90, 90])
    return float(ra_deg), float(dec_deg)       # return as plain Python floats, not numpy scalars


def phase_angle_deg(sun_position_m: np.ndarray, target_position_m: np.ndarray, earth_position_m: np.ndarray) -> float:
    """Sun-target-Earth phase angle (degrees): the angle, AS SEEN FROM THE
    TARGET, between the direction to the Sun and the direction to the Earth.

    Direct port of:
        xa = sun.x - target.x ; xb = earth.x - target.x   (and y, z likewise)
        phase = acos( (xa.xb) / (|xa||xb|) ) in degrees
    """
    to_sun = sun_position_m - target_position_m      # vector from the target toward the Sun
    to_earth = earth_position_m - target_position_m  # vector from the target toward the Earth

    # cos(angle between two vectors) = dot product / (product of lengths).
    cos_phase = np.dot(to_sun, to_earth) / (np.linalg.norm(to_sun) * np.linalg.norm(to_earth))
    # Clip against floating-point drift pushing the cosine slightly outside
    # [-1, 1], which would otherwise make arccos return NaN.
    cos_phase = np.clip(cos_phase, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_phase)))   # arccos gives radians; convert to degrees


def apply_light_time_correction(
    target_position_at_t_m: np.ndarray,     # target's TRUE position at reception time t
    target_velocity_at_t_m_s: np.ndarray,   # target's TRUE velocity at reception time t
    earth_position_at_t_m: np.ndarray,      # Earth's position at reception time t (treated as fixed)
    *,
    max_iterations: int = 3,                # how many refinement passes to run at most
    tolerance_s: float = 1e-3,               # stop early once the light-time estimate barely changes
) -> tuple[np.ndarray, float]:
    """Estimate where the target *appeared* to be at emission time t - tau,
    given its true state at reception time t, by first-order iteration.

    NEW FEATURE relative to the old Java program, which explicitly left
    this uncorrected (see module docstring). The old code's own comment
    explains why it was skipped there: Orekit's `NumericalPropagator` in
    this setup produces the state at a requested time but has no built-in
    Chebyshev/interpolated ephemeris to cheaply re-query an *earlier*
    arbitrary time, so a real light-time correction would have needed a
    second propagation per particle per output step.

    Here, we approximate the target's trajectory near t with a straight
    line (position, velocity) -- which is an excellent approximation
    because the light-time itself (tens of minutes for a dust particle a
    few AU from Earth) is minute compared to the days-to-years timescale
    over which the trajectory actually curves:

        tau_0 = |r_target(t) - r_earth(t)| / c
        r_target(t - tau) ~= r_target(t) - v_target(t) * tau
        iterate tau <- |r_target(t - tau) - r_earth(t)| / c until it stops
        changing (Earth's own position at time t is treated as fixed here;
        for daily-cadence output the light-time is a tiny fraction of a
        day, so re-querying Earth's position at t - tau as well would
        change the result negligibly).

    Returns (corrected_target_position_m, light_time_s).
    """
    tau = 0.0   # current light-time estimate (seconds), starts at zero
    # Start the iteration from the target's true (uncorrected) position.
    corrected_position = np.array(target_position_at_t_m, dtype=float)

    for _ in range(max_iterations):   # refine the estimate up to max_iterations times
        # Distance between the (current guess of) apparent position and Earth.
        distance_m = np.linalg.norm(corrected_position - earth_position_at_t_m)
        # Light-time = distance / speed of light.
        new_tau = distance_m / SPEED_OF_LIGHT_M_S
        # Where the target WOULD have been `new_tau` seconds earlier, assuming
        # it moved in a straight line at its current velocity.
        corrected_position = target_position_at_t_m - target_velocity_at_t_m_s * new_tau

        if abs(new_tau - tau) < tolerance_s:
            # The estimate barely changed from the last pass -- good enough,
            # stop early instead of running all max_iterations passes.
            tau = new_tau
            break
        tau = new_tau   # otherwise, keep this as the new estimate and loop again

    return corrected_position, tau   # final apparent position and light-time (seconds)


def compute_observation(
    *,
    target_position_m: np.ndarray,     # the particle's/comet's true position (SI, metres)
    target_velocity_m_s: np.ndarray,   # the particle's/comet's true velocity (SI, m/s)
    sun_position_m: np.ndarray,        # the Sun's position at the same moment (SI, metres)
    earth_position_m: np.ndarray,      # Earth's position at the same moment (SI, metres)
    apply_light_time: bool = True,     # whether to apply the light-time correction above
) -> Observation:
    """One-stop helper: optionally applies the light-time correction, then
    computes RA/DEC (geocentric) and the phase angle (which the old code
    always evaluated using the *un*-corrected target position -- the phase
    angle here is likewise computed on the (possibly corrected) apparent
    position, which is the physically consistent choice for a real
    observation and differs at most negligibly from using the true
    position, given how small the light-time correction is compared to the
    scales that set the phase angle).
    """
    light_time_s = 0.0                        # default: no correction applied
    apparent_position_m = target_position_m   # default: apparent = true position

    if apply_light_time:
        # Overwrite the defaults above with the corrected values.
        apparent_position_m, light_time_s = apply_light_time_correction(
            target_position_m, target_velocity_m_s, earth_position_m
        )

    # "Geocentric" means "relative to Earth" -- subtract Earth's position.
    geocentric_vector = apparent_position_m - earth_position_m
    ra_deg, dec_deg = ra_dec_from_geocentric_vector(geocentric_vector)
    phase_deg = phase_angle_deg(sun_position_m, apparent_position_m, earth_position_m)

    # Package everything computed above into one Observation object.
    return Observation(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        phase_angle_deg=phase_deg,
        light_time_s=light_time_s,
        geocentric_position_m=geocentric_vector,
    )


def to_hms(angle_deg: float) -> str:
    """Format a RA-like degree value as H/min/sec, matching the old code's
    `toHMinSec` helper (angle / 15 to go from degrees to hours)."""
    total_hours = abs(angle_deg) / 15.0            # 360 degrees = 24 hours, so divide by 15
    sign = -1.0 if angle_deg < 0 else 1.0           # remember the original sign
    h = int(np.floor(total_hours)) * sign            # whole number of hours (signed)
    remainder_min = (total_hours - abs(h)) * 60.0    # leftover fraction of an hour, converted to minutes
    m = int(np.floor(remainder_min))                 # whole number of minutes
    s = (remainder_min - m) * 60.0                   # leftover fraction of a minute, converted to seconds
    return f"{int(h):02d}H {m:02d}min {s:06.3f}sec"  # assemble the final "HH H MMmin SS.SSSsec" string


def to_dms(angle_deg: float) -> str:
    """Format a DEC-like degree value as deg/min/sec, matching the old
    code's `toDegMinSec` helper."""
    sign = -1.0 if angle_deg < 0 else 1.0    # remember the original sign
    total = abs(angle_deg)                    # work with the unsigned magnitude
    d = int(np.floor(total)) * sign            # whole number of degrees (signed)
    remainder_min = (total - abs(d)) * 60.0    # leftover fraction of a degree, converted to arcminutes
    m = int(np.floor(remainder_min))            # whole number of arcminutes
    s = (remainder_min - m) * 60.0              # leftover fraction of an arcminute, converted to arcseconds
    return f"{int(d)}\u00b0 {m:02d}' {s:06.3f}''"  # assemble the final "D° MM' SS.SSS''" string
