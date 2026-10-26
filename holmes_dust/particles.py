"""
particles.py
============
Generation of individual dust particles ejected from the comet nucleus.

This module is a direct, cleaned-up port of the "particle number loop" /
"bvalue loop" body from the old Java code (the block that samples
`particle_radius`, computes `Ybeta` and `Yej`, and picks an ejection
direction). Only the branch that is actually *enabled* in all six source
files (`locked_beta_and_ejection_speed == 1`, `locked_ejection_speed_spherichal
== 1`) has been ported; the disabled Gaussian-velocity-perturbation branch
was legacy/dead code and is intentionally left out (see config.py's
PARTICLE_BINS docstring for details).

Nothing here talks to REBOUND -- this module only produces plain numbers
(radius, beta, speed, direction, initial state vector). simulation.py is
responsible for turning a DustParticle into an actual REBOUND particle.
"""

from __future__ import annotations  # allows the `np.ndarray | None` union syntax below

import dataclasses   # for the @dataclasses.dataclass decorator (see DustParticle)

import numpy as np   # all the vector/array math in this file uses numpy

from .config import EjectionDirectionMode, ParticleSizeBin  # types this module reads from config.py


# A plain, read-only container: one fully-described dust particle. Building
# one of these does NOT talk to REBOUND or do any physics -- it just bundles
# together numbers already computed by the functions further down this file.
@dataclasses.dataclass(frozen=True)
class DustParticle:
    """One fully-specified virtual dust grain, ready to be added to a
    simulation."""
    particle_id: int                     # 0, 1, 2, ... -- unique index across the whole population
    bin_name: str                        # which size class this particle came from, e.g. "b1_small"
    radius_mm: float                     # this particle's own randomly-sampled radius (millimetres)
    beta: float                          # ratio of radiation-pressure force to solar gravity
    ejection_speed_m_s: float            # how fast this particle was ejected from the comet (m/s)
    direction_unit_vector: np.ndarray    # length-3, unit norm -- which way it was ejected
    initial_position_m: np.ndarray       # = comet position at ejection epoch (metres, SI)
    initial_velocity_m_s: np.ndarray     # = comet velocity + ejection_speed * direction (m/s, SI)
    # The two random numbers the isotropic direction was built from (see
    # sample_isotropic_theta_u). Stored so they can be written to the
    # legacy Java-layout output (its "theta" and "u" columns) and so a run
    # can be rebuilt exactly from an old output file. NaN for TOWARD_SUN.
    theta: float = float("nan")
    u: float = float("nan")
    # Optional particle number shown in legacy file names (b1p<N>.csv).
    # None -> particle_id + 1 (the Java program counted particles from 1).
    legacy_number: int | None = None


def sample_particle_radius_mm(rng: np.random.Generator, size_bin: ParticleSizeBin) -> float:
    """Uniform sampling of the particle radius within its size bin.

    Direct port of e.g.
        particle_radius = RG.nextDouble()*0.009+0.001;   // b1: 0.001-0.01 mm
    generalised to read the bin's own min/max instead of hard-coding them
    per branch.
    """
    # rng.uniform(low, high) draws one random float in [low, high).
    return rng.uniform(size_bin.radius_min_mm, size_bin.radius_max_mm)


def beta_from_radius_mm(radius_mm: float) -> float:
    """Radiation-pressure-to-gravity ratio (Burns et al. 1979 / Lamy et al.
    style fit), as a function of particle radius in millimetres.

    Direct port of:
        Ybeta = 0.0002*Math.pow(particle_radius, -1.0438);

    NOTE (verify-in-project): the coefficients 0.0002 and -1.0438 are taken
    verbatim from the old code's comment attribution ("Burns et al Langraf
    et al"); this repository does not contain the original paper, so the
    exact source/derivation of these two numbers cannot be confirmed from
    the Java files alone.
    """
    # Smaller radius -> larger beta (radiation pressure matters more for
    # tiny grains), because the exponent -1.0438 is negative.
    return 0.0002 * radius_mm ** (-1.0438)


def ejection_speed_from_radius_mm(radius_mm: float) -> float:
    """Ejection speed (m/s) as a function of particle radius in millimetres.

    Direct port of:
        Yej = 600.99*Math.exp(-0.6086*particle_radius);

    NOTE (verify-in-project): attributed in the old comments to "Reach et
    al", same caveat as beta_from_radius_mm above.
    """
    # Larger radius -> lower ejection speed (heavier grains are harder to
    # blow off the nucleus), because of the negative exponent.
    return 600.99 * np.exp(-0.6086 * radius_mm)


def sample_isotropic_theta_u(rng: np.random.Generator) -> tuple[float, float]:
    """Draw the two random numbers of the old code's isotropic direction:
        theta = U(0, 2*pi)   (azimuth)
        u     = U(-1, 1)     (cosine of the polar angle, i.e. the z component)
    Drawn in exactly this order (theta first), so a given seed produces the
    same particles as earlier versions of this package.
    """
    theta = rng.uniform(0.0, 2.0 * np.pi)   # random azimuth angle, full circle
    u = rng.uniform(-1.0, 1.0)              # random "height" along the sphere's axis
    # The old code nudged u/theta away from the exact boundary values to
    # avoid a degenerate direction vector; np.random.Generator.uniform's
    # half-open interval already excludes the upper bound, so that nudge is
    # not needed here, but the same defensive clamp is kept for parity.
    u = float(np.clip(u, -0.999999, 0.999999))   # keep u strictly inside (-1, 1)
    return float(theta), u


def direction_from_theta_u(theta: float, u: float) -> np.ndarray:
    """Unit vector from (theta, u), exactly as in the old code:
        x = sqrt(1-u^2) * cos(theta)
        y = sqrt(1-u^2) * sin(theta)
        z = u
    Also used to REBUILD a Java particle from the theta/u columns of its
    old output file (see legacy_java.py)."""
    r_xy = np.sqrt(1.0 - u * u)             # radius of the circle at "height" u
    return np.array([r_xy * np.cos(theta), r_xy * np.sin(theta), u])


def sample_isotropic_direction(rng: np.random.Generator) -> np.ndarray:
    """Uniform-on-the-sphere unit direction vector.

    This parametrisation ("z uniform, azimuth uniform") is mathematically
    correct for isotropic sampling, so it is kept as-is rather than replaced
    with e.g. `rng.normal(size=3)` + normalise (an equally valid alternative
    that was NOT what the old code did).
    """
    theta, u = sample_isotropic_theta_u(rng)   # the two random numbers
    return direction_from_theta_u(theta, u)    # -> 3D unit vector


def make_particle(
    *,
    particle_id: int,
    bin_name: str,
    radius_mm: float,
    direction: np.ndarray,
    comet_position_m: np.ndarray,
    comet_velocity_m_s: np.ndarray,
    theta: float = float("nan"),
    u: float = float("nan"),
    legacy_number: int | None = None,
) -> DustParticle:
    """Build one DustParticle from its radius and ejection direction.
    Shared by generate_particles() (random particles) and by
    legacy_java.particles_from_java_run() (particles rebuilt from an old
    Java output folder), so both use literally the same physics."""
    beta = beta_from_radius_mm(radius_mm)               # radiation-pressure ratio from the radius
    speed = ejection_speed_from_radius_mm(radius_mm)     # ejection speed from the radius
    return DustParticle(
        particle_id=particle_id,
        bin_name=bin_name,
        radius_mm=float(radius_mm),
        beta=float(beta),
        ejection_speed_m_s=float(speed),
        direction_unit_vector=np.asarray(direction, dtype=float),
        # Every particle starts exactly at the comet's position at the
        # outburst epoch -- matches the old code, which never modelled an
        # extended ejection duration or an offset release point.
        initial_position_m=np.array(comet_position_m, dtype=float),
        # Comet velocity + ejection speed * direction = the particle's true
        # initial velocity in the barycentric frame.
        initial_velocity_m_s=np.asarray(comet_velocity_m_s, dtype=float) + speed * np.asarray(direction, dtype=float),
        theta=float(theta),
        u=float(u),
        legacy_number=legacy_number,
    )


def sample_toward_sun_direction(comet_position_m: np.ndarray, sun_position_m: np.ndarray) -> np.ndarray:
    """Unit vector from the comet toward the Sun.

    Direct port of the (never-enabled) `locked_ejection_speed_toward_sun == 1`
    branch. Provided for completeness / future use, but no supplied
    scenario actually turns this mode on.
    """
    delta = sun_position_m - comet_position_m   # vector pointing from comet to Sun
    norm = np.linalg.norm(delta)                # length of that vector
    if norm == 0.0:
        # Comet and Sun can never really coincide, but guard against a
        # division-by-zero crash if they somehow did.
        raise ValueError("Comet and Sun positions coincide; cannot form a direction vector")
    return delta / norm   # normalise to unit length


def generate_particles(
    *,
    bins,                                      # which ParticleSizeBin objects to generate for
    n_particles_per_bin: int,                  # how many particles to make PER bin
    direction_mode: EjectionDirectionMode,     # isotropic sphere, or toward-Sun
    comet_position_m: np.ndarray,              # comet's position at the outburst epoch (SI, metres)
    comet_velocity_m_s: np.ndarray,            # comet's velocity at the outburst epoch (SI, m/s)
    sun_position_m: np.ndarray | None = None,  # only needed for TOWARD_SUN mode
    rng: np.random.Generator,                  # the random number generator to draw from
) -> list[DustParticle]:
    """Generate the full particle population for one scenario.

    Equivalent to the old code's nested
        for (particlenumber in 1..maxparticles)
            for (bvalue in <active bins>)
    loop, flattened into one list instead of being interleaved with
    per-particle simulation/propagation (see simulation.py for why
    generation and propagation are kept as two separate steps here).
    """
    # Fail fast with a clear message if the caller asked for TOWARD_SUN mode
    # but forgot to also supply the Sun's position.
    if direction_mode is EjectionDirectionMode.TOWARD_SUN and sun_position_m is None:
        raise ValueError("sun_position_m is required for EjectionDirectionMode.TOWARD_SUN")

    particles: list[DustParticle] = []   # the growing list of finished DustParticle objects
    particle_id = 0                      # running counter, becomes each particle's unique id
    for size_bin in bins:                # outer loop: once per size class (e.g. b1_small, b2_medium, ...)
        for _ in range(n_particles_per_bin):   # inner loop: n_particles_per_bin times per size class
            # 1. Pick this particle's own random radius, within its bin's range.
            radius_mm = sample_particle_radius_mm(rng, size_bin)
            # 2. Pick this particle's ejection direction, according to the
            #    configured mode (theta/u are kept for the legacy output).
            if direction_mode is EjectionDirectionMode.ISOTROPIC_SPHERE:
                theta, u = sample_isotropic_theta_u(rng)
                direction = direction_from_theta_u(theta, u)
            else:
                theta, u = float("nan"), float("nan")
                direction = sample_toward_sun_direction(comet_position_m, sun_position_m)
            # 3. beta, ejection speed and the initial state all follow from
            #    radius + direction (see make_particle above).
            particles.append(
                make_particle(
                    particle_id=particle_id,
                    bin_name=size_bin.name,
                    radius_mm=radius_mm,
                    direction=direction,
                    comet_position_m=comet_position_m,
                    comet_velocity_m_s=comet_velocity_m_s,
                    theta=theta,
                    u=u,
                )
            )
            particle_id += 1   # next particle gets the next id
    return particles   # the complete population, ready for simulation.py
