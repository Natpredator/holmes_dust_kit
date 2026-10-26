"""
holmes_dust
===========
Clean-room Python/REBOUND/REBOUNDx redesign of the old Java/Orekit
"SlaveMode_Holmes*" comet-dust-trail simulator.

See README.md at the repository root for the architecture overview and the
mapping from old Java flags to the configuration objects in this package.

Module map
----------
config.py        - scenario & simulation parameters (pure data)
scenarios.py     - concrete historical scenarios (epoch table + flag profiles)
ephemeris.py     - state vectors for Sun/planets/comet (Horizons + legacy table)
particles.py     - dust particle generation (radius -> beta -> ejection speed/direction)
constants.py     - physical constants (G, planetary GM values)
simulation.py    - REBOUND/REBOUNDx simulation building & propagation
observations.py  - RA/DEC, phase angle, light-time correction
io_export.py     - streamed CSV output (full trajectory + named snapshots)
run_scenario.py  - command-line entry point tying all of the above together

This file (__init__.py) runs automatically the very first time anyone does
`import holmes_dust` or `from holmes_dust import ...`. Its only job here is
to re-export the handful of names people will want most often, so they can
write `from holmes_dust import Body` instead of the longer
`from holmes_dust.config import Body`.
"""

# Pull these specific names out of config.py (see config.py for what each
# one actually is/does) so this package's top level can re-offer them.
from .config import (
    Body,                       # enum: identifies Sun/planets/comet/Earth
    EjectionDirectionMode,      # enum: how dust ejection direction is picked
    EphemerisSource,            # enum: where position/velocity data comes from
    IntegrationConfig,          # dataclass: propagation settings (integrator, duration, ...)
    OutputFormat,               # enum: modern CSV / legacy Java CSV layout / both
    ParticleGenerationConfig,   # dataclass: how many particles, which bins, which RNG seed
    ParticleSizeBin,            # dataclass: one dust-size class (e.g. "small", radius range)
    RadiationModel,             # enum: which radiation-pressure model to use
    ScenarioConfig,             # dataclass: the top-level "everything for one run" object
    SnapshotSpec,                # dataclass: one named output-date slice (forecast/matching)
)

# __all__ controls what `from holmes_dust import *` would import, and also
# documents, in one place, exactly which names this package considers its
# public surface. Every name here must match something imported above.
__all__ = [
    "Body",
    "EjectionDirectionMode",
    "EphemerisSource",
    "IntegrationConfig",
    "OutputFormat",
    "ParticleGenerationConfig",
    "ParticleSizeBin",
    "RadiationModel",
    "ScenarioConfig",
    "SnapshotSpec",
]
