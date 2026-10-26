# Comet Dust Trail Simulator (17P/Holmes & 10P/Tempel 2)

A Python/REBOUND N-body simulator that models the ejection, propagation and
observability of dust particles released during a cometary outburst. It is a
clean-room reimplementation of an earlier Java/Orekit simulator, built around
[REBOUND](https://rebound.readthedocs.io/) for gravitational dynamics,
[REBOUNDx](https://reboundx.readthedocs.io/) for radiation pressure and
Poynting-Robertson drag, and JPL ephemerides
([Horizons](https://ssd.jpl.nasa.gov/horizons/) online, or a local `.bsp`
file offline) for initial conditions.

The package can also **reproduce the old Java program's output**: same files,
same columns, same particles, same force model. When two old Java runs of the
1892 outburst were reproduced particle by particle over the full 137 years,
the median difference from the Java positions was 0.006–0.009° on the sky (see
[Validation](#validation)).

## Contents

1. [What it does](#what-it-does)
2. [Installation](#installation)
3. [How to run](#how-to-run)
4. [What the output looks like](#what-the-output-looks-like)
5. [Checking the model against old Java results](#checking-the-model-against-old-java-results)
6. [Command-line reference](#command-line-reference)
7. [Available scenarios](#available-scenarios)
8. [Observation planning](#observation-planning)
9. [Validation](#validation)
10. [Project structure](#project-structure)
11. [Running the test suite](#running-the-test-suite)
12. [Notes for reviewers](#notes-for-reviewers)
13. [Authors](#authors)
14. [License](#license)

## What it does

For a chosen historical comet outburst (e.g. 17P/Holmes, 1892-11-04):

1. Gets the state vectors of the comet, the Sun and the planets at the
   outburst epoch.
2. Generates virtual dust particles. Each particle gets a random radius and a
   random ejection direction. Its radiation-pressure coefficient (`beta`) and
   ejection speed follow from the radius.
3. Integrates all particles together in one N-body simulation (Sun + planets,
   optionally the comet nucleus), with solar radiation pressure modelled in
   one of two ways:
    - `reboundx_beta` (default): REBOUNDx `radiation_forces`, which includes
      Poynting-Robertson drag;
    - `legacy_sun_gm_reduction`: exactly the old Java force model, a radial
      push of `beta · GM_sun / r²` with no PR drag.
4. **Once per simulated day** (at 12:00 UTC, the same as the Java program),
   records each particle's position, geocentric RA/DEC and phase angle, and
   writes them to CSV files, in the modern layout, the old Java layout, or both.

```
epoch + profile ──► initial states (Horizons / local .bsp / Java table)
                ──► dust particles (radius → beta → speed, direction)
                ──► REBOUND integration, one output row per particle per day
                ──► RA/DEC, phase angle
                ──► CSV: modern layout and/or old Java layout
```

## Installation

You need Python 3.9 or newer.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (via WSL2)

`reboundx` cannot be built on native Windows. Install WSL2 in an
administrator PowerShell:

```powershell
wsl --install
```

Reboot if prompted. Then run these commands inside the Ubuntu (WSL) window:

```bash
sudo apt update
sudo apt install -y build-essential python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `reboundx` fails to build against an already-installed `rebound`:

```bash
pip uninstall -y reboundx
pip install --no-build-isolation reboundx
```

### Optional: work without internet

By default, initial states and the daily Earth positions are downloaded from
JPL Horizons, and cached in `.horizons_cache/`. To work offline, download a
JPL planetary ephemeris file once, for example `de440.bsp` (covers
1550–2650, ~115 MB) from `https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/`.
Then add `--spk de440.bsp` to any command below. In this mode the comet's
start state comes from the table copied from the Java program.

## How to run

All commands are run from the `holmes_dust_kit/` folder. Each block below is
**one command**; the `\` at the end of a line means "continued on the next
line", so copy the whole block.

### Step 1 – check the installation (a few seconds)

```bash
python run_scenario.py --epoch 1892-11-04 --profile 1892_no_comet_gravity \
    --n-particles-per-bin 5 --duration-days 30 --seed 42
```

This should finish in a few seconds and write `output/trajectory_full.csv`.

### Step 2a – a run with the same settings as the Java program

This mirrors `SlaveMode_Holmes1892_Refactor_1892.java`: 400 particles of
size bin b1, all planets, no comet gravity, the Java comet start state, the
Java force model (no PR drag), RA/DEC without light-time correction, and the
output written in the Java file layout:

```bash
python run_scenario.py --epoch 1892-11-04 --profile 1892_no_comet_gravity \
    --ephemeris-source horizons_with_legacy_comet \
    --radiation-model legacy_sun_gm_reduction --no-light-time \
    --output-format legacy --output-dir ./output/holmes_1892_legacy --seed 42
```

To run offline, replace `--ephemeris-source horizons_with_legacy_comet` with
`--spk de440.bsp`.

The particles are random, just as in Java. A different `--seed` gives a
different set, and no seed reproduces the Java program's own random numbers.
To recompute *exactly the particles of an existing Java run*, use
[`--reproduce-java`](#checking-the-model-against-old-java-results).

### Step 2b – a run with the full modern physics

Initial states from Horizons, REBOUNDx radiation forces with
Poynting-Robertson drag, light-time correction, and both output layouts:

```bash
python run_scenario.py --epoch 1892-11-04 --profile 1892_no_comet_gravity \
    --output-format both --output-dir ./output/holmes_1892_full --seed 42
```

### Long runs: time, disk space, running in the background

- A full 1892 run covers 50140 days (1892-11-04 → 2030-02-14), so it writes
  50141 rows per particle.
- It takes roughly 20 minutes to 2 hours, depending on the machine, the
  number of particles and the force model. The legacy force model with a
  local `.bsp` file is the fastest option.
- Disk space for 400 particles: about 5 GB for `legacy`, about 7 GB for
  `modern`, and the sum of the two for `both`.

Run long jobs in the background and follow the progress in the log:

```bash
nohup python run_scenario.py --epoch 1892-11-04 --profile 1892_no_comet_gravity \
    --output-format legacy --output-dir ./output/run1 --seed 42 > run1.log 2>&1 &

tail -f run1.log      # progress every 100 simulated days; Ctrl+C only stops watching
```

## What the output looks like

The output rate is **one row per particle per simulated day**, the same as
the Java `stepT = 86400 s`. The rate is set by `output_cadence_s` in
`holmes_dust/config.py`, and the run length by the profile (or by
`--duration-days`).

`--output-format` chooses how those rows are split into files:

| Layout | File | Contents |
|---|---|---|
| `modern` | `trajectory_full.csv` | every particle on every day, one row each, with a header row |
| `modern` | `snapshot_f10.csv` … `snapshot_f22.csv` | every particle on one forecast date |
| `modern` | `snapshot_m2.csv`, `snapshot_m14.csv` | particles near a real observation on its date |
| `legacy` | `b1p1.csv` … `b1p400.csv` | **one file per particle**, one row per day (exactly the Java p-files) |
| `legacy` | `b1f10.csv` … `b1f22.csv` | **one file per forecast date**, one row per particle (Java f-files) |
| `legacy` | `b1m2.csv`, `b1m14.csv` | matching files (Java m-files) |

`both` writes all of the above. The forecast and matching dates are the
ones of the Java program:

| File | Date | Condition |
|---|---|---|
| f10, f11, f12 | 2026-08-01, 2026-10-01, 2026-12-01 | all particles |
| f13, f14, f15 | 2027-02-01, 2027-03-01, 2027-08-01 | all particles |
| f16, f17, f18 | 2027-10-01, 2027-12-01, 2028-02-01 | all particles |
| f19, f20, f21, f22 | 2028-10-01, 2029-02-01, 2029-10-01, 2030-02-01 | all particles |
| m2 | 2013-02-17 | RA within 242° ± 40° and DEC within −40.4° ± 40° |
| m14 | 2015-02-15 | RA within 21° ± 40° and DEC within 27.2° ± 40° |

### Columns of the Java layout

There is no header row. The names, columns and number formatting are the
same as in the Java program; this was checked byte for byte against 10,000
real Java lines. Each p-file line is:

```
JD, step K, date, x, y, z, RA_hms, DEC_dms, RA_deg, DEC_deg, radius_mm, theta, u, epoch_index, phase_deg
```

- `x, y, z`: position relative to the Solar-System Barycentre (ICRF axes), in AU.
- RA/DEC: geocentric.
- `radius_mm, theta, u`: the particle's radius and the two random numbers of
  its ejection direction. They are the same on every line.
- f- and m-files have the same columns, minus `step K`.

Three differences from the Java files are intentional:

- `step` counts 1, 2, 3, … The Java counter skipped numbers.
- The time label is the true 12:00 UTC. The Java labels drifted to
  `11:59:23` through the leap seconds.
- Every file is complete and properly closed.

### Columns of the modern layout

| Column | Meaning |
|---|---|
| `julian_day`, `date_utc` | Date of the row (`julian_day` = Julian Day Number, as in the Java files) |
| `particle_id`, `bin_name` | Which particle, which size class |
| `radius_mm`, `beta`, `ejection_speed_m_s` | Physical parameters of the particle |
| `geocentric_x_au` / `_y_au` / `_z_au` | Position relative to Earth (as used for RA/DEC) |
| `barycentric_x_au` / `_y_au` / `_z_au` | Position relative to the Solar-System Barycentre, the same as Java x/y/z |
| `ra_deg`, `dec_deg`, `ra_hms`, `dec_dms` | Geocentric Right Ascension / Declination |
| `phase_angle_deg` | Sun–particle–Earth angle |
| `light_time_s` | Light-travel time used for the correction (0 when it is off) |
| `theta`, `u` | The two random numbers of the ejection direction |

## Checking the model against old Java results

The most direct test of the REBOUND model is to take the particles of an
existing Java run, integrate exactly the same particles again, and compare
the positions day by day. Every Java line contains the particle's radius,
theta and u, and those three numbers fully define the particle.

You need the Java results folder (`orekit-results`, with `b1p*.csv` and/or
`b1f*.csv` files) and three commands:

```bash
# 1. Look at the folder: epoch, particles, complete/damaged files
python tools/inspect_java_results.py /path/to/orekit-results

# 2. Integrate the same particles again with the Java settings (legacy force, no light-time)
python run_scenario.py --epoch 1892-11-04 --profile 1892_no_comet_gravity \
    --reproduce-java /path/to/orekit-results --output-dir ./output/repro

# 3. Compare
python tools/compare_with_java.py /path/to/orekit-results ./output/repro
```

`--reproduce-java` turns on everything needed for an exact comparison: the
Java comet start state, the Java force model, no light-time correction, and
both output layouts. If some p-files are missing (deleted to save space, for
example), those particles are rebuilt from the forecast files, because line
*k* of a Java forecast file belongs to particle *k*. Add `--spk de440.bsp` to
run offline.

**Reading the comparison.** The tool prints the RA/DEC difference between
the new run and Java for each particle and day, and for every forecast file.
For a correct model, expect:

- day 0–1000: about 0.0003° or less. This residual is a 32-second time-scale
  convention for the Earth, not a dust-position difference.
- end of the run (2030): a median around 0.01°, with roughly 90% of the
  particles within 0.1°.
- a few particles far off, by degrees or more. These particles passed close
  to Jupiter, which amplifies tiny ephemeris differences (chaos). Their
  number is small, and the median is not affected.

**Other helpers.** `tools/rebuild_legacy_snapshots.py IN OUT` regenerates
complete f/m files from the p-files. The Java program did not close its f/m
files, so some of them are cut short or empty.

## Command-line reference

| Flag | Description | Default |
|---|---|---|
| `--epoch` | Historical outburst date (see [Available scenarios](#available-scenarios)) | required |
| `--profile` | Physics profile | required |
| `--ephemeris-source` | `jpl_horizons`, `horizons_with_legacy_comet`, `spk_file_with_legacy_comet`, `legacy_table` | `jpl_horizons` |
| `--spk FILE.bsp` | Planets/Earth from a local JPL file, comet start state from the Java table (offline) | – |
| `--radiation-model` | `reboundx_beta` (with PR drag) or `legacy_sun_gm_reduction` (Java) | profile (`reboundx_beta`) |
| `--no-light-time` | RA/DEC without light-time correction, as in Java | correction on |
| `--output-format` | `modern`, `legacy` (Java layout) or `both` | `modern` |
| `--output-dir` | Folder for the output files | `./output` |
| `--n-particles-per-bin` | Number of particles per size class | profile (400 for 1892) |
| `--duration-days` / `--duration-years` | Length of the run | profile (50140 days for 1892) |
| `--seed` | Random seed for the particles | random |
| `--reproduce-java DIR` | Recompute the particles of an old Java run (see above) | – |
| `--max-particles N` | With `--reproduce-java`: only the first N particles | all |
| `--only-complete` | With `--reproduce-java`: skip particle files that are cut short | off |

`python run_scenario.py --help` lists all the flags with their allowed values.

## Available scenarios

**Epochs** (`--epoch`):

| Epoch | Comet | Java index | Notes |
|---|---|---|---|
| `2007-10-20` … `2007-10-28` | 17P/Holmes | 0–8 | Candidate dates around the 2007 outburst |
| `1892-11-04` | 17P/Holmes | 9 | Discovery outburst |
| `1893-01-16` | 17P/Holmes | 10 | Secondary flare-up |
| `1899-07-04`, `2015-01-26` | 17P/Holmes (assumed) | 11, 12 | Comet not named in the original source |
| `1873-06-25`, `1894-04-23`, `1904-11-10` | 10P/Tempel 2 | 13–15 | Labelled 10P/Tempel 2 in the original source |

**Profiles** (`--profile`). The run lengths are the exact values hard-coded
in the Java programs:

| Profile | Mirrors Java file | Planets | Comet gravity | Bins | Particles | Length | f/m files |
|---|---|---|---|---|---|---|---|
| `2007_major_planets_only` | `SlaveMode_Holmes2007*.java` | Venus–Saturn | on | b1, b2, b3 | 3 × 50 | 6205 d | – |
| `2007_all_planets_extended` | `…Refactor_2007.java` | Mercury–Neptune | on | b1, b2, b3 | 3 × 500 | 9131 d | f10–f25, m2, m14 |
| `1892_no_comet_gravity` | `…Refactor_1892.java` | Mercury–Neptune | off | b1 | 400 | 50140 d | f10–f22, m2, m14 |
| `historical_candidate_default` | `…Refactor.java` | Mercury–Neptune | off | b1, b2, b3 | 3 × 200 | 48910 d | f10–f22, m2, m14 |

Size bins (particle radius): b1 = 0.001–0.01 mm, b2 = 0.01–0.1 mm,
b3 = 0.1–1 mm.

## Observation planning

`tools/plan_observations.py` generalises the project's Skyfield scripts
(`holmes_<year>_<date>_<site>.py`). It writes the same hourly text lines
(RA/DEC, Alt/Az) for Kouvola, Utah, Siding Spring, Rio Hurtado and Feregnal.
It needs `pip install skyfield`.

```bash
# a fixed point in space: identical output to the old scripts
python tools/plan_observations.py \
    --point -1.790431897831839 -2.1448241371064594 -2.273927653630378 \
    --label Kouvola_1892_2027_2_FILAMENT1 --site Kouvola --start 2027-01-01T00 --hours 8000

# the dust within 0.1 AU of that point on 2027-02-01, followed as it moves
python tools/plan_observations.py --track ./output/repro --near -1.79 -2.14 -2.27 \
    --on 2027-02-01 --radius 0.1 --site Kouvola Utah --start 2027-01-15T00 --hours 720 --out plans/
```

The `--point` output is byte-for-byte identical to the original
`holmes_1892_2027_2_filament1_kouvola.py`. The `--track` mode exists because
the dust moves: a particle at that point on 2027-02-01 moves about 0.01 AU a
day, and a month earlier or later its sky position is about 2° away from the
fixed point.

## Validation

Checks performed on real output (REBOUND 5.2 / REBOUNDx 5.1):

**1. Exact reproduction of two Java runs, particle by particle** (1892
outburst, legacy force model, DE405 planets, 50140 days):

| Days after outburst | 1 | 100 | 1000 | 10000 | 30000 | 50140 (2030) |
|---|---|---|---|---|---|---|
| Run A (78 particles): median sky difference (deg) | 0.0003 | 0.00006 | 0.00004 | 0.0008 | 0.004 | 0.006 |
| Run A: median position difference (AU) | 1e-11 | 9e-8 | 2e-6 | 9e-5 | 4e-4 | 6e-4 |
| Run B (100 particles with p-files): median sky difference (deg) | 0.0003 | 0.00006 | 0.00004 | 0.001 | 0.004 | 0.009 |

- In run B, all 200 particles were compared at the 13 forecast dates
  (2026–2030). 100 of them were rebuilt from the forecast files. The median
  difference was 0.006–0.014°, with about 90% of the particles within 0.1°.
- At the end of run A, 50 of the 78 particles are within 0.01° of Java and
  76 are within 1°. In run B, 89 of 100 are within 0.1° and 95 within 1°.
- The few particles that end up far off had close encounters with Jupiter
  (0.05–0.26 AU). Such encounters amplify any tiny difference in the
  planetary ephemeris. That is expected chaotic behaviour, not a model error.
- The constant 0.0003° at day 0 comes from the Earth position, not from the
  dust. This package treats the epochs as TDB, Orekit as UTC, 32.184 s apart.
  Evaluating the Earth 32.184 s later reduces the difference to 0.00003°.

**2. The physics the Java program left out.** The same particles integrated
with REBOUNDx radiation forces, which include Poynting-Robertson drag,
differ from Java by a median of 0.8° after 137 years. That difference is the
real effect of PR drag, not a numerical error.

**3. An independent run agrees statistically.** A normal run of this package
(400 new random particles, Horizons initial conditions) places the centre of
the dust cloud within 0.03–0.1° of the Java cloud over the first 570 days,
with a similar spread. The remaining 0.09° comes from the slightly different
1892 orbit of 17P that Horizons serves today.

**4. Unit tests with real REBOUND.** The legacy force moves a grain exactly
like a Sun of mass (1 − β) would; the two agree to 1e-9 AU after 8 years
(`tests/test_real_rebound.py`).

## Project structure

```
holmes_dust_kit/
├── run_scenario.py              # command-line entry point
├── requirements.txt
├── holmes_dust/
│   ├── config.py                # all settings (dataclasses, enums), incl. output cadence
│   ├── scenarios.py             # outburst epochs + ready-made physics profiles
│   ├── ephemeris.py             # Horizons client, offline .bsp reader, Java start states
│   ├── constants.py             # physical constants (G, planetary GM values)
│   ├── particles.py             # dust particle generation
│   ├── simulation.py            # REBOUND / REBOUNDx integration, legacy force
│   ├── observations.py          # RA/DEC, phase angle, light-time correction
│   ├── io_export.py             # modern CSV output
│   └── legacy_java.py           # Java file layout: write, read, rebuild
├── tools/
│   ├── inspect_java_results.py      # summary of one Java results folder
│   ├── compare_with_java.py         # new run vs Java, per particle and per forecast file
│   ├── rebuild_legacy_snapshots.py  # complete f/m files from p-files
│   ├── plan_observations.py         # hourly Alt/Az per observatory (Skyfield)
│   └── check_against_horizons.py    # RA/DEC here vs Horizons' own observer ephemeris
└── tests/                       # unit tests, offline (fakes for rebound/reboundx)
```

Every module starts with a docstring explaining its role, and the code is
commented throughout.

## Running the test suite

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v       # 81 tests, offline
PYTHONPATH=. python3 -m unittest tests.test_real_rebound -v # physics checks with real rebound
```

The main suite needs neither the network nor `rebound`/`reboundx`; the
REBOUND-dependent orchestration is checked with lightweight fakes. The
real-REBOUND physics tests must run in their own process, as shown above.
The offline-ephemeris test runs when `HOLMES_TEST_SPK=/path/to/de440.bsp` is
set.

## Notes for reviewers

- The `beta(radius)` and `ejection_speed(radius)` formulas
  (`holmes_dust/particles.py`) are ported verbatim from the original program.
  The original only credits them to author names in comments, so the exact
  literature source has not been verified independently.
- The assumed nucleus mass used when comet self-gravity is on
  (`comet_assumed_gm_m3_s2`) is carried over from the original as an
  assumption, not a measured value.
- The legacy force model keeps the Sun at full mass and adds the outward term
  `beta·GM/r²` to each grain. For the grain this is identical to the Java
  `sun.setGM(mu*(1-beta))`, while the integrated planets keep their correct
  orbits. Lowering the Sun's mass in an N-body code would shift Jupiter by
  tens of degrees over a century.
- The simulation stays in the Solar-System-Barycentre frame used by Horizons
  and by the Earth positions for RA/DEC. `IntegrationConfig.recenter_to_com`
  is off by default.
- Two epochs (`1899-07-04`, `2015-01-26`) are tagged as 17P/Holmes by
  inference from context. The source does not label them explicitly.

## Authors

- **Markku Nissinen** (contact person), Ursa Astronomical Association and Taurus Hill Observatory
- **Maria Gritsevich** (project leader), Instituto de Astrofísica de Andalucía and Ursa Astronomical Association
- **Grigorii Razuvaev**
- **Arsenii Zhuravlev**
- **Jorma Ryske**, Ursa Astronomical Association

This software accompanies the Zenodo record
[Recommendations for Observing the Dust Trails from the 1892 and 2007 Outbursts of Comet 17P/Holmes](https://zenodo.org/records/23297613).

## License

Released under the [MIT License](LICENSE). Note that the required
dependencies [REBOUND](https://github.com/hannorein/rebound) and
[REBOUNDx](https://github.com/dtamayo/reboundx) are distributed under their
own licenses.
