# sim-git

Particle deposition in the human nasal airway: m-AIA case setups and the Python
post-processing that turns solver output into deposition statistics and ParaView
files.

## Layout

| path | contents |
|---|---|
| `src/` | post-processing scripts (see below) |
| `setup/<case>/` | one m-AIA case each: `properties_*.toml`, `geometry.toml`, `stl.zip`, `part.txt`, SLURM `run.sh`, `env_maia` |
| `results/` | deposition point clouds from past runs (`dep_*.vtp`) |
| `*_centerline.dat` | left/right airway centerlines (VMTK output), used for penetration depth |
| `particle.log`, `partData_particle_0.Netcdf` | solver output of the current working case |

## Install

```sh
uv sync            # Python >= 3.10
```

## Workflow

### 1. Generate particle initial conditions

`src/particles.py` has one subcommand per sampling method. Each writes
`<stem>.txt` for the solver (columns: diameter, density, x, y, z) and
`<stem>.vtp` for ParaView. Copy the `.txt` into the case directory.

```sh
python src/particles.py sphere --center X Y Z --radius R [--stl FILE] ...
python src/particles.py stl FILE [--normal-offset D] [--scale S] [--flip-normal]
python src/particles.py box --min X Y Z --max X Y Z
```

Options common to all three: `-n` count (default 56000), `-o` output stem
(default `part`), `--diameters` to draw uniformly from (default: the 14 values
from 0.1 to 10 µm), `--density` (default 1000), `--seed` for a reproducible set,
`--plot` for a histogram PNG. `python src/particles.py sphere -h` lists the rest.

- **`stl`** samples a surface and steps each point off it along the normal.
- **`box`** fills an axis-aligned box.
- **`sphere`** fills the shell `--r-min <= r <= --radius` with a radial
  `--profile`:
  - `uniform` (default) — constant density; the count per unit radius grows as r².
  - `inverse_square` — density ∝ 1/r², so every radius gets the same count.

  With `--stl`, it keeps only points outside that surface (`--inside` inverts
  this), at least `--clearance` mm from it. The profile is applied before the
  surface test, so where the sphere overlaps the surface the delivered count is
  the profile times the locally allowed fraction.

  The STL **must be a closed surface** — the test is the signed distance to it,
  which is meaningless on an open one, so an STL with open edges is refused.
  `particle_filter.stl` is closed and encloses the head and airway;
  `breathing_tract.stl` and the cap patches are not.

#### Example: filtered sphere around the nostril

From the repo root: 56,000 particles in a 120 mm sphere centred on the nostril,
outside `particle_filter.stl` with 1 mm clearance, equal count per radius.
Writes `part_sphere.txt` and `part_sphere.vtp`:

```sh
python src/particles.py sphere --center -3.5 1108 1595 --radius 120 \
    --stl particle_filter.stl --clearance 1 \
    --profile inverse_square --r-min 10 --seed 42 -o part_sphere
```

Nothing survives inside r ≈ 8 mm here — the centre is inside the head — hence
`--r-min 10`. Raising `--clearance` lowers the acceptance (0.69 at 1 mm, 0.63
at 5 mm) but still delivers the full count.

The same is available from Python via `Particles.sample_position(method=...)`,
with the options as keyword arguments (`stl_file`, `clearance`, `inside_stl`,
`profile`, `r_min`, ...).

### 2. Run the solver

On the cluster, from inside a case directory:

```sh
sbatch run.sh              # flow only
sbatch run_part.sh         # flow + particles, where the case has one
```

Relevant output: `particle.log` (one line per removed particle),
`partData_particle_0.Netcdf` (initial conditions), `point_data_*.Netcdf` (probes).

### 3. Post-process

Run from the directory holding the solver output. All scripts read and write the
working directory.

```sh
# deposition statistics + 5 figures + particles.vtp
python src/read_part.py

# .vtp from the log alone, no NetCDF needed
python src/read_part.py --log particle.log --vtp-only -o particles.vtp

# per-diameter deposition rate -> result.csv
python src/deposition_stats.py particle.log

# validation against Ito et al.; expects steady_state_validation.csv
# and {7.5,15,30}l/result.csv from the step above
python src/plot_stats.py

# probe time series: extract, then plot
python src/process_point_data.py -i . -o probes.dat
python src/plot_probes.py
```

`read_part.py --help` lists the input paths (`--log`, `--ic`, `--left`, `--right`).
It produces `particles.vtp`, `depth_vs_distance.pdf`, `depth_pdf_by_diam.pdf`,
`depth_pdf_all.pdf`, `diam_pdf_by_status.pdf` and
`deposition_fraction_vs_ic_distance.pdf`.

## particle.log format

Whitespace separated, one line per removed particle. Columns used downstream
(0-based):

| col | meaning |
|---|---|
| 1 | timestep |
| 2 | particle id, with extra bits packed above the low 32 |
| 3 | diameter |
| 4–6 | position at removal |
| 9 | **1 = deposited on the wall, 0 = removed without depositing** |

A line means the particle was removed, not that it deposited — column 9 is the
only thing that says which. A particle can appear more than once; the last
appearance wins.

## Particle fate

`read_part.py` classifies every initialized particle:

| status | meaning |
|---|---|
| 0 | never removed inside the valid region |
| 1 | deposited on the airway wall |
| 2 | inhaled — removed in-region without depositing, i.e. left through the outlet |

Status 2 assumes the outlet is the only way out of the valid region. The script
prints the shallowest inhaled depth as a check: it should sit just under the
terminal centerline depth.

## Case-specific constants

Two things in `src/read_part.py` are tied to this geometry and need revisiting
for a different airway:

- `spatial_filter()` — the y/z cutoffs that discard particles removed upstream of
  the nose.
- `NOSTRIL` — the reference point for the initial-condition distance and
  projected velocity.
