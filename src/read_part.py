import numpy as np
import netCDF4 as nc
import pandas as pd
import subprocess
from subprocess import PIPE, run
import glob
import sys
import os
import math
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def spatial_filter(
      df
    ):
    '''Keep only particles inside the valid region, drop the rest.'''
    y, z = df["y"], df["z"]
    remove = (
        (y < 1095) |
        ((y >= 1095) & (y < 1125) & (z < 1585)) |
        ((y >= 1125) & (y < 1185) & (z < 1550))
    )
    return df[~remove].reset_index(drop=True)

def df_to_vtp(
      df,
      fileName = "particles.vtp"
    ):

    df = spatial_filter(df)

    # scalars
    time        = df["t"].to_numpy()
    partId      = df["id"].to_numpy()
    diam        = df["diam"].to_numpy()
    # vectors
    position    = df[["x", "y", "z"]].to_numpy()
    position_ic = df[["ic_x", "ic_y", "ic_z"]].to_numpy()
    velocity_ic = df[["ic_u", "ic_v", "ic_w"]].to_numpy()
    
    n = len(partId)
    conn = " ".join(map(str, range(n)))
    offs = " ".join(map(str, range(1, n+1)))
    asf = lambda a: " ".join(f"{x:.9g}" for x in np.ravel(a))

    with open(fileName, "w") as f:
        f.write(f'''<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
<PolyData>
  <Piece NumberOfPoints="{n}" NumberOfVerts="{n}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">
    <PointData Scalars="time" Vectors="partVel">
      <DataArray type="Float32" Name="IC velocity" NumberOfComponents="3" format="ascii">{asf(velocity_ic)}</DataArray>
      <DataArray type="Float32" Name="IC position" NumberOfComponents="3" format="ascii">{asf(position_ic)}</DataArray>
      <DataArray type="Float32" Name="time" format="ascii">{asf(time)}</DataArray>
      <DataArray type="Float32" Name="globalParticleID" format="ascii">{asf(partId)}</DataArray>
      <DataArray type="Float32" Name="diameter" format="ascii">{asf(diam)}</DataArray>
      <DataArray type="Float32" Name="Deposition time" format="ascii">{asf(time)}</DataArray>

    </PointData>
    <CellData/>
    <Points>
      <DataArray type="Float32" NumberOfComponents="3" format="ascii">{asf(position)}</DataArray>
    </Points>
    <Verts>
      <DataArray type="Int32" Name="connectivity" format="ascii">{conn}</DataArray>
      <DataArray type="Int32" Name="offsets" format="ascii">{offs}</DataArray>
    </Verts>
  </Piece>
</PolyData>
</VTKFile>''')
    

def read_particle_netcdf(
      fileName
    ):

    '''Read and convert a NetCDF particle file into Pandas dataframe. '''
    data = nc.Dataset(fileName, mode='r').variables
    len = data['partPos'].shape[0]//3
    array = np.empty(shape=(len, 8)) # x,y,z, u,v,w, diameter and partID
    array[:,0:3] = np.reshape(data['partPos'], shape = (len,3))
    array[:,3:6] = np.reshape(data['partVel'], shape = (len,3))
    array[:,6]   = np.reshape(data['partDia'], shape = (len,))
    array[:,7]   = np.reshape(data['globalPartID'], shape = (len,))
    df = pd.DataFrame(array, columns=["x", "y", "z", "u", "v", "w", "d", "id"])
    df["id"] = df["id"].astype("int")
      # extra post-processing
    origin = np.array([-3.5, 1108.0, 1595.0])
    rel = df[["x", "y", "z"]].values - origin
    df["velocity magnitude"] = df[["u","v","w"]].pow(2).sum(axis=1).pow(0.5)
    df["distance to origin"] = np.linalg.norm(rel, axis=1)
    df["projected velocity"] = - (
        df[["u", "v", "w"]].values * rel
      ).sum(axis=1) / df["distance to origin"]
    return df

def read_particle_log(
      fileName = 'particle.log'
    ):

    '''Read particle.log file and convert it into Pandas dataframe.'''
    array = np.loadtxt(partLogfileName, usecols=range(1, 7))
    df =  pd.DataFrame(array, columns = ["t", "id", "diam", "x", "y", "z"])
    df[["t", "id"]] = df[["t", "id"]].astype("int")
    df["id"] = np.int32(df["id"] & 0xFFFFFFFF)

    # a particle can appear more than once; keep only its latest appearance
    df = df.drop_duplicates("id", keep="last").reset_index(drop=True)

    # keep only particles inside the valid region, drop the rest
    df = spatial_filter(df)
    return df


def add_IC(
      df_ic, 
      df_log
    ):
    '''Use both log-file and initial particle file dataframes and append them into one.'''
    log_ic = df_log[["id"]].merge(df_ic, on="id").reset_index(drop=True)
    log_ic.columns = ['ic_id', 'ic_x', 'ic_y', 'ic_z', 'ic_u', 'ic_v', 'ic_w', 'diam', 'ic velocity', 'ic distance to origin', 'ic projected velocity']
    log_ic = log_ic.drop(columns=['diam'])
    if not (log_ic["ic_id"] == df_log["id"]).all():
      raise ValueError("ic_id and id columns do not match")
    df = pd.concat([df_log,log_ic],axis=1)
    df = df.drop(columns=["ic_id"])
    return df

def read_centerlines(
      leftName = "left_centerline.dat", 
      rightName = "right_centerline.dat"
    ):

    args = dict(
        sep=r"\s+",
        header=0,
        usecols=[0, 1, 2],
    )

    # flip due to centerline beginning from the trachea
    left  = pd.read_csv(leftName,  **args).iloc[::-1].reset_index(drop=True)
    right = pd.read_csv(rightName, **args).iloc[::-1].reset_index(drop=True)

    return mean_centerline(left, right)

def mean_centerline(
      left,
      right,
      n = None
    ):
    '''Average the left and right centerlines into a single mean centerline.

    The two lines may have different point counts, so each is resampled onto a
    common normalized arc-length parameter before averaging.
    '''
    def resample(df, s):
        coords = df[["X", "Y", "Z"]].to_numpy()
        seg = np.concatenate(([0.0],
                np.linalg.norm(np.diff(coords, axis=0), axis=1).cumsum()))
        t = seg / seg[-1]
        return np.column_stack([np.interp(s, t, coords[:, k]) for k in range(3)])

    if n is None:
        n = max(len(left), len(right))
    s = np.linspace(0.0, 1.0, n)
    mean = 0.5 * (resample(left, s) + resample(right, s))
    center = pd.DataFrame(mean, columns=["X", "Y", "Z"])
    return compute_distance(center)

def compute_distance(
    df 
  ):
    # compute the difference in coordinates between two neighbouring points 
    # and use that to build a "length"/"depth" array
    d = pd.concat([pd.Series([0]), (df[["X","Y","Z"]].diff().iloc[1:].pow(2).sum(axis=1).pow(0.5)).cumsum()])
    df["depth"] = d
    return df 

def add_addtributes(df, center):
  # Convert needed information into NumPy arrays
  points_df     = df[["x", "y", "z"]].to_numpy()
  points_center = center[["X", "Y", "Z"]].to_numpy()
  depths        = center["depth"].to_numpy()

  # Build KD-tree on the mean centerline and query the nearest point
  tree = cKDTree(points_center)
  nearest_dist, idx = tree.query(points_df)

  # Write back to a copy of df
  out = df.copy()
  out["distance"] = nearest_dist
  out["depth"]    = depths[idx]

  return out

def combine_particles(df_ic, out):
    '''Build the full particle dataframe (all initialized particles) with a
    deposition status flag and, where deposited, the penetration depth/distance.

    status 0 : not deposited          -> penetration depth / distance are NaN
    status 1 : deposited
    status 2 : deposited at the maximum (terminal) penetration depth
    '''
    dep = out[["id", "distance", "depth"]].rename(
        columns={"depth": "penetration depth"})

    full = df_ic.merge(dep, on="id", how="left")

    full["status"] = np.where(full["penetration depth"].notna(), 1, 0)
    full.loc[full["penetration depth"] == out["depth"].max(), "status"] = 2

    return full

def deposition_depth_pdf(
      df,
      diam = None,
      bins = 20,
      rtol = 1e-3,
      depth_col = "depth",
      diam_col = "diam",
    ):
    '''Compute the probability density of deposition depth.

    Parameters
    ----------
    df : DataFrame
        Must contain `depth_col` and (if `diam` is given) `diam_col`.
    diam : float or None
        If given, only particles whose diameter matches `diam` (within a
        relative tolerance `rtol`) are used. If None, all diameters are used.
    bins : int or sequence
        Number of bins, or explicit bin edges, passed to np.histogram.
    rtol : float
        Relative tolerance for matching `diam`.

    Returns
    -------
    centers : ndarray
        Bin centers of the depth axis.
    density : ndarray
        Probability density at each bin center (integrates to 1 over depth).
    '''
    depth = df[depth_col].to_numpy()

    if diam is not None:
        mask  = np.isclose(df[diam_col].to_numpy(), diam, rtol=rtol)
        depth = depth[mask]

    depth = depth[np.isfinite(depth)]
    if depth.size == 0:
        raise ValueError("no particles matched the requested diameter")

    density, edges = np.histogram(depth, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density

partICfileName = 'partData_particle_0.Netcdf'
partLogfileName = 'particle.log'

df_ic = read_particle_netcdf(partICfileName)
df_log = read_particle_log()

df = add_IC(df_ic, df_log)

df_to_vtp(df)

center = read_centerlines()
out = add_addtributes(df, center)

# full set of all initialized particles, with deposition status + attributes
full = combine_particles(df_ic, out)



fig, ax = plt.subplots()
# ax.set_xscale("log")

out.plot.scatter(x="distance", y="depth", ax=ax, label="all", color="red", alpha = 0.5)
# df.plot.scatter(x="diam", y="ic projected velocity", ax=ax, label="deposited", color="blue", alpha = 0.5)
fig.savefig("diam_vs_distance.pdf", dpi=300, bbox_inches="tight")


# PDF of deposition depth vs diameter as a contour plot
# exclude particles at the terminal (maximum) depth
out_pdf = out[out["depth"] < 250.0]

# common depth bins so every diameter shares the same x-axis
diams = np.array(sorted(out_pdf["diam"].unique()))
edges = np.linspace(out_pdf["depth"].min(), out_pdf["depth"].max(), 21)

pdf_grid = np.empty((len(diams), len(edges) - 1))
for i, diam in enumerate(diams):
    depth_centers, density = deposition_depth_pdf(out_pdf, diam=diam, bins=edges)
    pdf_grid[i] = density

D, DIA = np.meshgrid(depth_centers, diams)   # x = depth, y = diameter

fig, ax = plt.subplots()
cs = ax.contourf(D, DIA, pdf_grid, levels=10, cmap="viridis")
fig.colorbar(cs, ax=ax, label="probability density")
ax.set_xlabel("deposition depth")
ax.set_ylabel("diameter")
ax.set_yscale("log")
ax.set_xlim([0,100])
ax.set_ylim([1e-5, 2.5e-2])
fig.savefig("depth_pdf_by_diam.pdf", dpi=300, bbox_inches="tight")


# PDF of deposition depth over all particles (all diameters pooled)
fig, ax = plt.subplots()
centers, density = deposition_depth_pdf(out_pdf)
width = (centers[1] - centers[0]) if len(centers) > 1 else 1.0
ax.bar(centers, density, width=width, color="black", alpha=0.6, label="all")
ax.set_xlabel("deposition depth")
ax.set_ylabel("probability density")
ax.legend()
fig.savefig("depth_pdf_all.pdf", dpi=300, bbox_inches="tight")


# Diameter PDF for three particle populations, as a grouped bar chart
diams = np.array(sorted(full["d"].unique()))
x = np.arange(len(diams))

def diam_pmf(sub):
    # fraction of ALL particles in each diameter bin of this subset
    return (sub["d"].value_counts()
            .reindex(diams, fill_value=0).to_numpy() / len(full))

groups = {
    "inhaled":    full[full["status"] == 2],
    "deposited + inhaled": full[full["status"] >= 1],
    "all":      full,
}

width = 0.8 / len(groups)
fig, ax = plt.subplots(figsize=(6,3))
colors= ["#ff7f0e","#1f77b4", "#acacac"]
for i, (label, sub) in enumerate(groups.items()):
    offset = (i - (len(groups) - 1) / 2) * width
    ax.bar(x + offset, diam_pmf(sub), width=width, label=label, color=colors[i])
ax.set_xlabel("$d_{\mathrm{p}} ~[mm]$")
ax.set_ylabel("$N/N_{\mathrm{p}}$")
ax.set_xticks(x)
ax.set_xticklabels([f"{d:.3g}" for d in diams], rotation=45)
ax.legend(loc="upper right")
fig.savefig("diam_pdf_by_status.pdf", dpi=300, bbox_inches="tight")


# Deposition fraction vs IC distance, pooled over all diameters
ic     = full["distance to origin"].to_numpy()
status = full["status"].to_numpy()

edges   = np.linspace(ic.min(), ic.max(), 21)
centers = 0.5 * (edges[:-1] + edges[1:])
which   = np.clip(np.digitize(ic, edges) - 1, 0, len(centers) - 1)

total = np.bincount(which,               minlength=len(centers))
dep   = np.bincount(which[status == 1],  minlength=len(centers))
term  = np.bincount(which[status == 2],  minlength=len(centers))

with np.errstate(invalid="ignore", divide="ignore"):
    frac_dep  = np.where(total > 0, dep  / total, 0.0)
    frac_term = np.where(total > 0, term / total, 0.0)

bw = edges[1] - edges[0]
fig, ax = plt.subplots(figsize=(6,3))

# background: total particle count per bin, to show the extent of IC distance
ax2 = ax.twinx()
ax2.bar(centers, total, width=bw, color="gray", alpha=0.5, zorder=0,
        label="all particles (count)")
ax2.set_ylabel("particle count")

# deposition fractions on top of the background
ax.set_zorder(ax2.get_zorder() + 1)
ax.patch.set_visible(False)
ax.bar(centers - 0.2 * bw, frac_dep,  width=0.4 * bw, label="deposited + inhaled ")
ax.bar(centers + 0.2 * bw, frac_term, width=0.4 * bw, label="inhaled")
ax.set_xlabel("$d_{\mathrm{IC}} ~[mm]$")
ax.set_ylabel("fraction of particles in bin")

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2)
fig.savefig("deposition_fraction_vs_ic_distance.pdf", dpi=300, bbox_inches="tight")
