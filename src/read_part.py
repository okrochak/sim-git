import argparse

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree

NOSTRIL = np.array([-3.5, 1108.0, 1595.0])


def spatial_filter(df):
    '''Drop particles removed outside the airway, upstream of the nose.'''
    y, z = df["y"], df["z"]
    remove = (
        (y < 1095) |
        ((y >= 1095) & (y < 1125) & (z < 1585)) |
        ((y >= 1125) & (y < 1185) & (z < 1550))
    )
    return df[~remove].reset_index(drop=True)


def read_particle_log(fileName = "particle.log"):
    '''Removed particles from the solver log. Column 9 is 1 if deposited.'''
    array = np.loadtxt(fileName, usecols=[1, 2, 3, 4, 5, 6, 9])
    df = pd.DataFrame(array, columns=["t", "id", "diam", "x", "y", "z", "dep_status"])
    df[["t", "dep_status"]] = df[["t", "dep_status"]].astype("int")
    # the solver packs extra bits above the 32-bit id
    df["id"] = (df["id"].astype("int64") & 0xFFFFFFFF).astype("int32")

    # a particle can appear more than once; keep only its latest appearance
    df = df.drop_duplicates("id", keep="last").reset_index(drop=True)
    return spatial_filter(df)


def read_particle_netcdf(fileName):
    '''Initial conditions of every particle.'''
    data = nc.Dataset(fileName, mode="r").variables
    n = data["partPos"].shape[0] // 3
    pos = np.asarray(data["partPos"]).reshape(n, 3)
    vel = np.asarray(data["partVel"]).reshape(n, 3)
    df = pd.DataFrame({
        "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
        "u": vel[:, 0], "v": vel[:, 1], "w": vel[:, 2],
        "d":  np.asarray(data["partDia"]).ravel(),
        "id": np.asarray(data["globalPartID"]).ravel().astype("int32"),
    })

    rel = pos - NOSTRIL
    df["velocity magnitude"]  = np.linalg.norm(vel, axis=1)
    df["distance to origin"]  = np.linalg.norm(rel, axis=1)
    df["projected velocity"]  = -(vel * rel).sum(axis=1) / df["distance to origin"]
    return df


def add_IC(df_ic, df_log):
    '''Attach each logged particle's initial condition.'''
    ic = df_ic.drop(columns=["d"]).rename(columns={
        "x": "ic_x", "y": "ic_y", "z": "ic_z",
        "u": "ic_u", "v": "ic_v", "w": "ic_w",
        "velocity magnitude": "ic velocity",
        "distance to origin": "ic distance to origin",
        "projected velocity": "ic projected velocity",
    })
    df = df_log.merge(ic, on="id", how="left", validate="one_to_one")
    if df["ic_x"].isna().any():
        raise ValueError("some logged particles are missing from the IC file")
    return df


def write_vtp(df, fileName = "particles.vtp"):
    '''Write the particles as a point cloud, with whichever arrays are present.'''
    cloud = pv.PolyData(df[["x", "y", "z"]].to_numpy(float))
    for name, col, dtype in (("timestep",         "t",          "int64"),
                             ("globalParticleID", "id",         "int32"),
                             ("diameter",         "diam",       float),
                             ("density",          "density",    float),
                             ("deposited",        "dep_status", "int8")):
        if col in df:
            cloud[name] = df[col].to_numpy(dtype)
    for name, cols in (("IC position", ["ic_x", "ic_y", "ic_z"]),
                       ("IC velocity", ["ic_u", "ic_v", "ic_w"])):
        if set(cols) <= set(df.columns):
            cloud[name] = df[cols].to_numpy(float)
    cloud.save(fileName)


def read_centerlines(leftName = "left_centerline.dat",
                     rightName = "right_centerline.dat"):
    '''Mean centerline, reversed so that depth 0 is at the nose.'''
    args = dict(sep=r"\s+", header=0, usecols=[0, 1, 2])
    left  = pd.read_csv(leftName,  **args).iloc[::-1].reset_index(drop=True)
    right = pd.read_csv(rightName, **args).iloc[::-1].reset_index(drop=True)
    return mean_centerline(left, right)


def mean_centerline(left, right):
    '''Resample both lines onto a common arc length, average, accumulate depth.'''
    def resample(df, s):
        coords = df[["X", "Y", "Z"]].to_numpy()
        seg = np.concatenate(([0.0],
                np.linalg.norm(np.diff(coords, axis=0), axis=1).cumsum()))
        return np.column_stack([np.interp(s, seg / seg[-1], coords[:, k])
                                for k in range(3)])

    s = np.linspace(0.0, 1.0, max(len(left), len(right)))
    mean = 0.5 * (resample(left, s) + resample(right, s))
    center = pd.DataFrame(mean, columns=["X", "Y", "Z"])
    step = np.linalg.norm(np.diff(mean, axis=0), axis=1)
    center["depth"] = np.concatenate(([0.0], step.cumsum()))
    return center


def add_attributes(df, center):
    '''Nearest-centerline distance and penetration depth of every particle.'''
    tree = cKDTree(center[["X", "Y", "Z"]].to_numpy())
    dist, idx = tree.query(df[["x", "y", "z"]].to_numpy())
    return df.assign(distance=dist, depth=center["depth"].to_numpy()[idx])


def combine_particles(df_ic, out):
    '''All initialized particles, flagged by fate.

    status 0 : never removed inside the valid region -> depth/distance are NaN
    status 1 : deposited on the airway wall, per the solver
    status 2 : inhaled -- removed in-region without depositing, so it left
               through the outlet at the end of the centerline

    Status 2 assumes the outlet is the only way out of the valid region. The
    printed depth is the check: if it drops well below the terminal depth,
    particles are leaving some other way and status 2 no longer holds.
    '''
    dep = out[["id", "distance", "depth", "dep_status"]].rename(
        columns={"depth": "penetration depth"})
    full = df_ic.merge(dep, on="id", how="left", validate="one_to_one")

    full["status"] = np.where(full["dep_status"] == 1, 1, 2)
    full.loc[full["penetration depth"].isna(), "status"] = 0

    inhaled = full.loc[full["status"] == 2, "penetration depth"]
    print("inhaled %d particles, shallowest at depth %.1f of %.1f"
          % (len(inhaled), inhaled.min(), out["depth"].max()))
    return full


def deposition_depth_pdf(df, diam = None, bins = 20, rtol = 1e-3):
    '''Probability density of deposition depth, optionally for one diameter.'''
    depth = df["depth"].to_numpy()
    if diam is not None:
        depth = depth[np.isclose(df["diam"].to_numpy(), diam, rtol=rtol)]

    depth = depth[np.isfinite(depth)]
    if depth.size == 0:
        raise ValueError("no particles matched the requested diameter")

    density, edges = np.histogram(depth, bins=bins, density=True)
    return 0.5 * (edges[:-1] + edges[1:]), density


def save(fig, fileName):
    fig.savefig(fileName, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_depth_vs_distance(out, fileName):
    fig, ax = plt.subplots()
    out.plot.scatter(x="distance", y="depth", ax=ax, color="red", alpha=0.5)
    save(fig, fileName)


def plot_depth_pdf_by_diam(dep, fileName, min_count = 50):
    '''Depth PDF per diameter as a contour. Sparse diameters are dropped.'''
    counts = dep["diam"].value_counts()
    diams = np.array(sorted(counts[counts >= min_count].index))
    edges = np.linspace(dep["depth"].min(), dep["depth"].max(), 21)
    grid = np.array([deposition_depth_pdf(dep, diam=d, bins=edges)[1] for d in diams])
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots()
    cs = ax.contourf(*np.meshgrid(centers, diams), grid, levels=10, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="probability density")
    ax.set(xlabel="deposition depth", ylabel="diameter", yscale="log", xlim=(0, 100))
    save(fig, fileName)


def plot_depth_pdf_all(dep, fileName):
    centers, density = deposition_depth_pdf(dep)
    width = centers[1] - centers[0] if len(centers) > 1 else 1.0

    fig, ax = plt.subplots()
    ax.bar(centers, density, width=width, color="black", alpha=0.6)
    ax.set(xlabel="deposition depth", ylabel="probability density")
    save(fig, fileName)


def plot_diam_by_status(full, fileName):
    '''Fate of the particles of each diameter, as a percentage of that diameter.'''
    diams = np.array(sorted(full["d"].unique()))
    totals = full["d"].value_counts().reindex(diams).to_numpy()
    groups = {
        "inhaled":             full["status"] == 2,
        "deposited + inhaled": full["status"].isin([1, 2]),
    }

    x = np.arange(len(diams))
    width = 0.8 / len(groups)
    colors = ["#ff7f0e", "#1f77b4"]

    fig, ax = plt.subplots(figsize=(6, 3))
    for i, (label, mask) in enumerate(groups.items()):
        share = (full[mask]["d"].value_counts()
                 .reindex(diams, fill_value=0).to_numpy() / totals * 100)
        ax.bar(x + (i - (len(groups) - 1) / 2) * width, share,
               width=width, label=label, color=colors[i], zorder=1)

    ax.set(xlabel=r"$d_{\mathrm{p}} ~[mm]$",
           ylabel="Deposition frequency [%]", xticks=x)
    ax.set_xticklabels([f"{d:.3g}" for d in diams], rotation=45)
    ax.legend(loc="upper right")
    save(fig, fileName)


def plot_dep_fraction_vs_ic(full, fileName, bins = 20):
    '''Fate against how far the particle started from the nose.'''
    ic = full["distance to origin"].to_numpy()
    status = full["status"].to_numpy()

    edges = np.linspace(ic.min(), ic.max(), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    which = np.clip(np.digitize(ic, edges) - 1, 0, len(centers) - 1)

    total = np.bincount(which,              minlength=len(centers))
    dep   = np.bincount(which[status == 1], minlength=len(centers))
    term  = np.bincount(which[status == 2], minlength=len(centers))
    frac = lambda a: np.where(total > 0, a / np.maximum(total, 1), 0.0)

    bw = edges[1] - edges[0]
    fig, ax = plt.subplots(figsize=(6, 3))

    # total count behind the fractions, to show where the particles started
    ax2 = ax.twinx()
    ax2.bar(centers, total, width=bw, color="gray", alpha=0.5, zorder=0,
            label="all particles (count)")
    ax2.set_ylabel("particle count")

    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.bar(centers - 0.2 * bw, frac(dep),  width=0.4 * bw, label="deposited")
    ax.bar(centers + 0.2 * bw, frac(term), width=0.4 * bw, label="inhaled")
    ax.set(xlabel=r"$d_{\mathrm{IC}} ~[mm]$",
           ylabel="fraction of particles in bin")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2)
    save(fig, fileName)


def main():
    ap = argparse.ArgumentParser(
        description="Particle deposition post-processing.",
        epilog="log to vtk only:  read_part.py --log particle.log --vtp-only")
    ap.add_argument("--log",   default="particle.log")
    ap.add_argument("--ic",    default="partData_particle_0.Netcdf",
                    help="NetCDF initial conditions, needed for the plots")
    ap.add_argument("--left",  default="left_centerline.dat")
    ap.add_argument("--right", default="right_centerline.dat")
    ap.add_argument("-o", "--vtp", default="particles.vtp")
    ap.add_argument("--vtp-only", action="store_true",
                    help="write the .vtp from the log alone and stop")
    args = ap.parse_args()

    df = read_particle_log(args.log)
    if args.vtp_only:
        write_vtp(df, args.vtp)
        return

    df_ic = read_particle_netcdf(args.ic)
    df = add_IC(df_ic, df)
    write_vtp(df, args.vtp)

    out = add_attributes(df, read_centerlines(args.left, args.right))
    full = combine_particles(df_ic, out)
    dep = out[out["dep_status"] == 1]

    plot_depth_vs_distance(out, "depth_vs_distance.pdf")
    plot_depth_pdf_by_diam(dep, "depth_pdf_by_diam.pdf")
    plot_depth_pdf_all(dep, "depth_pdf_all.pdf")
    plot_diam_by_status(full, "diam_pdf_by_status.pdf")
    plot_dep_fraction_vs_ic(full, "deposition_fraction_vs_ic_distance.pdf")


if __name__ == "__main__":
    main()
