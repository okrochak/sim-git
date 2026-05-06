#!/usr/bin/env python3
"""
Parse a particle deposition log file and convert it to VTK format for ParaView.

Input file format (whitespace/tab-separated):
    col 1:  status flag (e.g., REMOVED)
    col 2:  timestep
    col 3:  partId
    col 4:  diameter
    ...
    col 10: deposition status
    col 11: x deposition location
    col 12: y deposition location
    col 13: z deposition location
    ...

Output:
    A .vtp (PolyData) file containing one point per deposited particle,
    with timestep, partId, diameter, and deposition_status as point data arrays.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_deposition_log(path: Path) -> pd.DataFrame:
    """Read the log file and return a DataFrame with the columns we care about."""
    # Read all columns as strings first so that 'REMOVED' and '-nan' don't break parsing,
    # then convert numeric columns explicitly.
    df = pd.read_csv(
        path,
        sep=r"\s+",          # any whitespace (handles tabs and multiple spaces)
        header=None,
        engine="python",
        comment="#",         # ignore comment lines if any
        dtype=str,
        skip_blank_lines=True,
    )

    # Column indices in the file are 1-based per the user description; pandas is 0-based.
    # We need: col 2 -> idx 1, col 3 -> idx 2, col 10 -> idx 9, col 11 -> idx 10,
    #         col 12 -> idx 11, col 13 -> idx 12.
    needed = {
        "timestep": 1,
        "partId": 2,
        "diameter": 3,
        "deposition_status": 9,
        "x": 10,
        "y": 11,
        "z": 12,
    }

    missing_cols = [name for name, idx in needed.items() if idx >= df.shape[1]]
    if missing_cols:
        raise ValueError(
            f"Input file does not have enough columns. "
            f"Got {df.shape[1]} columns, missing: {missing_cols}"
        )

    out = pd.DataFrame({name: df.iloc[:, idx] for name, idx in needed.items()})

    # Convert types. '-nan' / 'nan' strings will become NaN automatically.
    out["timestep"] = pd.to_numeric(out["timestep"], errors="coerce").astype("Int64")
    out["partId"] = pd.to_numeric(out["partId"], errors="coerce").astype("Int64")
    out["diameter"] = pd.to_numeric(out["diameter"], errors="coerce")
    out["deposition_status"] = pd.to_numeric(out["deposition_status"], errors="coerce")
    for c in ("x", "y", "z"):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Drop rows where the deposition coordinates are NaN — those particles weren't deposited
    # at a valid location and would produce garbage points in ParaView.
    before = len(out)
    out = out.dropna(subset=["x", "y", "z"]).reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        print(f"[info] Dropped {dropped} rows with NaN coordinates "
              f"(likely non-deposited particles).")

    return out


def write_vtp(df: pd.DataFrame, out_path: Path) -> None:
    """Write the DataFrame to a VTK PolyData (.vtp) file using pyvista."""
    try:
        import pyvista as pv
    except ImportError as e:
        raise SystemExit(
            "pyvista is required. Install with: pip install pyvista"
        ) from e

    points = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    cloud = pv.PolyData(points)

    # Attach scalar arrays. pyvista will store these as point_data.
    cloud["timestep"] = df["timestep"].to_numpy(dtype=np.int64)
    cloud["partId"] = df["partId"].to_numpy(dtype=np.int64)
    cloud["diameter"] = df["diameter"].to_numpy(dtype=np.float64)
    cloud["deposition_status"] = df["deposition_status"].to_numpy(dtype=np.float64)

    cloud.save(str(out_path))
    print(f"[info] Wrote {len(df)} points to {out_path}")


def write_legacy_vtk(df: pd.DataFrame, out_path: Path) -> None:
    """Fallback writer that produces a legacy ASCII .vtk POLYDATA file with no dependencies."""
    n = len(df)
    with open(out_path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Particle deposition locations\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n")
        for x, y, z in df[["x", "y", "z"]].to_numpy():
            f.write(f"{x} {y} {z}\n")

        # Each point as its own VTK_VERTEX cell so they render in ParaView.
        f.write(f"VERTICES {n} {2 * n}\n")
        for i in range(n):
            f.write(f"1 {i}\n")

        f.write(f"POINT_DATA {n}\n")

        f.write("SCALARS timestep int 1\nLOOKUP_TABLE default\n")
        for v in df["timestep"].to_numpy():
            f.write(f"{int(v)}\n")

        f.write("SCALARS partId long 1\nLOOKUP_TABLE default\n")
        for v in df["partId"].to_numpy():
            f.write(f"{int(v)}\n")

        f.write("SCALARS diameter float 1\nLOOKUP_TABLE default\n")
        for v in df["diameter"].to_numpy():
            f.write(f"{v}\n")

        f.write("SCALARS deposition_status float 1\nLOOKUP_TABLE default\n")
        for v in df["deposition_status"].to_numpy():
            f.write(f"{v}\n")

    print(f"[info] Wrote {n} points to {out_path} (legacy ASCII VTK)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="Path to the deposition log file")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output path (.vtp or .vtk). Defaults to <input>.vtp")
    p.add_argument("--legacy", action="store_true",
                   help="Write legacy ASCII .vtk instead of .vtp (no pyvista needed)")
    args = p.parse_args()

    if not args.input.exists():
        sys.exit(f"Input file not found: {args.input}")

    df = parse_deposition_log(args.input)
    if df.empty:
        sys.exit("No valid deposition rows found.")

    if args.output is None:
        suffix = ".vtk" if args.legacy else ".vtp"
        args.output = args.input.with_suffix(suffix)

    if args.legacy or args.output.suffix.lower() == ".vtk":
        write_legacy_vtk(df, args.output)
    else:
        write_vtp(df, args.output)


if __name__ == "__main__":
    main()
