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

def df_to_vtp(
      df, 
      fileName = "particles.vtp"
    ):

    # scalars
    time        = df.iloc[:,0].to_numpy()
    partId      = df.iloc[:,1].to_numpy()
    diam        = df.iloc[:,-1].to_numpy()
    # vectors
    position    = df.iloc[:,2:5].to_numpy()
    position_ic = df.iloc[:,5:8].to_numpy() 
    velocity_ic = df.iloc[:,8:11].to_numpy() 
    
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
    df["velocity magnitude"] = df[["u","v","w"]].pow(2).sum(axis=1).pow(0.5)
    df["distance to origin"] = df[["x","y","z"]].pow(2).sum(axis=1).pow(0.5)
    df["projected velocity"] = - (
        df[["u", "v", "w"]].values *
        df[["x", "y", "z"]].values
      ).sum(axis=1) / df["distance to origin"] 
    return df 

def read_particle_log(
      fileName = 'particle.log'
    ):

    '''Read particle.log file and convert it into Pandas dataframe.'''
    array = np.loadtxt(partLogfileName, usecols=range(1, 6))
    df =  pd.DataFrame(array, columns = ["t", "id", "x", "y", "z"])
    df[["t", "id"]] = df[["t", "id"]].astype("int") 
    df["id"] = np.int32(df["id"] & 0xFFFFFFFF)
    return df


def add_IC(
      df_ic, 
      df_log
    ):
    '''Use both log-file and initial particle file dataframes and append them into one.'''
    log_ic = df_log[["id"]].merge(df_ic, on="id").reset_index(drop=True)
    log_ic.columns = ['ic_id', 'ic_x', 'ic_y', 'ic_z', 'ic_u', 'ic_v', 'ic_w', 'diam', 'ic velocity', 'ic distance to origin', 'ic projected velocity']
    
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

    return compute_distance(left), compute_distance(right)

def compute_distance(
    df 
  ):
    
    d = pd.concat([pd.Series([0]), (df[["X","Y","Z"]].diff().iloc[1:].pow(2).sum(axis=1).pow(0.5)).cumsum()])
    df["depth"] = d
    return df 

def add_addtributes(df, left, right):
  # Convert needed information into NumPy arrays
  points_df    = df[["x", "y", "z"]].to_numpy()
  points_left  = left[["X", "Y", "Z"]].to_numpy()
  points_right = right[["X", "Y", "Z"]].to_numpy()
  left_depths  = left["depth"].to_numpy()
  right_depths = right["depth"].to_numpy()

  # Build KD-trees
  tree_left  = cKDTree(points_left)
  tree_right = cKDTree(points_right)

  # Query nearest point in each tree
  dist_left, idx_left   = tree_left.query(points_df)
  dist_right, idx_right = tree_right.query(points_df)

  # Decide which side is closest (ties go to left with <=)
  use_left = dist_left <= dist_right

  nearest_dist  = np.where(use_left, dist_left,  dist_right)
  nearest_depth = np.where(use_left,
                            left_depths[idx_left],
                            right_depths[idx_right])
  side_flag = np.where(use_left, "left", "right")

  # Write back to a copy of df
  out = df.copy()
  out["distance"] = nearest_dist
  out["depth"]    = nearest_depth
  out["side"]     = side_flag

  return out

partICfileName = '7.5l/out_particles/partData_particle_0.Netcdf'
partLogfileName = '7.5l/particle.log'

df_ic = read_particle_netcdf(partICfileName)
df_log = read_particle_log()

df = add_IC(df_ic, df_log)

df_to_vtp(df)

left, right = read_centerlines()
out = add_addtributes(df,left,right)



fig, ax = plt.subplots()
ax.set_xscale("log")
df_ic.plot.scatter(x="d", y="projected velocity", ax=ax, label="all", color="red", alpha = 0.5)
df.plot.scatter(x="diam", y="ic projected velocity", ax=ax, label="deposited", color="blue", alpha = 0.5)
fig.savefig("scatter_diameter_projected_velocity.pdf", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots()
ax.set_yscale("log")
df_ic.plot.scatter(x="distance to origin", y="d", ax=ax, label="all", color="red", alpha = 0.5)
df.plot.scatter(x="ic distance to origin", y="diam", ax=ax, label="deposited", color="blue", alpha = 0.5)
fig.savefig("scatter_diameter.pdf", dpi=300, bbox_inches="tight")


fig, ax = plt.subplots()
ax.set_xscale("log")

out.plot.scatter(x="diam", y="depth", ax=ax, label="deposited", color="blue", alpha = 0.5)
fig.savefig("proj_velocity_diam.pdf", dpi=300, bbox_inches="tight")
# ax.legend()
# fig.savefig("two_scatter.pdf", bbox_inches="tight")
# ax = out.plot.scatter(x="t", y="depth")
# fig = ax.get_figure()
# fig.savefig("scatter.pdf", dpi=300, bbox_inches="tight")
print('done')
