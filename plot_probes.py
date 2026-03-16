import pandas as pd      
import matplotlib.pyplot as plt
import numpy as np                                                            

# Read probe coordinates for legend labels                                    
points = pd.read_csv("point.dat", sep=r"\s+", header=None, names=["x", "y",
"z"])
labels = [f"({row.x}, {row.y}, {row.z})" for _, row in points.iterrows()]

# Read probe data and detect number of probes from columns
df = pd.read_csv("probes.dat", sep=r"\s+", header=0)

probe_ids = sorted(set(
    col.split("_p")[1] for col in df.columns if "_p" in col
), key=int)

for i, pid in enumerate(probe_ids):
    u, v, w = df[f"u_p{pid}"], df[f"v_p{pid}"], df[f"w_p{pid}"]
    vel = (u**2 + v**2 + w**2)**0.5
    label = labels[i] if i < len(labels) else f"Probe {pid}"
    plt.plot(df["time_step"], vel, label=label)

plt.xlabel("time")
plt.ylabel("Velocity magnitude")
plt.legend()
plt.savefig("probes_vel_vs_time.png", dpi=200)