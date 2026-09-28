import matplotlib.pyplot as plt
import pandas as pd

# probe coordinates, used as legend labels
points = pd.read_csv("point.dat", sep=r"\s+", header=None, names=["x", "y", "z"])
labels = [f"({r.x}, {r.y}, {r.z})" for _, r in points.iterrows()]

df = pd.read_csv("probes.dat", sep=r"\s+", header=0)
probe_ids = sorted({col.split("_p")[1] for col in df.columns if "_p" in col}, key=int)

fig, ax = plt.subplots()
for i, pid in enumerate(probe_ids):
    vel = (df[f"u_p{pid}"]**2 + df[f"v_p{pid}"]**2 + df[f"w_p{pid}"]**2)**0.5
    ax.plot(df["time_step"], vel, label=labels[i] if i < len(labels) else f"Probe {pid}")

ax.set(xlabel="time step", ylabel="Velocity magnitude")
ax.legend()
fig.savefig("probes_vel_vs_time.png", dpi=200, bbox_inches="tight")
