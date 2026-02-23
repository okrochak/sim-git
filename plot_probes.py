import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("probes.dat", sep=r"\s+", header=0)  # first row as header
df['vel_p1'] = (df['u_p1']**2 + df['v_p1']**2 + df['w_p1']**2)**0.5
df['vel_p2'] = (df['u_p2']**2 + df['v_p2']**2 + df['w_p2']**2)**0.5
df['vel_p3'] = (df['u_p3']**2 + df['v_p3']**2 + df['w_p3']**2)**0.5
df['vel_p4'] = (df['u_p4']**2 + df['v_p4']**2 + df['w_p4']**2)**0.5

plt.plot(df["time_step"], df["vel_p1"], label="Probe 1 velocity")
plt.plot(df["time_step"], df["vel_p2"], label="Probe 2 velocity")
plt.plot(df["time_step"], df["vel_p3"], label="Probe 3 velocity")
plt.plot(df["time_step"], df["vel_p4"], label="Probe 4 velocity")

plt.xlabel("time")
plt.ylabel("Velocity magnitude")
plt.xlim([0, 860000])
# plt.ylim([0, 0.18])
plt.legend()
plt.savefig("probes_vel_vs_time.png", dpi=200)
