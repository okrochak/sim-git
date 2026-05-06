import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("steady_state_validation.csv", skipinitialspace=True)
exp_7 = pd.read_csv("7.5l/result.csv", skipinitialspace=True)  # or however you load it
exp_15 = pd.read_csv("15l/result.csv", skipinitialspace=True)  # or however you load it
exp_30 = pd.read_csv("30l/result.csv", skipinitialspace=True)  # or however you load it

fig, ax = plt.subplots()
x = df.iloc[:, 0]
for col in df.columns[1:]:
    ax.plot(x, df[col], marker='o', label=f"Ito et al. Q={col} l/s")

# Convert experimental: diameter mm -> micron, dep_rate fraction -> %
ax.plot(exp_7["diameter"] * 1e3, exp_7["dep_rate"] * 100,
        marker='s', color='blue', linestyle='--', label="m-AIA Q=7.5 l/s")
ax.plot(exp_15["diameter"] * 1e3, exp_15["dep_rate"] * 100,
        marker='s', color='orange', linestyle='--', label="m-AIA Q=15.0 l/s")
ax.plot(exp_30["diameter"] * 1e3, exp_30["dep_rate"] * 100,
        marker='s', color='green', linestyle='--', label="m-AIA Q=30.0 l/s")

ax.set_xscale("log")
ax.set_ylabel("DF [%]")
ax.set_xlabel("$d_p$ [micron]")
ax.legend()
ax.grid(True, which='both', linestyle='-', alpha=0.5)
fig.savefig("plot.pdf")