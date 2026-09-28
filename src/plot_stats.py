import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("steady_state_validation.csv", skipinitialspace=True)

fig, ax = plt.subplots()
x = df.iloc[:, 0]
for col in df.columns[1:]:
    ax.plot(x, df[col], marker="o", label=f"Ito et al. Q={col} l/s")

# m-AIA: diameter mm -> micron, dep_rate fraction -> %
for flow, color in (("7.5", "blue"), ("15", "orange"), ("30", "green")):
    exp = pd.read_csv(f"{flow}l/result.csv", skipinitialspace=True)
    ax.plot(exp["diameter"] * 1e3, exp["dep_rate"] * 100, marker="s",
            color=color, linestyle="--", label=f"m-AIA Q={float(flow):.1f} l/s")

ax.set(xscale="log", xlabel="$d_p$ [micron]", ylabel="DF [%]")
ax.legend()
ax.grid(True, which="both", linestyle="-", alpha=0.5)
fig.savefig("plot.pdf", bbox_inches="tight")
