import pandas as pd
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "particle.log"

df = pd.read_csv(filepath, sep=r"\s+", header=None, usecols=[3, 9], names=["diameter", "status"])

stats = df.groupby(["diameter", "status"]).size().unstack(fill_value=0)
stats.columns.name = None
stats = stats.rename(columns={0: "not_deposited", 1: "deposited"})
stats["total"] = stats.sum(axis=1)
stats["dep_rate"] = stats["deposited"] / stats["total"]

print(stats.to_string())
stats.to_csv('result.csv',sep=',')
