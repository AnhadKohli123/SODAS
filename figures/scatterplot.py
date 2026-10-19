import pandas as pd
import matplotlib.pyplot as plt

g = pd.read_csv("results/val/grid.csv")
g = g[g["merge"] == "GREEDYNMM"]
g = g.groupby(["model_res", "tile", "overlap", "full"]).agg(lat=("lat_median_ms", "mean"), ap=("AP_S", "mean")).reset_index()
b = pd.read_csv("results/val/baselines.csv").groupby(["model_res", "imgsz"]).agg(lat=("lat_median_ms", "mean"), ap=("AP_S", "mean")).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, res in zip(axes, sorted(g["model_res"].unique())):
    d = g[g["model_res"] == res]
    sc = ax.scatter(d["lat"], d["ap"], c=d["tile"], cmap="viridis", marker="o")
    bb = b[b["model_res"] == res]
    ax.scatter(bb["lat"], bb["ap"], c="red", marker="s", label="standard")
    for _, r in bb.iterrows():
        ax.annotate(int(r["imgsz"]), (r["lat"], r["ap"]), textcoords="offset points", xytext=(4, 4))
    ax.set_title(f"trained at {res}")
    ax.set_xlabel("median latency (ms/image)")
    ax.legend()
axes[0].set_ylabel("AP_S")
fig.colorbar(sc, ax=axes, label="tile size")
fig.savefig("figures/scatter_latency_aps.png", dpi=200, bbox_inches="tight")
