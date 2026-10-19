import argparse
import pandas as pd
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--res", type=int, default=640)
p.add_argument("--full", type=int, default=0)
a = p.parse_args()

df = pd.read_csv("results/val/grid.csv")
df = df[(df["model_res"] == a.res) & (df["full"] == bool(a.full)) & (df["merge"] == "GREEDYNMM")]
ap = df.pivot_table(index="tile", columns="overlap", values="AP_S", aggfunc="mean")
lat = df.pivot_table(index="tile", columns="overlap", values="lat_median_ms", aggfunc="mean")

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(ap.values, cmap="viridis")
ax.set_xticks(range(len(ap.columns)), ap.columns)
ax.set_yticks(range(len(ap.index)), ap.index)
for i in range(ap.shape[0]):
    for j in range(ap.shape[1]):
        ax.text(j, i, f"{ap.values[i, j]:.1f}\n{lat.values[i, j]:.0f}ms", ha="center", va="center", color="w", fontsize=8)
ax.set_xlabel("overlap")
ax.set_ylabel("tile size")
fig.colorbar(im, label="AP_S")
fig.savefig(f"figures/heatmap_{a.res}_full{a.full}.png", dpi=200, bbox_inches="tight")
