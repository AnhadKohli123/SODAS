import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6})
KEY = ["model_res", "seed", "imgsz", "mode", "tile", "overlap", "full"]


def load(path):
    d = pd.read_csv(path)
    d.columns = d.columns.str.strip()
    for c in ["tile", "overlap", "full"]:
        if c not in d:
            d[c] = -1
    return d


base = load("results/val/baselines.csv")
grid = load("results/val/grid.csv")
df = pd.concat([base, grid], ignore_index=True)
df = df.drop_duplicates(subset=KEY, keep="last")
df["tile"] = df["tile"].fillna(-1)
os.makedirs("figures", exist_ok=True)


def panel_data(res, seeds):
    d = df[(df.model_res == res) & (df.seed.isin(seeds))]
    d = d.groupby(["imgsz", "mode", "tile", "overlap", "full"], as_index=False)[["AP_S", "lat_median_ms"]].mean()
    std = d[d["mode"] == "standard"].sort_values("imgsz")
    t = d[(d["mode"] == "tiled") & (d["full"].astype(str).isin(["False", "0", "0.0"]))].sort_values("tile")
    tf = d[(d["mode"] == "tiled") & (d["full"].astype(str).isin(["True", "1", "1.0"]))].sort_values("tile")
    return std, t, tf


fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.55))
for a, res, seeds, title in ((ax[0], 640, [0], "(a) 640-pixel model"), (ax[1], 960, [0, 1], "(b) 960-pixel model (mean of 2 seeds)")):
    std, t, tf = panel_data(res, seeds)
    a.plot(std.lat_median_ms, std.AP_S, "-o", c="#1f4e9c", ms=4, lw=1, label="standard (resolution)")
    for _, r in std.iterrows():
        a.annotate(str(int(r.imgsz)), (r.lat_median_ms, r.AP_S), textcoords="offset points", xytext=(0, -10), ha="center", fontsize=6.5, color="#1f4e9c")
    a.scatter(t.lat_median_ms, t.AP_S, marker="s", c="#d98a1c", s=22, label="tiled")
    a.scatter(tf.lat_median_ms, tf.AP_S, marker="^", c="#b2261f", s=26, label="tiled + full image")
    for _, r in t.iterrows():
        lab = str(int(r.tile)) + (" o.4" if r.overlap > 0.3 else "")
        a.annotate(lab, (r.lat_median_ms, r.AP_S), textcoords="offset points", xytext=(0, -10), ha="center", fontsize=6.5, color="#8a5a10")
    for _, r in tf.iterrows():
        a.annotate(str(int(r.tile)), (r.lat_median_ms, r.AP_S), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6.5, color="#8a1c16")
    a.set_xscale("log")
    a.set_xticks([30, 50, 100, 200, 400, 600])
    a.set_xticklabels(["30", "50", "100", "200", "400", "600"])
    a.minorticks_off()
    a.set_xlim(25, 800)
    a.set_xlabel("median latency per image (ms, log scale)")
    a.set_title(title, fontsize=8)
    a.grid(alpha=0.25, lw=0.4)
ax[0].set_ylabel(r"AP$_S$ (val)")
ax[0].legend(loc="lower right", fontsize=6.5, frameon=False)
plt.tight_layout(pad=0.4)
plt.savefig("figures/fig_frontier.pdf")
plt.savefig("figures/fig_frontier.png", dpi=140)
plt.close()

d = df[(df.model_res == 640) & (df.seed == 0)]
rows = []
for s in (640, 960, 1280):
    r = d[(d["mode"] == "standard") & (d.imgsz == s)].iloc[0]
    rows.append((str(s) + "\nstd", r.R50_tiny, r.R50_S, "#999999", "#c8c8c8"))
for tl in (320, 512, 640):
    r = d[(d["mode"] == "tiled") & (d.tile == tl) & (d.overlap == 0.2) & (d["full"].astype(str).isin(["True", "1", "1.0"]))].iloc[0]
    rows.append((str(tl) + "\n+full", r.R50_tiny, r.R50_S, "#1b9e77", "#84cbb7"))
fig, a = plt.subplots(figsize=(3.4, 2.3))
for i, (lab, tiny, small, c1, c2) in enumerate(rows):
    a.bar(i - 0.2, tiny, 0.4, color=c1)
    a.bar(i + 0.2, small, 0.4, color=c2)
    a.text(i - 0.2, tiny + 1, "%d" % round(tiny), ha="center", fontsize=6.5)
    a.text(i + 0.2, small + 1, "%d" % round(small), ha="center", fontsize=6.5)
a.set_xticks(range(len(rows)))
a.set_xticklabels([r[0] for r in rows], fontsize=6.5)
a.set_ylabel("recall at IoU 0.5 (%)")
a.set_ylim(0, 80)
a.spines[["top", "right"]].set_visible(False)
a.text(0.02, 0.97, "dark: tiny, light: small\ngrey: standard, green: tiled", transform=a.transAxes, va="top", fontsize=6.5)
plt.tight_layout(pad=0.4)
plt.savefig("figures/fig_recall.pdf")
plt.savefig("figures/fig_recall.png", dpi=140)
print(df[df.model_res == 960].groupby(["imgsz", "mode", "tile", "full"])[["AP_S", "lat_median_ms"]].mean().round(1))
