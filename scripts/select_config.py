import json
import numpy as np
import pandas as pd

df = pd.read_csv("results/val/grid.csv")
df = df[df["merge"] == "GREEDYNMM"]
key = ["model_res", "tile", "overlap", "full", "merge"]
g = df.groupby(key).agg(ap_s=("AP_S", "mean"), sd=("AP_S", "std"), n=("seed", "count"), lat=("lat_median_ms", "mean")).reset_index()

frozen = {}
for res, d in g.groupby("model_res"):
    best = d.loc[d["ap_s"].idxmax()]
    se = 0.0 if np.isnan(best["sd"]) else best["sd"] / np.sqrt(best["n"])
    pick = d[d["ap_s"] >= best["ap_s"] - se].sort_values("lat").iloc[0]
    frozen[str(int(res))] = {"tile": int(pick["tile"]), "overlap": float(pick["overlap"]), "full": bool(pick["full"]), "merge": pick["merge"]}
    print(res, frozen[str(int(res))], round(pick["ap_s"], 2), round(pick["lat"], 1))

json.dump(frozen, open("results/frozen.json", "w"), indent=2)
