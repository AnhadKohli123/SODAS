import csv
import numpy as np


def read(path):
    rows = list(csv.reader(open(path)))
    head = rows[0] + ["tile", "overlap", "full", "merge"]
    out = []
    for r in rows[1:]:
        d = dict(zip(head, r))
        for k in ("model_res", "seed", "imgsz", "AP_S", "lat_median_ms", "slices_mean", "tile", "overlap"):
            if k in d and d[k] != "":
                d[k] = float(d[k])
        d["full"] = str(d.get("full", "")) in ("True", "1", "1.0")
        out.append(d)
    return out


base = read("results/val/baselines.csv")
grid = read("results/val/grid.csv")


def std(res, seed, size):
    return next(d for d in base if d["model_res"] == res and d["seed"] == seed and d["imgsz"] == size)


print("Efficiency eta = (AP_S tiled - AP_S standard) / (latency tiled - latency standard), AP_S points per ms")
print("model seed  setting                  eta vs standard at the model input size")
for res in (640, 960):
    for seed in (0, 1):
        s0 = std(res, seed, res)
        s16 = std(res, seed, 1600)
        for d in sorted((g for g in grid if g["model_res"] == res and g["seed"] == seed), key=lambda g: (g["tile"], g["overlap"], g["full"])):
            e0 = (d["AP_S"] - s0["AP_S"]) / (d["lat_median_ms"] - s0["lat_median_ms"])
            print("%d    %d     tile %d o%.1f %-5s  %6.2f  (%.1f ms)" % (res, seed, d["tile"], d["overlap"], "+full" if d["full"] else "", e0, d["lat_median_ms"]))
        eta16 = (s16["AP_S"] - s0["AP_S"]) / (s16["lat_median_ms"] - s0["lat_median_ms"])
        print("%d    %d     standard 1600 vs standard %d: eta = %.2f" % (res, seed, res, eta16))


def fit(rows):
    x = np.array([r["slices_mean"] for r in rows])
    y = np.array([r["lat_median_ms"] for r in rows])
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return a, b, r2, np.abs(y - pred).max(), len(rows)


print("\nCost model T = a + b * n (n = slices per image), least squares")
groups = [
    ("640 model seed 0, tiles 512-800", [g for g in grid if g["model_res"] == 640 and g["seed"] == 0 and g["tile"] >= 512]),
    ("640 model seed 0, tile 320", [g for g in grid if g["model_res"] == 640 and g["seed"] == 0 and g["tile"] == 320]),
    ("640 model seed 1, tiles 512-640", [g for g in grid if g["model_res"] == 640 and g["seed"] == 1]),
    ("960 model seed 0", [g for g in grid if g["model_res"] == 960 and g["seed"] == 0]),
    ("960 model seed 1", [g for g in grid if g["model_res"] == 960 and g["seed"] == 1]),
]
for name, rows in groups:
    a, b, r2, err, n = fit(rows)
    print("%-34s a=%.1f ms  b=%.1f ms  R2=%.2f  max error=%.0f ms  (%d configs)" % (name, a, b, r2, err, n))
