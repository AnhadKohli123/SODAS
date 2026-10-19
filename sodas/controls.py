import numpy as np


def identity_cfg():
    return {"mode": "tiled", "tile": 10000, "overlap": 0.0, "full": False, "merge": "GREEDYNMM"}


def corrupt_offsets(dets, tile, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for d in dets:
        x, y, w, h = d["bbox"]
        x += rng.choice([-1, 1]) * tile * rng.integers(1, 3)
        y += rng.choice([-1, 1]) * tile * rng.integers(1, 3)
        out.append({**d, "bbox": [float(x), float(y), w, h]})
    return out
