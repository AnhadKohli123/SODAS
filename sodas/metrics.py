
import contextlib
import io
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

RANGES = {
    "all": [0, 1e10],
    "tiny": [0, 256],
    "mid_small": [256, 1024],
    "S": [0, 1024],
    "M": [1024, 9216],
    "L": [9216, 1e10],
}


def _mean(a):
    a = a[a > -1]
    return float(a.mean()) * 100 if a.size else 0.0


def evaluate(gt_path, dets, max_dets=500):
    if not dets:
        return {k: 0.0 for k in ["AP", "AP50"] + [f"AP_{n}" for n in RANGES if n != "all"] + [f"R50_{n}" for n in RANGES]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_path)
        dt = gt.loadRes(dets)
        E = COCOeval(gt, dt, "bbox")
        E.params.maxDets = [1, 10, max_dets]
        E.params.areaRng = list(RANGES.values())
        E.params.areaRngLbl = list(RANGES.keys())
        E.evaluate()
        E.accumulate()
    p, r = E.eval["precision"], E.eval["recall"]
    out = {"AP": _mean(p[:, :, :, 0, -1]), "AP50": _mean(p[0, :, :, 0, -1])}
    for a, n in enumerate(RANGES):
        if n != "all":
            out[f"AP_{n}"] = _mean(p[:, :, :, a, -1])
        out[f"R50_{n}"] = _mean(r[0, :, a, -1])
    return out


def efficiency(ap_tiled, ap_std, t_tiled, t_std):
    d = t_tiled - t_std
    return (ap_tiled - ap_std) / d if d > 0 else float("nan")
