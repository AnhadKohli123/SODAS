import argparse
import contextlib
import io
import json
import os
import tempfile
import numpy as np
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sodas.infer import tag_of, weights_path, run_config

p = argparse.ArgumentParser()
p.add_argument("--res", type=int, default=640)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--boot", type=int, default=200)
p.add_argument("--boot_seed", type=int, default=0)
p.add_argument("--thr", type=float, default=0.25)
p.add_argument("--make", action="store_true")
p.add_argument("--perclass", action="store_true")
a = p.parse_args()

GT = "data/coco_json/val.json"
IMGS = "data/yolo/val/images"
TILED = {"mode": "tiled", "tile": 512, "overlap": 0.2, "full": True, "merge": "GREEDYNMM"}
CFGS = {
    "tiled 512 + full": (a.res, TILED),
    "standard 1600": (1600, {"mode": "standard"}),
    "standard 1280": (1280, {"mode": "standard"}),
}


def dets_path(imgsz, cfg):
    return f"results/val/dets/{tag_of(a.res, a.seed, imgsz, cfg)}.json"


dets = {}
for name, (imgsz, cfg) in CFGS.items():
    path = dets_path(imgsz, cfg)
    if not os.path.exists(path):
        if not a.make:
            raise SystemExit(f"missing {path}: run run_all.sh first, or add --make to generate it")
        row, d = run_config(weights_path(a.res, a.seed), imgsz, cfg, GT, IMGS, 1, 0, 0.001)
        os.makedirs("results/val/dets", exist_ok=True)
        json.dump(d, open(path, "w"))
    dets[name] = json.load(open(path))

with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO(GT)

print(f"model {a.res}, seed {a.seed}, val, {len(gt.getImgIds())} images")


def per_image_eval(d):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(d)
        E = COCOeval(gt, dt, "bbox")
        E.params.maxDets = [1, 10, 500]
        E.params.areaRng = [[0, 1024]]
        E.params.areaRngLbl = ["S"]
        E.evaluate()
    return E


def accumulate(E, idx):
    I = len(E.params.imgIds)
    K = len(E.params.catIds)
    rec = E.params.recThrs
    T = len(E.params.iouThrs)
    prec = -np.ones((T, len(rec), K))
    for k in range(K):
        items = [E.evalImgs[k * I + i] for i in idx]
        items = [e for e in items if e is not None]
        if not items:
            continue
        sc = np.concatenate([e["dtScores"][:500] for e in items])
        order = np.argsort(-sc, kind="mergesort")
        dtm = np.concatenate([e["dtMatches"][:, :500] for e in items], axis=1)[:, order]
        dig = np.concatenate([e["dtIgnore"][:, :500] for e in items], axis=1)[:, order]
        gig = np.concatenate([e["gtIgnore"] for e in items])
        npig = np.count_nonzero(gig == 0)
        if npig == 0:
            continue
        tp = np.cumsum(np.logical_and(dtm, np.logical_not(dig)), axis=1).astype(float)
        fp = np.cumsum(np.logical_and(np.logical_not(dtm), np.logical_not(dig)), axis=1).astype(float)
        for t in range(T):
            if tp.shape[1] == 0:
                prec[t, :, k] = 0
                continue
            r = tp[t] / npig
            q = tp[t] / (fp[t] + tp[t] + np.spacing(1))
            q = np.maximum.accumulate(q[::-1])[::-1]
            pos = np.searchsorted(r, rec, side="left")
            ok = pos < len(q)
            out = np.zeros(len(rec))
            out[ok] = q[pos[ok]]
            prec[t, :, k] = out
    v = prec[prec > -1]
    return float(v.mean()) * 100 if v.size else 0.0


evals = {n: per_image_eval(d) for n, d in dets.items()}
nI = len(gt.getImgIds())
full = np.arange(nI)
base = {n: accumulate(E, full) for n, E in evals.items()}

print("\nAP small recomputed from the saved detections")
for n in evals:
    print(f"  {n:18s} {base[n]:.2f}")

rng = np.random.default_rng(a.boot_seed)
samples = [rng.integers(0, nI, nI) for _ in range(a.boot)]
print(f"\nPaired bootstrap over images ({a.boot} resamples, seed {a.boot_seed}), tiled 512 + full minus standard")
for ref in ("standard 1600", "standard 1280"):
    diffs = np.array([accumulate(evals["tiled 512 + full"], s) - accumulate(evals[ref], s) for s in samples])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  vs {ref}: {base['tiled 512 + full'] - base[ref]:+.2f} AP small, 95% interval {lo:.2f} to {hi:.2f}")


def xyxy(b):
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], 1)


def iou(A, B):
    x1 = np.maximum(A[:, None, 0], B[None, :, 0])
    y1 = np.maximum(A[:, None, 1], B[None, :, 1])
    x2 = np.minimum(A[:, None, 2], B[None, :, 2])
    y2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    bb = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def match_stats(d):
    gts = {}
    for g in gt.dataset["annotations"]:
        gts.setdefault((g["image_id"], g["category_id"]), []).append(g)
    dts = {}
    for x in d:
        if x["score"] >= a.thr:
            dts.setdefault((x["image_id"], x["category_id"]), []).append(x)
    small = found = fp = n_det = 0
    for key in set(gts) | set(dts):
        G = gts.get(key, [])
        D = dts.get(key, [])
        n_det += len(D)
        if G and D:
            M = iou(xyxy([g["bbox"] for g in G]), xyxy([x["bbox"] for x in D]))
            hit_g = M.max(1) >= 0.5
            hit_d = M.max(0) >= 0.5
        else:
            hit_g = np.zeros(len(G), bool)
            hit_d = np.zeros(len(D), bool)
        is_small = np.array([g["area"] < 1024 for g in G], bool)
        small += int(is_small.sum())
        found += int((hit_g & is_small).sum())
        fp += int((~hit_d).sum())
    return small, found, fp, n_det


print(f"\nSmall-object recall and unmatched detections at score >= {a.thr} (IoU 0.5, same class)")
for n in ("tiled 512 + full", "standard 1280"):
    small, found, fp, n_det = match_stats(dets[n])
    print(f"  {n:18s} found {found} of {small} small objects ({100 * found / small:.1f}%), detections matching nothing {fp} of {n_det}")

if a.perclass:
    from ultralytics import YOLO
    cfg = yaml.safe_load(open("configs/data_visdrone.yaml"))
    cfg["path"] = os.path.abspath("data/yolo")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    r = YOLO(weights_path(a.res, a.seed)).val(data=tmp.name, split="val", imgsz=a.res, conf=0.001, plots=False, verbose=False)
    print(f"\nPer-class mAP50, standard inference at {a.res}, val")
    for i, c in enumerate(r.box.ap_class_index):
        print(f"  {r.names[int(c)]:18s} {100 * r.box.ap50[i]:.1f}")
    print(f"  {'all classes':18s} {100 * r.box.map50:.1f}")
