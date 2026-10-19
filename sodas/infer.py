import numpy as np
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.slicing import get_slice_bboxes

from sodas.data import image_list
from sodas.latency import gpu_info, reset_mem, peak_mem_mb, summarize, timed
from sodas.metrics import evaluate


def weights_path(res, seed):
    return f"runs/y8s_{res}_s{seed}/weights/best.pt"


def tag_of(res, seed, imgsz, cfg):
    if cfg["mode"] == "standard":
        return f"m{res}_s{seed}_std{imgsz}"
    return f"m{res}_s{seed}_t{cfg['tile']}_o{cfg['overlap']}_f{int(cfg['full'])}_{cfg['merge']}"


def load(model_path, imgsz, conf=0.001):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(model_path), confidence_threshold=conf, device=device, image_size=imgsz
    )


def make_predictor(model, cfg):
    if cfg["mode"] == "standard":
        return lambda p: get_prediction(str(p), model)
    merge = cfg.get("merge", "GREEDYNMM")
    metric = "IOU" if merge == "NMS" else "IOS"
    return lambda p: get_sliced_prediction(
        str(p), model,
        slice_height=cfg["tile"], slice_width=cfg["tile"],
        overlap_height_ratio=cfg["overlap"], overlap_width_ratio=cfg["overlap"],
        perform_standard_pred=cfg["full"],
        postprocess_type=merge, postprocess_match_metric=metric, postprocess_match_threshold=0.5,
        verbose=0,
    )


def to_coco(iid, result):
    out = []
    for o in result.object_prediction_list:
        x1, y1, x2, y2 = o.bbox.to_xyxy()
        out.append({"image_id": iid, "category_id": o.category.id + 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": o.score.value})
    return out


def n_slices(h, w, cfg):
    if cfg["mode"] == "standard":
        return 1
    n = len(get_slice_bboxes(
        image_height=h, image_width=w, slice_height=cfg["tile"], slice_width=cfg["tile"],
        auto_slice_resolution=False, overlap_height_ratio=cfg["overlap"], overlap_width_ratio=cfg["overlap"],
    ))
    return n + 1 if cfg["full"] and n > 1 else n


def run_config(model_path, imgsz, cfg, gt_json, img_dir, passes=3, warmup=5, conf=0.001, limit=None):
    model = load(model_path, imgsz, conf)
    pred = make_predictor(model, cfg)
    items = image_list(gt_json, img_dir)
    if limit:
        items = items[:limit]
    for _, p, _, _ in items[:warmup]:
        pred(p)
    reset_mem()
    dets, times = [], {iid: [] for iid, _, _, _ in items}
    stage = {"slice": [], "prediction": [], "postprocess": []}
    slices = []
    for k in range(passes):
        for iid, p, h, w in items:
            res, ms = timed(pred, p)
            times[iid].append(ms)
            if k == 0:
                dets += to_coco(iid, res)
                d = res.durations_in_seconds
                for s in stage:
                    stage[s].append(d.get(s, 0.0) * 1000)
                slices.append(n_slices(h, w, cfg))
    row = evaluate(gt_json, dets)
    row.update(summarize(list(times.values())))
    row.update(
        slice_ms=float(np.mean(stage["slice"])), forward_ms=float(np.mean(stage["prediction"])),
        merge_ms=float(np.mean(stage["postprocess"])), slices_mean=float(np.mean(slices)),
        slices_max=int(np.max(slices)), peak_mem_mb=peak_mem_mb(), gpu=gpu_info(),
    )
    return row, dets
