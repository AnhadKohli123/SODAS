import argparse
import itertools
import os
import yaml
from sodas.infer import run_config, weights_path, tag_of
from sodas.data import append_csv, save_json

p = argparse.ArgumentParser()
p.add_argument("--res", type=int, required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--split", default="val")
p.add_argument("--ablation", action="store_true")
a = p.parse_args()

g = yaml.safe_load(open("configs/grid.yaml"))
merges = g["merge_ablation"] if a.ablation else g["merge"]
gt = f"data/coco_json/{a.split}.json"
imgs = f"data/yolo/{a.split}/images"
out_csv = f"results/{a.split}/grid.csv"

for tile, ov, full, merge in itertools.product(g["tile_sizes"], g["overlaps"], g["full_image"], merges):
    cfg = {"mode": "tiled", "tile": tile, "overlap": ov, "full": full, "merge": merge}
    tag = tag_of(a.res, a.seed, a.res, cfg)
    dpath = f"results/{a.split}/dets/{tag}.json"
    if os.path.exists(dpath):
        continue
    row, dets = run_config(weights_path(a.res, a.seed), a.res, cfg, gt, imgs, g["passes"], g["warmup"], g["conf"])
    row.update(model_res=a.res, seed=a.seed, imgsz=a.res, **cfg)
    append_csv(out_csv, row)
    save_json(dpath, dets)
    print(tag, row["AP_S"], row["lat_median_ms"])
