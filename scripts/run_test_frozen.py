import json
import yaml
from sodas.infer import run_config, weights_path, tag_of
from sodas.data import append_csv, save_json

g = yaml.safe_load(open("configs/grid.yaml"))
frozen = json.load(open("results/frozen.json"))
gt = "data/coco_json/test.json"
imgs = "data/yolo/test/images"

for res, fc in frozen.items():
    res = int(res)
    for seed in g["seeds"]:
        cfgs = [(res, {"mode": "standard"}), (960, {"mode": "standard"}), (res, {"mode": "tiled", **fc})]
        for imgsz, cfg in cfgs:
            row, dets = run_config(weights_path(res, seed), imgsz, cfg, gt, imgs, g["passes"], g["warmup"], g["conf"])
            row.update(model_res=res, seed=seed, imgsz=imgsz, **cfg)
            append_csv("results/test/test.csv", row)
            save_json(f"results/test/dets/{tag_of(res, seed, imgsz, cfg)}.json", dets)
            print(res, seed, cfg["mode"], row["AP_S"], row["lat_median_ms"])
