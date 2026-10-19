import argparse
import yaml
from sodas.infer import run_config, weights_path, tag_of
from sodas.data import append_csv, save_json

p = argparse.ArgumentParser()
p.add_argument("--res", type=int, required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--split", default="val")
a = p.parse_args()

g = yaml.safe_load(open("configs/grid.yaml"))
gt = f"data/coco_json/{a.split}.json"
imgs = f"data/yolo/{'test' if a.split == 'test' else a.split}/images"
for imgsz in g["baselines"]:
    cfg = {"mode": "standard"}
    row, dets = run_config(weights_path(a.res, a.seed), imgsz, cfg, gt, imgs, g["passes"], g["warmup"], g["conf"])
    row.update(model_res=a.res, seed=a.seed, imgsz=imgsz, **cfg)
    append_csv(f"results/{a.split}/baselines.csv", row)
    save_json(f"results/{a.split}/dets/{tag_of(a.res, a.seed, imgsz, cfg)}.json", dets)
    print(imgsz, row["AP_S"], row["lat_median_ms"])
