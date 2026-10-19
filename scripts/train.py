import argparse
import yaml
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--cfg", required=True)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

cfg = yaml.safe_load(open(a.cfg))
model = YOLO(cfg.pop("model"))
model.train(**cfg, seed=a.seed, name=f"y8s_{cfg['imgsz']}_s{a.seed}")
