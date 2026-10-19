import csv
import json
from pathlib import Path


def load_coco(path):
    with open(path) as f:
        return json.load(f)


def image_list(gt_json, img_dir):
    d = load_coco(gt_json)
    return [(im["id"], Path(img_dir) / im["file_name"], im["height"], im["width"]) for im in d["images"]]


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def append_csv(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    new = not Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
