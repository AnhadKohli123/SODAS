import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--image", required=True)
p.add_argument("--crop", type=int, nargs=4, required=True, metavar=("X", "Y", "W", "H"))
p.add_argument("--std", required=True)
p.add_argument("--tiled", required=True)
p.add_argument("--gt", default="data/coco_json/test.json")
p.add_argument("--img-dir", default="data/yolo/test/images")
p.add_argument("--thr", type=float, default=0.3)
p.add_argument("--out", default="figures/qualitative.png")
a = p.parse_args()

gt = json.load(open(a.gt))
iid = next(i["id"] for i in gt["images"] if i["file_name"] == a.image)
img = Image.open(f"{a.img_dir}/{a.image}").convert("RGB")
x0, y0, cw, ch = a.crop
crop = img.crop((x0, y0, x0 + cw, y0 + ch))


def boxes(items, key=None, thr=0.0):
    out = []
    for d in items:
        if d["image_id"] != iid or (key and d["score"] < thr):
            continue
        x, y, w, h = d["bbox"]
        if x + w > x0 and x < x0 + cw and y + h > y0 and y < y0 + ch:
            out.append((x - x0, y - y0, w, h))
    return out


panels = [
    ("ground truth", boxes(gt["annotations"]), "lime"),
    ("standard", boxes(json.load(open(a.std)), True, a.thr), "red"),
    ("tiled", boxes(json.load(open(a.tiled)), True, a.thr), "cyan"),
]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (title, bs, c) in zip(axes, panels):
    ax.imshow(crop)
    for x, y, w, h in bs:
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor=c, linewidth=1.2))
    ax.set_title(f"{title} ({len(bs)})")
    ax.axis("off")
fig.savefig(a.out, dpi=200, bbox_inches="tight")

