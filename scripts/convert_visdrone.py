import json
import shutil
from pathlib import Path
from PIL import Image

RAW = Path("data/raw")
OUT = Path("data/yolo")
COCO = Path("data/coco_json")
SPLITS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val", "test": "VisDrone2019-DET-test-dev"}
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]

COCO.mkdir(parents=True, exist_ok=True)

for split, folder in SPLITS.items():
    img_out = OUT / split / "images"
    lab_out = OUT / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)
    images, anns = [], []
    aid = 1
    for iid, ip in enumerate(sorted((RAW / folder / "images").glob("*.jpg")), 1):
        w, h = Image.open(ip).size
        shutil.copy2(ip, img_out / ip.name)
        images.append({"id": iid, "file_name": ip.name, "width": w, "height": h})
        lines = []
        ap = RAW / folder / "annotations" / (ip.stem + ".txt")
        if ap.exists():
            for row in ap.read_text().strip().splitlines():
                v = row.strip().split(",")
                x, y, bw, bh, score, cat = (int(float(t)) for t in v[:6])
                if score == 0 or cat < 1 or cat > 10 or bw <= 0 or bh <= 0:
                    continue
                cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
                lines.append(f"{cat - 1} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
                anns.append({"id": aid, "image_id": iid, "category_id": cat, "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
                aid += 1
        (lab_out / (ip.stem + ".txt")).write_text("\n".join(lines))
    if split != "train":
        cats = [{"id": i + 1, "name": n} for i, n in enumerate(NAMES)]
        json.dump({"images": images, "annotations": anns, "categories": cats}, open(COCO / f"{split}.json", "w"))
    print(split, len(images), len(anns))
