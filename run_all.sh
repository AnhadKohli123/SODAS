#!/usr/bin/env bash
# Reproduces the val tables and the frozen test-dev run. Usage:
#   bash run_all.sh                full pipeline (trains 4 models, about 19 h on a T4)
#   SKIP_TRAIN=1 bash run_all.sh   reuse runs/y8s_{640,960}_s{0,1}/weights/best.pt
# The committed results/ files are replaced by freshly computed ones (git checkout results restores them).
set -euo pipefail
cd "$(dirname "$0")"

pip install -q -r requirements.txt
pip install -q -e .
mkdir -p runs results/val results/test/dets logs figures

rm -f results/val/baselines.csv results/val/grid.csv results/test/test.csv results/frozen.json

bash scripts/download_visdrone.sh
python scripts/convert_visdrone.py
sed -i "s#^path: .*#path: $(pwd)/data/yolo#" configs/data_visdrone.yaml

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  for res in 640 960; do
    for seed in 0 1; do
      if [ ! -f runs/y8s_${res}_s${seed}/weights/best.pt ]; then
        python scripts/train.py --cfg configs/train_${res}.yaml --seed $seed 2>&1 | tee logs/train_${res}_s${seed}.txt
      fi
    done
  done
fi

for res in 640 960; do
  for seed in 0 1; do
    python scripts/run_baselines.py --res $res --seed $seed 2>&1 | { grep -v 'ignoring invalid prediction' || true; } | tee -a logs/val_eval.txt
    python scripts/run_sahi_grid.py --res $res --seed $seed 2>&1 | { grep -v 'ignoring invalid prediction' || true; } | tee -a logs/val_eval.txt
  done
done

python scripts/select_config.py | tee logs/select_config.txt
cat results/frozen.json
python scripts/run_test_frozen.py 2>&1 | { grep -v 'ignoring invalid prediction' || true; } | tee logs/test_frozen.txt

python figures/make_report_figures.py

python - <<'PY'
import csv

def read(path):
    rows = list(csv.reader(open(path)))
    head = rows[0] + ["tile", "overlap", "full", "merge"]
    return [dict(zip(head, r)) for r in rows[1:]]

def num(v):
    try:
        return "%.1f" % float(v)
    except (TypeError, ValueError):
        return str(v)

cols = ["model_res", "seed", "imgsz", "mode", "AP_S", "R50_tiny", "lat_median_ms", "tile", "overlap", "full"]
for name, path in (("VAL standard", "results/val/baselines.csv"), ("VAL tiled", "results/val/grid.csv"), ("TEST-DEV (frozen)", "results/test/test.csv")):
    print("\n" + name)
    print(" ".join(cols))
    for d in read(path):
        print(" ".join(num(d[c]) if c in ("AP_S", "R50_tiny", "lat_median_ms") else str(d.get(c, "")) for c in cols))
PY
