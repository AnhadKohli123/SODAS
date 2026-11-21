#!/usr/bin/env bash
# Reproduces the val tables and the frozen test-dev run. Usage:
#   bash run_all.sh            full pipeline (trains 4 models, many hours on a T4)
#   SKIP_TRAIN=1 bash run_all.sh   reuse runs/y8s_{640,960}_s{0,1}/weights/best.pt
set -euo pipefail
cd "$(dirname "$0")"

pip install -q -r requirements.txt
pip install -q -e .
mkdir -p runs results/val results/test/dets results/tables figures logs

bash scripts/download_visdrone.sh
python scripts/convert_visdrone.py
sed -i "s#^path: .*#path: $(pwd)/data/yolo#" configs/data_visdrone.yaml

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  for res in 640 960; do
    for seed in 0 1; do
      [ -f runs/y8s_${res}_s${seed}/weights/best.pt ] || \
        python scripts/train.py --cfg configs/train_${res}.yaml --seed $seed 2>&1 | tee logs/train_${res}_s${seed}.txt
    done
  done
fi

for res in 640 960; do
  for seed in 0 1; do
    python scripts/run_baselines.py --res $res --seed $seed 2>&1 | grep -v 'ignoring invalid prediction' | tee -a logs/val_eval.txt
    python scripts/run_sahi_grid.py --res $res --seed $seed 2>&1 | grep -v 'ignoring invalid prediction' | tee -a logs/val_eval.txt
  done
done

python scripts/select_config.py | tee logs/select_config.txt
cat results/frozen.json
python scripts/run_test_frozen.py 2>&1 | grep -v 'ignoring invalid prediction' | tee logs/test_frozen.txt

python figures/make_scatter.py
python figures/make_heatmap.py --res 640 --full 0
python figures/make_heatmap.py --res 960 --full 0

python - <<'PY'
import pandas as pd
b = pd.read_csv("results/val/baselines.csv"); g = pd.read_csv("results/val/grid.csv")
cols = ["model_res", "seed", "imgsz", "mode", "AP_S", "R50_tiny", "lat_median_ms"]
print("\nVAL standard:\n", b[cols].round(1).to_string(index=False))
print("\nVAL tiled:\n", g[cols + ["tile", "overlap", "full"]].round(1).to_string(index=False))
print("\nTEST-DEV (frozen):\n", pd.read_csv("results/test/test.csv")[cols].round(1).to_string(index=False))
PY
