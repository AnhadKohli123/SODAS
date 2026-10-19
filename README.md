REPO structure
```
SODAS/
├── README.md
├── requirements.txt
├── .gitignore                  # data/, runs/, *.pt
├── configs/
│   ├── data_visdrone.yaml      # ultralytics dataset file
│   ├── train_640.yaml
│   ├── train_960.yaml
│   └── grid.yaml               # tile sizes 320/512/640/800 × overlap × full-image pass × merge rule
├── data/                       # gitignored
│   ├── raw/VisDrone2019-DET-{train,val,test-dev}/
│   ├── yolo/                   # converted labels + images
│   └── coco_json/              # val/test GT in COCO format for pycocotools
├── scripts/
│   ├── download_visdrone.sh
│   ├── convert_visdrone.py     # VisDrone → YOLO + COCO json
│   ├── train.py                # YOLOv8s @640 / @960, seeds
│   ├── run_baselines.py        # standard inference at 640, 960, (1280)
│   ├── run_sahi_grid.py        # 16 tiled configs per model, saves detections + per-stage timings
│   ├── select_config.py        # best on val, cheapest within 1 SE
│   └── run_test_frozen.py      # score frozen configs once on test-dev
├── sodas/
│   ├── data.py
│   ├── infer.py                # standard + SAHI wrappers
│   ├── latency.py              # timer: slice / forward / merge, median + IQR, GPU info
│   ├── metrics.py              # AP, AP_S, size-bin recall, efficiency η
│   └── controls.py             # identity tiling, shuffled box mapping, etc.
├── tests/
│   ├── test_identity_tiling.py # must match standard AP within 0.2
│   └── test_box_mapping.py
├── results/
│   ├── val/ test/              # csv per config
│   └── tables/                 # latency-vs-accuracy table
├── figures/
│   ├── make_scatter.py         # (latency, AP_S)
│   ├── make_heatmap.py         # tile size × overlap
│   └── make_qualitative.py     # standard vs tiled crops
├── notebooks/
│   └── explore.ipynb
├── runs/                       # gitignored, training outputs
└── report/                     # CVPR-format LaTeX
```
## Run locally (NVIDIA GPU)

Needs Python 3.10+, an NVIDIA GPU with 8 GB+ VRAM (6 GB works at 640 with a smaller batch), and CUDA drivers.

### 1. Setup
```bash
git clone https://github.com/AnhadKohli123/SODAS.git
cd SODAS
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
The last line must print `True` and your GPU name. Pick the torch CUDA build that matches your driver at pytorch.org.

### 2. Data
```bash
bash scripts/download_visdrone.sh     # Windows: run in Git Bash, or download the zips manually into data/raw
python scripts/convert_visdrone.py
```
Then edit `configs/data_visdrone.yaml` so `path:` is the absolute path to `data/yolo` on your machine, for example `/home/you/SODAS/data/yolo` or `C:/Users/you/SODAS/data/yolo`.

### 3. Smoke test (about 10 min)
Set `epochs: 1` in `configs/train_640.yaml` temporarily, then:
```bash
python scripts/train.py --cfg configs/train_640.yaml --seed 0
export SODAS_MODEL=runs/y8s_640_s0/weights/best.pt
export SODAS_GT=data/coco_json/val.json
export SODAS_IMG=data/yolo/val/images
pytest -q
```
On Windows PowerShell, use `$env:SODAS_MODEL="runs/y8s_640_s0/weights/best.pt"` (and likewise for the other two) instead of `export`.

Then set `epochs` back to 100 and delete the test run: `rm -rf runs/y8s_640_s0` (PowerShell: `Remove-Item -Recurse runs/y8s_640_s0`).

### 4. Full training
```bash
for res in 640 960; do
  for seed in 0 1; do
    python scripts/train.py --cfg configs/train_$res.yaml --seed $seed
  done
done
```
If 960 runs out of memory, lower `batch` in `configs/train_960.yaml`. To resume an interrupted run:
`yolo train resume model=runs/y8s_640_s0/weights/last.pt`

### 5. Evaluate on val
```bash
for res in 640 960; do
  for seed in 0 1; do
    python scripts/run_baselines.py --res $res --seed $seed
    python scripts/run_sahi_grid.py --res $res --seed $seed
  done
done
```
The grid is resumable: rerun the same command and it skips finished configs. For the merge-rule ablation, add `--ablation` to the grid command.

### 6. Freeze the config and score test-dev once
```bash
python scripts/select_config.py
python scripts/run_test_frozen.py
```

### 7. Figures
```bash
mkdir -p figures
python figures/make_scatter.py
python figures/make_heatmap.py --res 640 --full 0
python figures/make_heatmap.py --res 960 --full 0
```

### Notes
- Latency numbers are only comparable on the same GPU. Run all timing on one machine, and keep nothing else running on the GPU.
- Results land in `results/` (CSV and detection JSON), weights in `runs/`, and figures in `figures/`.
