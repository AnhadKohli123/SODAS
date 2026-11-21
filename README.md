# SODAS: Small-Object Detection in Aerial Scenes

UCS532 Open Challenge, Problem P10. Thapar Institute of Engineering and Technology, Patiala.
Anhad Kohli (1024170401), Jasmandeep Sharma (1024170269). Supervisor: Dr. Shailendra Tiwari.

## Problem
YOLOv8s resizes drone images to 640 px and loses small objects. SAHI tiled inference recovers them but costs time. We measure whether tiling beats simply raising the input resolution, at equal latency, and whether the answer changes when the model is trained at 960 px.

## Dataset
VisDrone2019-DET (https://github.com/VisDrone/VisDrone-Dataset): 6,471 train, 548 val, 1,610 test-dev images, 10 classes. Ignored regions and "others" are dropped in conversion. `bash scripts/download_visdrone.sh` downloads it and `python scripts/convert_visdrone.py` converts it to YOLO and COCO format. Nothing is committed to the repo.

## Protocol
All design choices are made on val. `scripts/select_config.py` picks the cheapest configuration within one standard error of the best AP small and writes `results/frozen.json`. `scripts/run_test_frozen.py` then scores test-dev once on that frozen configuration. No tuning on test-dev.

## Reproduce
```
bash run_all.sh                  # full pipeline including training
SKIP_TRAIN=1 bash run_all.sh     # reuse trained checkpoints in runs/
```
Seeds 0 and 1 are fixed (`configs/grid.yaml`) and printed in the logs. Checkpoints and the val grid CSVs are not stored in the repo; download them from https://www.kaggle.com/datasets/akohli123/sodasweights

## Hardware and runtime
Tesla T4 (Google Colab and Kaggle). Training: up to 100 epochs per model, four models (640 and 960 input, seeds 0 and 1), about 19 hours in total (640: 3.6 and 3.8 h, 960: 5.65 and 5.64 h; the 640 seed-1 run was resumed once). Evaluation: the val grids took roughly 4.5 hours across several sessions and the test-dev run about 3.7 hours (about 8 hours in total). Latency is the median per image, end to end including image read.

## Results
All numbers are from the committed files in `results/` and `logs/`.

Val (548 images), AP small (seed 0 / seed 1), tiny-object recall R50 and median latency. The 640 model's seed-1 grid is reduced (see the report); The 960 rows are the mean of two seeds, with both seeds in the AP small column:

| Model | Inference | AP small (s0 / s1) | Tiny R50 | Latency |
|---|---|---|---|---|
| 640 | standard 640 | 12.3 / 12.3 | 22.5 | 28.7 ms |
| 640 | standard 1280 | 20.1 / 19.8 | 43.8 | 36.9 ms |
| 640 | tile 512 + full | 21.1 / 20.4 | 57.5 | 168 ms |
| 960 | standard 960 | 18.8 / 18.8 | 37.8 | 36 ms |
| 960 | standard 1600 | 23.7 / 23.5 | 51.1 | 70 ms |
| 960 | tile 640 + full | 24.3 / 24.1 | 60.5 | 293 ms |

The 640 rows' Tiny R50 and latency are seed 0.

Test-dev (1,610 images), configuration frozen on val, one run, mean of two seeds (`results/test/test.csv`):

| Model | Inference | AP small | Tiny R50 | Latency |
|---|---|---|---|---|
| 640 | standard 640 | 7.7 | 15.7 | 30.0 ms |
| 640 | standard 1600 | 13.8 | 37.3 | 71.7 ms |
| 640 | tile 512 + full | 13.8 | 43.5 | 210.5 ms |
| 960 | standard 960 | 11.9 | 27.5 | 36.1 ms |
| 960 | standard 1600 | 15.5 | 38.8 | 71.6 ms |
| 960 | tile 640 + full | 16.0 | 47.0 | 290.9 ms |

On test-dev the 640 model's advantage over high-resolution inference disappears (13.6 and 13.9 AP small for tiling against 13.8 and 13.8 at 1600 px in the two seeds). The 960 model keeps a small one (+0.3, +0.7) and tiling still raises tiny-object recall by 6 to 8 points.

Full tables, ablation, failure analysis and the list of changes from the proposal are in the report.

## Run step by step
Needs Python 3.10+, an NVIDIA GPU with 8 GB+ VRAM (6 GB works at 640 with a smaller batch) and CUDA drivers. `bash run_all.sh` does all of this in one go.

1. Setup
```
git clone https://github.com/AnhadKohli123/SODAS.git
cd SODAS
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
2. Data
```
bash scripts/download_visdrone.sh     # Windows: Git Bash, or download the zips into data/raw
python scripts/convert_visdrone.py
```
Then set `path:` in `configs/data_visdrone.yaml` to the absolute path of `data/yolo`.

3. Smoke test (about 10 min): set `epochs: 1` in `configs/train_640.yaml`, run `python scripts/train.py --cfg configs/train_640.yaml --seed 0`, set `SODAS_MODEL=runs/y8s_640_s0/weights/best.pt`, `SODAS_GT=data/coco_json/val.json`, `SODAS_IMG=data/yolo/val/images`, run `pytest -q`, then set epochs back to 100 and delete the test run.

4. Training: `python scripts/train.py --cfg configs/train_$res.yaml --seed $seed` for res in 640, 960 and seed in 0, 1. Resume an interrupted run with `yolo train resume model=runs/y8s_640_s0/weights/last.pt`. Lower `batch` in `configs/train_960.yaml` if 960 runs out of memory.

5. Val evaluation: `python scripts/run_baselines.py --res $res --seed $seed` and `python scripts/run_sahi_grid.py --res $res --seed $seed`. The grid is resumable. `configs/grid.yaml` lists the full grid; the reported runs used a subset (9 tiled settings for the 640 model, 4 for the 960 model, see "Changes from the proposal" in the report).

6. Freeze and score test-dev once: `python scripts/select_config.py`, then `python scripts/run_test_frozen.py`.

7. Figures: `python figures/make_report_figures.py` draws the report's frontier and recall figures from `results/val/*.csv`; `figures/qualitative.py` draws qualitative crops (ground truth, standard and tiled detections for one image region). The two crop figures in the report were made from validation detections with this kind of script; the exact image names and crop boxes were not recorded, so they cannot be regenerated exactly.

Latency is only comparable on the same GPU: run all timing on one machine and keep nothing else on the GPU.

## Layout
```
SODAS/
├── README.md, run_all.sh, requirements.txt, .gitignore
├── configs/      data_visdrone.yaml, train_640.yaml, train_960.yaml, grid.yaml
├── scripts/      download_visdrone.sh, convert_visdrone.py, train.py, run_baselines.py,
│                 run_sahi_grid.py, select_config.py, run_test_frozen.py
├── sodas/        data.py, infer.py, latency.py, metrics.py, controls.py
├── tests/        test_identity_tiling.py, test_box_mapping.py
├── figures/      make_report_figures.py, qualitative.py
├── results/      val/ (grid.csv, baselines.csv), test/ (test.csv), frozen.json
└── logs/         training results.csv files, train_640_s1_resume.txt
```
`data/` and `runs/` are created locally and are not committed.

## Credits
Ultralytics YOLOv8 (AGPL-3.0), SAHI (Akyon et al.), pycocotools, VisDrone dataset (Zhu et al.). The inference grid, protocol, size-binned recall, bootstrap and cost-model analysis are our own.

## AI declaration
Claude (Anthropic) was used to help write and debug scripts, analyse results, and draft and typeset the report and slides. All experiments were run by the authors, and every reported number comes from the logs in this repository.
