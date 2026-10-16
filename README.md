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
