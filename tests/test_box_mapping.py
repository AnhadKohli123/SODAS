import os
import pytest
from sodas.controls import corrupt_offsets
from sodas.infer import run_config
from sodas.metrics import evaluate

M, GT, IMG = os.environ.get("SODAS_MODEL"), os.environ.get("SODAS_GT"), os.environ.get("SODAS_IMG")
RES = int(os.environ.get("SODAS_RES", "640"))
pytestmark = pytest.mark.skipif(not (M and GT and IMG), reason="set SODAS_MODEL, SODAS_GT, SODAS_IMG")


def test_corrupted_offsets_collapse_ap_s():
    cfg = {"mode": "tiled", "tile": 512, "overlap": 0.2, "full": False, "merge": "GREEDYNMM"}
    row, dets = run_config(M, RES, cfg, GT, IMG, passes=1, warmup=0, limit=100)
    bad = evaluate(GT, corrupt_offsets(dets, 512))
    assert bad["AP_S"] < 0.2 * row["AP_S"]
