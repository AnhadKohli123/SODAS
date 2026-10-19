import os
import pytest
from sodas.controls import identity_cfg
from sodas.infer import run_config

M, GT, IMG = os.environ.get("SODAS_MODEL"), os.environ.get("SODAS_GT"), os.environ.get("SODAS_IMG")
RES = int(os.environ.get("SODAS_RES", "640"))
pytestmark = pytest.mark.skipif(not (M and GT and IMG), reason="set SODAS_MODEL, SODAS_GT, SODAS_IMG")


def test_identity_tiling_matches_standard():
    std, _ = run_config(M, RES, {"mode": "standard"}, GT, IMG, passes=1, warmup=0, limit=100)
    til, _ = run_config(M, RES, identity_cfg(), GT, IMG, passes=1, warmup=0, limit=100)
    assert abs(std["AP"] - til["AP"]) <= 0.2
