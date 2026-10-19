import time
import numpy as np
import torch


def gpu_info():
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return f"{p.name}|cuda{torch.version.cuda}|torch{torch.__version__}"
    return f"cpu|torch{torch.__version__}"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, *args):
    sync()
    t = time.perf_counter()
    out = fn(*args)
    sync()
    return out, (time.perf_counter() - t) * 1000


def reset_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_mem_mb():
    return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0


def summarize(per_image_ms):
    m = np.array([np.median(v) for v in per_image_ms])
    return {"lat_median_ms": float(np.median(m)), "lat_q1_ms": float(np.percentile(m, 25)), "lat_q3_ms": float(np.percentile(m, 75))}
