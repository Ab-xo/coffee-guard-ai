"""CPU latency and size of an ONNX bundle (ONNX Runtime, batch 1, this machine).

Three timings, each p50/p95 over ``runs`` calls after ``warmup`` calls:
- ``model``: the ONNX forward pass alone;
- ``predict``: preprocessing + forward + softmax from a decoded photo;
- ``end_to_end``: decode an original full-resolution phone JPEG (bytes) + predict -
  what the API does per request (before the gates, which are negligible).
"""

from __future__ import annotations

import io
import os
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

from coffeeguard.inference.predictor import Predictor


def _timeit(fn, runs: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95))}


def onnx_params(model_path: Path) -> int:
    import onnx

    m = onnx.load(str(model_path), load_external_data=False)
    return int(sum(np.prod(t.dims) for t in m.graph.initializer))


def benchmark_bundle(
    bundle_dir: Path, photo: Path, runs: int = 200, warmup: int = 20, threads: int | None = None
) -> dict:
    predictor = Predictor(bundle_dir, threads=threads)
    raw = photo.read_bytes()
    with Image.open(io.BytesIO(raw)) as im:
        img = im.convert("RGB")
    x = predictor.prepare(img)[0][None]

    def end_to_end():
        with Image.open(io.BytesIO(raw)) as im:
            predictor.predict(im)

    return {
        "bundle": bundle_dir.name,
        "model": predictor.meta.get("model"),
        "onnx_mb": (bundle_dir / "model.onnx").stat().st_size / 1e6,
        "params_m": onnx_params(bundle_dir / "model.onnx") / 1e6,
        "photo": {"file": photo.name, "size": list(img.size), "bytes": len(raw)},
        "threads": predictor.session.get_session_options().intra_op_num_threads,
        "machine": {"cpu": platform.processor(), "logical_cores": os.cpu_count()},
        "model_only": _timeit(lambda: predictor.run(x), runs, warmup),
        "predict": _timeit(lambda: predictor.predict(img), runs, warmup),
        "end_to_end": _timeit(end_to_end, max(runs // 4, 20), max(warmup // 4, 5)),
    }
