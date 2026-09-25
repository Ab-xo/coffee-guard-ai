from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from coffeeguard.utils.hashing import fingerprint, sha256_bytes, sha256_file
from coffeeguard.utils.io import read_json, write_json
from coffeeguard.utils.runs import capture_env, create_run_dir
from coffeeguard.utils.seed import set_seed


def test_sha256_file_matches_bytes(tmp_path: Path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"coffee")
    assert sha256_file(p) == sha256_bytes(b"coffee")


def test_fingerprint_is_order_independent():
    assert fingerprint(["b", "a", "c"]) == fingerprint(["c", "b", "a"])
    assert fingerprint(["a"]) != fingerprint(["a", "b"])


def test_set_seed_is_reproducible():
    set_seed(123)
    a = (random.random(), np.random.rand())  # noqa: NPY002
    set_seed(123)
    b = (random.random(), np.random.rand())  # noqa: NPY002
    assert a == b


def test_json_handles_numpy_and_paths(tmp_path: Path):
    out = write_json(
        tmp_path / "x" / "m.json", {"f": np.float32(0.5), "a": np.arange(3), "p": Path("a/b")}
    )
    assert read_json(out) == {"f": 0.5, "a": [0, 1, 2], "p": "a/b"}


def test_run_dir_and_env(tmp_path: Path):
    run = create_run_dir("effnet v2/b0", root=tmp_path)
    assert run.is_dir() and run.name.endswith("effnet-v2-b0")
    env = capture_env()
    assert {"python", "platform", "packages", "git"} <= env.keys()
