"""The client encoder and server decoder must share one 1-bit format."""
from __future__ import annotations

import importlib.util
import io
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT.parent / "CrowdGPT-Server"
if not (SERVER / "orchestrator" / "compression.py").exists():
    pytest.skip("server checkout not present", allow_module_level=True)
sys.path.insert(0, str(SERVER))


@pytest.fixture(scope="module")
def client_module():
    tmp = tempfile.mkdtemp(prefix="ng-codec-")
    shutil.copy2(ROOT / "client.py", Path(tmp) / "client.py")
    spec = importlib.util.spec_from_file_location("ng_codec_client", Path(tmp) / "client.py")
    mod = importlib.util.module_from_spec(spec)
    saved = (sys.stdout, sys.stderr)
    root = logging.getLogger(); handlers, level = list(root.handlers), root.level
    fake = [io.TextIOWrapper(io.BytesIO(), encoding="utf-8") for _ in range(2)]
    cwd = os.getcwd()
    try:
        sys.stdout, sys.stderr = fake
        os.chdir(tmp)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
        sys.stdout, sys.stderr = saved
        root.handlers[:] = handlers; root.setLevel(level)
    yield mod
    shutil.rmtree(tmp, ignore_errors=True)


def test_client_block_decodes_with_server_codec(client_module):
    from orchestrator.compression import decompress_delta

    rng = np.random.default_rng(2)
    values = (rng.standard_normal(4096) * 0.01).astype(np.float32)
    blob = client_module._ngd1_compress(values)
    back = decompress_delta(blob)
    assert back.shape == values.shape
    assert np.all(np.sign(back) == np.sign(values))
    assert np.mean(np.abs(back - values)) < np.mean(np.abs(values))


def test_codec_size_target_for_a_million_params(client_module):
    values = np.zeros(1_000_000, dtype=np.float32)
    blob = client_module._ngd1_compress(values)
    assert len(blob) < 200_000
