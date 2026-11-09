"""Client-side LoRA track: freeze the base, train adapters, emit a compact payload.

The test uses a toy module with the same attribute names as SotaGPT so it can
run on CPU in a second. It proves the wiring, not 658M-model quality; the
full-model held-out comparison is a separate, documented unknown.
"""
from __future__ import annotations

import gzip
import importlib.util
import io
import logging
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client_module():
    old = os.environ.copy()
    os.environ["CROWDGPT_LORA"] = "1"
    os.environ["CROWDGPT_LORA_RANK"] = "4"
    tmp = tempfile.mkdtemp(prefix="ng-lora-client-")
    shutil.copy2(ROOT / "client.py", Path(tmp) / "client.py")
    spec = importlib.util.spec_from_file_location("ng_lora_client", Path(tmp) / "client.py")
    mod = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    saved_stdio = (sys.stdout, sys.stderr)
    root = logging.getLogger(); saved_handlers, saved_level = list(root.handlers), root.level
    fake = [io.TextIOWrapper(io.BytesIO(), encoding="utf-8") for _ in range(2)]
    try:
        # client.py wraps stdout/stderr around their .buffer at import time;
        # throwaway streams keep pytest's capture files untouched.
        sys.stdout, sys.stderr = fake
        os.chdir(tmp)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
        sys.stdout, sys.stderr = saved_stdio
        root.handlers[:] = saved_handlers; root.setLevel(saved_level)
        os.environ.clear(); os.environ.update(old)
    yield mod
    shutil.rmtree(tmp, ignore_errors=True)


class _Lin(nn.Module):
    def __init__(self, out_features, in_features):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = _Lin(8, 8); self.wk = _Lin(8, 8)
        self.wv = _Lin(8, 8); self.wo = _Lin(8, 8)


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = _Lin(16, 8); self.w2 = _Lin(16, 8); self.w3 = _Lin(8, 16)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attn(); self.mlp = _Mlp()


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block()])


def test_apply_lora_freezes_base_and_trains_only_adapters(client_module):
    model = _Toy()
    client_module._apply_lora(model, 4)
    assert model.is_lora is True and model.lora_rank == 4
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) == 14  # 7 adapters x 2 factors per block
    assert all(".lora_" in n for n in trainable)
    base = [p for n, p in model.named_parameters() if ".lora_" not in n]
    assert base and not any(p.requires_grad for p in base)


def test_lora_payload_uses_base_weight_offsets(client_module):
    model = _Toy()
    client_module._apply_lora(model, 4)
    payload = client_module._lora_payload(model)
    assert payload.startswith(b"NGL1")
    count = struct.unpack_from("<I", payload, 4)[0]
    assert count == 7  # seven adapter targets on the single block
    header = struct.Struct("<HIIIH")
    nlen, offset, rows, cols, rank = header.unpack_from(payload, 8)
    name = payload[8 + header.size:8 + header.size + nlen].decode()
    assert name == "blocks.0.attn.wq.weight"
    assert (offset, rows, cols, rank) == (0, 8, 8, 4)
    # second target starts after the 64-element wq weight
    pos = 8 + header.size + nlen + rows * rank * 2 + rank * cols * 2
    nlen2, offset2, rows2, cols2, rank2 = header.unpack_from(payload, pos)
    assert offset2 == 64 and (rows2, cols2, rank2) == (8, 8, 4)


def test_lora_delta_file_wraps_header_then_payload(client_module, tmp_path):
    model = _Toy()
    client_module._apply_lora(model, 4)
    with torch.no_grad():
        for block in model.blocks:
            block.attn.lora_wq.B.fill_(0.5)
    payload = client_module._lora_payload(model)
    header = b'{"isLoRA":true,"loraRank":4}'
    path = tmp_path / "delta.gz"
    size = client_module._write_lora_delta_file(path, struct.pack("<I", len(header)) + header, payload)
    assert size == path.stat().st_size
    with gzip.open(path, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        assert f.read(n) == header
        assert f.read() == payload
