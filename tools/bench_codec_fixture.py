#!/usr/bin/env python3
"""Measure the NGD1 codec on a slice of a real model update.

One full-parameter SGD step is taken on the 658M model, and a contiguous slice
of the resulting weight delta is compressed and decompressed. This is a local
model delta, not a captured production client delta; production capture still
requires a live round. Output is JSON for the docs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT.parent / "CrowdGPT-Server"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SERVER))
import client  # noqa: E402
from orchestrator.compression import compress_delta, decompress_delta  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=int, default=64_000_000)
    parser.add_argument("--seq", type=int, default=128)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    client.ENGRAM_ACTIVE = False
    client.LOSS_CHUNK = 64
    torch.manual_seed(7)
    model = client.SotaGPT("CUDA", device, gpu_layers=client.N_LAYERS, bs=1, sl=args.seq)
    model.use_ckpt = True
    flat = [p for n, p in model.named_parameters() if not n.startswith("engram.")]
    n = min(int(args.params), flat[0].numel())
    before = flat[0].detach().reshape(-1)[:n].float().cpu().numpy().copy()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    x = torch.randint(0, client.VOCAB_SIZE, (1, args.seq), device=device)
    y = torch.randint(0, client.VOCAB_SIZE, (1, args.seq), device=device)
    opt.zero_grad(set_to_none=True)
    loss, _ = client._fwl(model, x, y, args.seq, True, torch.bfloat16)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    delta = flat[0].detach().reshape(-1)[:n].float().cpu().numpy() - before
    start = time.perf_counter()
    blob = compress_delta(delta)
    encoded_s = time.perf_counter() - start
    start = time.perf_counter()
    back = decompress_delta(blob)
    decoded_s = time.perf_counter() - start
    err = np.abs(back - delta)
    print(json.dumps({
        "fixture": "local 658M model, one full-parameter SGD step, first contiguous slice",
        "params": int(n), "dense_bytes": int(delta.nbytes),
        "compressed_bytes": len(blob), "ratio": delta.nbytes / len(blob),
        "bytes_per_param": len(blob) / n,
        "mean_abs_error": float(err.mean()),
        "mean_abs_value": float(np.abs(delta).mean()),
        "encode_s": round(encoded_s, 4), "decode_s": round(decoded_s, 4),
        "loss": float(loss),
        "production_capture": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
