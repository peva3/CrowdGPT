#!/usr/bin/env python3
"""Microbenchmark the activation-checkpointing path in the training step.

This is a small CPU model, not the 658M production model. It measures the
direction and rough size of the FLOP overhead that P5.5 removes; it is not a
substitute for a production tokens/s run. Run:

    python tools/bench_model_paths.py
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import client


def _tiny_model():
    client.MODEL_CONFIG.update({
        "vocabSize": 64, "dim": 32, "nLayers": 2, "nHeads": 4,
        "nKvHeads": 1, "headDim": 8, "maxSeqLen": 32, "mlpHidden": 64,
        "weightTying": True,
    })
    client.VOCAB_SIZE = 64
    client.DIM = 32
    client.N_LAYERS = 2
    client.N_HEADS = 4
    client.N_KV_HEADS = 1
    client.HEAD_DIM = 8
    client.MAX_SEQ_LEN = 32
    client.MLP_HIDDEN = 64
    client.ENG_NUM_BUCKETS = 16
    client.ENGRAM_ACTIVE = False
    client.LOSS_CHUNK = 16
    torch.manual_seed(42)
    return client.SotaGPT("CPU", torch.device("cpu"), gpu_layers=0, bs=2, sl=16)


def _measure(model, x, y, steps, warmup):
    model.use_ckpt = model.use_ckpt  # explicit for readers
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    times = []
    for i in range(warmup + steps):
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss, _ = client._fwl(model, x, y, 16, False, None)
        loss.backward()
        opt.step()
        if i >= warmup:
            times.append(time.perf_counter() - start)
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))

    results = {}
    for use_ckpt in (True, False):
        model = _tiny_model()
        model.use_ckpt = use_ckpt
        times = _measure(model, x, y, args.steps, args.warmup)
        tokens = 2 * 16
        results[use_ckpt] = {
            "median_s": statistics.median(times),
            "tokens_per_s": tokens / statistics.median(times),
        }
    on = results[True]
    off = results[False]
    print(f"checkpoint on : {on['median_s']*1000:.2f} ms/step  {on['tokens_per_s']:.0f} tok/s")
    print(f"checkpoint off: {off['median_s']*1000:.2f} ms/step  {off['tokens_per_s']:.0f} tok/s")
    print(f"speedup off/on: {off['tokens_per_s']/on['tokens_per_s']:.2f}x")

    model = _tiny_model()
    model.use_ckpt = True
    with torch.no_grad():
        loss_on, _ = client._fwl(model, x, y, 16, False, None)
    model.use_ckpt = False
    with torch.no_grad():
        loss_off, _ = client._fwl(model, x, y, 16, False, None)
    print(f"forward loss delta (on-off): {abs(float(loss_on)-float(loss_off)):.8f}")
    print("scope: tiny CPU model (2 layers, dim 32), direction only")


if __name__ == "__main__":
    main()
