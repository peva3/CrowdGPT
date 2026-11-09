#!/usr/bin/env python3
"""Measure the full 658M model with a rank-16 LoRA adapter on one GPU.

This is a memory and speed probe, not a quality benchmark. It builds random
weights, freezes the base model, attaches rank-16 adapters, and runs a few
AdamW steps. Run with the project virtualenv:

    python tools/bench_lora_gpu.py --seq 512 --steps 5
    python tools/bench_lora_gpu.py --seq 2048 --steps 3

Measured on an RTX 4060 Ti 16 GB (2026-11-09):
    seq 512,  checkpoint on : 3.54 GB peak, 0.307 s/step, 1667 tok/s
    seq 512,  checkpoint off: 5.01 GB peak, 0.228 s/step, 2251 tok/s
    seq 2048, checkpoint on : 3.79 GB peak, ~1.0-1.3 s/step
The base model has 657,787,392 parameters; the adapters here total 8,552,448.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--checkpoint", choices=("on", "off"), default="on")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    torch.manual_seed(1)
    device = torch.device("cuda")
    client.ENGRAM_ACTIVE = False
    client.LOSS_CHUNK = 128
    model = client.SotaGPT("CUDA", device, gpu_layers=client.N_LAYERS,
                           bs=1, sl=args.seq)
    client._apply_lora(model, args.rank, 32.0)
    model.use_ckpt = args.checkpoint == "on"
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    tokens = torch.randint(0, client.VOCAB_SIZE, (1, args.seq), device=device)
    labels = torch.randint(0, client.VOCAB_SIZE, (1, args.seq), device=device)

    # Warm up allocator and kernels before the measured run.
    opt.zero_grad(set_to_none=True)
    loss, _ = client._fwl(model, tokens, labels, args.seq, True, torch.bfloat16)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(args.steps):
        opt.zero_grad(set_to_none=True)
        loss, _ = client._fwl(model, tokens, labels, args.seq, True, torch.bfloat16)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"gpu={torch.cuda.get_device_name(0)} seq={args.seq} rank={args.rank} "
          f"checkpoint={args.checkpoint}")
    print(f"trainable_params={sum(p.numel() for p in trainable)}")
    print(f"seconds_per_step={elapsed/args.steps:.4f} tokens_per_s={args.seq*args.steps/elapsed:.1f} "
          f"peak_GB={peak:.2f}")


if __name__ == "__main__":
    main()
